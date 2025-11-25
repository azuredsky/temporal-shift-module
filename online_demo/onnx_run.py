import numpy as np
import cv2
import os
from typing import Tuple
import io
import time
import onnx
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
import onnxruntime as ort

from mobilenet_v2_tsm import MobileNetV2

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True


def torch2onnx(torch_module: torch.nn.Module,
               torch_inputs: Tuple[torch.Tensor, ...],
               onnx_path: str):
    """Export PyTorch MobileNetV2-TSM model to ONNX."""
    torch_module.eval()

    # 先跑一遍 forward，确定输出个数
    with torch.no_grad():
        outs = torch_module(*torch_inputs)
    if isinstance(outs, (tuple, list)):
        num_outputs = len(outs)
    else:
        num_outputs = 1

    input_names = [f"i{i}" for i in range(len(torch_inputs))]
    output_names = [f"o{i}" for i in range(num_outputs)]

    with torch.no_grad():
        torch.onnx.export(
            torch_module,
            torch_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=10,
            do_constant_folding=True
        )

    # 用 onnx-simplifier 简化一下
    from onnxsim import simplify
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    assert check
    onnx.save(model_simp, onnx_path)


def get_executor(use_gpu: bool = True):
    """
    构建 onnxruntime executor

    返回
    -------
    executor: callable
        executor(inputs: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]
        inputs[0] 是当前帧 (1,3,224,224)，inputs[1:] 是隐藏状态 buffer。
    """
    torch_module = MobileNetV2(n_class=27)

    # 加载 PyTorch 权重
    ckpt_path = "mobilenetv2_jester_online.pth.tar"
    if not os.path.exists(ckpt_path):
        print('Downloading PyTorch checkpoint...')
        import urllib.request
        url = 'https://hanlab18.mit.edu/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
        urllib.request.urlretrieve(url, ckpt_path)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    torch_module.load_state_dict(state_dict)

    # 构造 dummy 输入 trace 计算图
    torch_inputs = (
        torch.rand(1, 3, 224, 224),
        torch.zeros([1, 3, 56, 56]),
        torch.zeros([1, 4, 28, 28]),
        torch.zeros([1, 4, 28, 28]),
        torch.zeros([1, 8, 14, 14]),
        torch.zeros([1, 8, 14, 14]),
        torch.zeros([1, 8, 14, 14]),
        torch.zeros([1, 12, 14, 14]),
        torch.zeros([1, 12, 14, 14]),
        torch.zeros([1, 20, 7, 7]),
        torch.zeros([1, 20, 7, 7]),
    )

    onnx_path = "mobilenetv2_jester_online.onnx"

    if not os.path.exists(onnx_path):
        print("Exporting ONNX model...")
        torch2onnx(torch_module, torch_inputs, onnx_path)
    else:
        print("Found existing ONNX model:", onnx_path)

    # 选择 provider
    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("Creating onnxruntime InferenceSession with providers:", providers)
    ort_sess = ort.InferenceSession(onnx_path, providers=providers)

    input_names = [inp.name for inp in ort_sess.get_inputs()]
    output_names = [out.name for out in ort_sess.get_outputs()]

    def executor(inputs: Tuple[np.ndarray, ...]):
        assert len(inputs) == len(input_names), \
            f"Expected {len(input_names)} inputs, got {len(inputs)}"

        ort_inputs = {input_names[i]: inputs[i] for i in range(len(inputs))}
        ort_outputs = ort_sess.run(output_names, ort_inputs)
        # 返回 Tuple[np.ndarray, ...]
        return tuple(ort_outputs)

    return executor


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 224, 224) 0 ~ 1.0
    return frame


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        # 老代码里是 Scale，如果你本地 torchvision 没有这个，可以换成 Resize
        self.worker = torchvision.transforms.Resize(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform_fn = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
    ])
    return transform_fn


catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


n_still_frame = 0


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'


def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    transform_fn = get_transform()
    print("Build Executor...")
    executor = get_executor(use_gpu=True)  # 如果你没有 CUDA onnxruntime，可以改成 False

    # hidden state buffers (numpy)
    buffer = (
        np.zeros((1, 3, 56, 56), dtype=np.float32),
        np.zeros((1, 4, 28, 28), dtype=np.float32),
        np.zeros((1, 4, 28, 28), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 12, 14, 14), dtype=np.float32),
        np.zeros((1, 12, 14, 14), dtype=np.float32),
        np.zeros((1, 20, 7, 7), dtype=np.float32),
        np.zeros((1, 20, 7, 7), dtype=np.float32),
    )

    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    i_frame = -1
    current_time = 1.0

    print("Ready!")
    while True:
        i_frame += 1
        ret, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if not ret:
            break

        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()

            pil_img = Image.fromarray(img).convert('RGB')
            img_tran = transform_fn([pil_img])
            input_var = img_tran.view(1, 3, img_tran.size(1), img_tran.size(2))
            img_np = input_var.detach().cpu().numpy().astype(np.float32)

            inputs = (img_np,) + buffer
            outputs = executor(inputs)

            feat = outputs[0]       # (1, 27)
            buffer = outputs[1:]    # updated buffers

            if SOFTMAX_THRES > 0:
                feat_np = feat.reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat, axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat, axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat)
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            t2 = time.time()
            print(f"{index} {catigories[idx]}")
            current_time = t2 - t1

        img_show = cv2.resize(img, (640, 480))
        img_show = img_show[:, ::-1]
        height, width, _ = img_show.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time if current_time > 0 else 0.0),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img_cat = np.concatenate((img_show, label), axis=0)
        cv2.imshow(WINDOW_NAME, img_cat)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
