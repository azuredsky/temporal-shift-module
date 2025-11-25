#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

static const float SOFTMAX_THRES = 0.0f;
static const bool HISTORY_LOGIT = true;
static const bool REFINE_OUTPUT = true;

// ===== 新增：最终后处理平滑开关和窗口大小 =====
static const bool POST_SMOOTH = true;
static const int POST_SMOOTH_WIN = 5;

static const char *WINDOW_NAME = "Video Gesture Recognition";

// 类别列表（和 Python 版保持一致）
static const std::vector<std::string> kCategories = {
    "Doing other things",            // 0
    "Drumming Fingers",              // 1
    "No gesture",                    // 2
    "Pulling Hand In",               // 3
    "Pulling Two Fingers In",        // 4
    "Pushing Hand Away",             // 5
    "Pushing Two Fingers Away",      // 6
    "Rolling Hand Backward",         // 7
    "Rolling Hand Forward",          // 8
    "Shaking Hand",                  // 9
    "Sliding Two Fingers Down",      // 10
    "Sliding Two Fingers Left",      // 11
    "Sliding Two Fingers Right",     // 12
    "Sliding Two Fingers Up",        // 13
    "Stop Sign",                     // 14
    "Swiping Down",                  // 15
    "Swiping Left",                  // 16
    "Swiping Right",                 // 17
    "Swiping Up",                    // 18
    "Thumb Down",                    // 19
    "Thumb Up",                      // 20
    "Turning Hand Clockwise",        // 21
    "Turning Hand Counterclockwise", // 22
    "Zooming In With Full Hand",     // 23
    "Zooming In With Two Fingers",   // 24
    "Zooming Out With Full Hand",    // 25
    "Zooming Out With Two Fingers"   // 26
};

// TSM 隐状态的 shape 定义
struct StateShape
{
    int c;
    int h;
    int w;
};

static const std::vector<StateShape> kStateShapes = {
    {3, 56, 56},
    {4, 28, 28},
    {4, 28, 28},
    {8, 14, 14},
    {8, 14, 14},
    {8, 14, 14},
    {12, 14, 14},
    {12, 14, 14},
    {20, 7, 7},
    {20, 7, 7}};

// 简单工具函数：求 argmax
int argmax(const std::vector<float> &v)
{
    int idx = 0;
    float maxv = v[0];
    for (int i = 1; i < (int)v.size(); ++i)
    {
        if (v[i] > maxv)
        {
            maxv = v[i];
            idx = i;
        }
    }
    return idx;
}

// 处理输出：和 Python 版 process_output 一致
int processOutput(int idx_, std::vector<int> &history)
{
    if (!REFINE_OUTPUT)
    {
        history.push_back(idx_);
        if (history.size() > 20)
            history.erase(history.begin(), history.end() - 20);
        return idx_;
    }

    const int max_hist_len = 20;

    // mask out illegal action
    if (idx_ == 7 || idx_ == 8 || idx_ == 21 || idx_ == 22 || idx_ == 3)
    {
        idx_ = history.back();
    }

    // use only single no action class
    if (idx_ == 0)
    {
        idx_ = 2;
    }

    // history smoothing
    if (idx_ != history.back())
    {
        if (history.size() >= 2)
        {
            int last = history[history.size() - 1];
            int last2 = history[history.size() - 2];
            if (!(last == last2))
            {
                idx_ = last;
            }
        }
    }

    history.push_back(idx_);
    if ((int)history.size() > max_hist_len)
    {
        history.erase(history.begin(), history.end() - max_hist_len);
    }

    return history.back();
}

// 预处理：Scale(短边=256) + CenterCrop(224) + Normalize（mean/std） + NCHW
void preprocessFrame(const cv::Mat &frameBGR, float *outputData,
                     int inputH = 224, int inputW = 224)
{
    cv::Mat rgb;
    cv::cvtColor(frameBGR, rgb, cv::COLOR_BGR2RGB);

    // Scale shorter side to 256
    int srcH = rgb.rows;
    int srcW = rgb.cols;
    int shortSide = std::min(srcH, srcW);
    float scale = 256.0f / shortSide;
    int newH = int(srcH * scale + 0.5f);
    int newW = int(srcW * scale + 0.5f);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(newW, newH));

    // Center crop 224x224
    int x1 = (newW - inputW) / 2;
    int y1 = (newH - inputH) / 2;
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    int x2 = std::min(x1 + inputW, resized.cols);
    int y2 = std::min(y1 + inputH, resized.rows);

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat crop = resized(roi);

    if (crop.cols != inputW || crop.rows != inputH)
    {
        cv::resize(crop, crop, cv::Size(inputW, inputH));
    }

    // Normalize to [0,1], then (x - mean) / std, output NCHW
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    // layout: outputData[c * H * W + y * W + x]
    for (int y = 0; y < inputH; ++y)
    {
        const cv::Vec3b *rowPtr = crop.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputW; ++x)
        {
            const cv::Vec3b &px = rowPtr[x];
            for (int c = 0; c < 3; ++c)
            {
                float v = static_cast<float>(px[c]) / 255.0f; // 0-1
                v = (v - mean[2 - c]) / std[2 - c];           // 注意：OpenCV Vec3b 是 [B,G,R]
                // PyTorch 通常是 RGB 顺序，这里转换时我们在 cvtColor 用了 BGR->RGB，
                // 所以这里也可以直接用 mean/std[c]。保险起见可按需调整。
                outputData[c * inputH * inputW + y * inputW + x] = v;
            }
        }
    }
}

// Softmax（可选）
void softmaxInplace(std::vector<float> &v)
{
    float maxv = v[0];
    for (float x : v)
        maxv = std::max(maxv, x);
    float sum = 0.0f;
    for (float &x : v)
    {
        x = std::exp(x - maxv);
        sum += x;
    }
    for (float &x : v)
        x /= sum;
}

int main()
{
    std::string modelPath = "mobilenetv2_jester_online_fp16.mnn"; // 先用 MNNConvert 得到这个模型

    // 1. 加载 MNN 模型
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!net)
    {
        std::cerr << "Failed to load MNN model: " << modelPath << std::endl;
        return -1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU; // 需要 GPU / Vulkan 可以改这里
    config.numThread = 4;
    auto session = net->createSession(config);
    if (!session)
    {
        std::cerr << "Failed to create MNN session." << std::endl;
        return -1;
    }

    // 2. 获取输入/输出（假设名字是 i0..i10, o0..o10）
    auto inputMap = net->getSessionInputAll(session);
    auto outputMap = net->getSessionOutputAll(session);

    // 当前帧输入 i0
    MNN::Tensor *inputFrame = nullptr;
    if (inputMap.count("i0") > 0)
    {
        inputFrame = inputMap["i0"];
    }
    else
    {
        std::cerr << "Input i0 not found in model." << std::endl;
        return -1;
    }

    // 隐状态输入 i1..i10
    std::vector<MNN::Tensor *> stateInputs(10, nullptr);
    for (int i = 0; i < 10; ++i)
    {
        std::string name = "i" + std::to_string(i + 1);
        if (inputMap.count(name) == 0)
        {
            std::cerr << "Input " << name << " not found in model." << std::endl;
            return -1;
        }
        stateInputs[i] = inputMap[name];
    }

    // 输出：o0 = logits, o1..o10 = 新的 state
    MNN::Tensor *logitsTensor = nullptr;
    if (outputMap.count("o0") > 0)
    {
        logitsTensor = outputMap["o0"];
    }
    else
    {
        std::cerr << "Output o0 not found in model." << std::endl;
        return -1;
    }
    std::vector<MNN::Tensor *> stateOutputs(10, nullptr);
    for (int i = 0; i < 10; ++i)
    {
        std::string name = "o" + std::to_string(i + 1);
        if (outputMap.count(name) == 0)
        {
            std::cerr << "Output " << name << " not found in model." << std::endl;
            return -1;
        }
        stateOutputs[i] = outputMap[name];
    }

    // host 侧 Tensor 用于复制数据
    std::shared_ptr<MNN::Tensor> inputFrameHost(new MNN::Tensor(inputFrame, MNN::Tensor::CAFFE));
    std::vector<std::shared_ptr<MNN::Tensor>> stateInputHosts;
    std::vector<std::shared_ptr<MNN::Tensor>> stateOutputHosts;
    stateInputHosts.reserve(10);
    stateOutputHosts.reserve(10);

    for (int i = 0; i < 10; ++i)
    {
        stateInputHosts.emplace_back(new MNN::Tensor(stateInputs[i], MNN::Tensor::CAFFE));
        stateOutputHosts.emplace_back(new MNN::Tensor(stateOutputs[i], MNN::Tensor::CAFFE));
    }

    std::shared_ptr<MNN::Tensor> logitsHost(new MNN::Tensor(logitsTensor, MNN::Tensor::CAFFE));

    // 3. 初始化隐藏状态 buffer（全部为零）
    std::vector<std::vector<float>> stateBuffers(10);
    for (int i = 0; i < 10; ++i)
    {
        const auto &s = kStateShapes[i];
        int sz = s.c * s.h * s.w;
        stateBuffers[i].assign(sz, 0.0f);
    }

    // 4. OpenCV 摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 640, 480);
    cv::moveWindow(WINDOW_NAME, 0, 0);

    bool fullScreen = false;
    int idx = 0;                    // 原始预测（已经经过 processOutput）
    int displayIdx = 0;             // ===== 新增：最终平滑后用于显示的结果 =====
    std::vector<int> history = {2}; // 初始类别 history
    std::vector<std::vector<float>> historyLogit;
    historyLogit.reserve(12);

    // ===== 新增：最终平滑用的历史队列 =====
    std::vector<int> postHistory;

    int frameIndex = -1;
    double currentTime = 1.0; // 避免最开始除零

    std::cout << "Ready!" << std::endl;

    while (true)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::cerr << "Failed to read frame from camera." << std::endl;
            break;
        }
        frameIndex += 1;

        auto t1 = std::chrono::high_resolution_clock::now();

        if (frameIndex % 2 == 0)
        {
            // ---- 预处理当前帧 ----
            float *inputData = inputFrameHost->host<float>();
            preprocessFrame(frame, inputData, 224, 224);
            inputFrame->copyFromHostTensor(inputFrameHost.get());

            // ---- 写入隐藏状态 ----
            for (int i = 0; i < 10; ++i)
            {
                float *bufPtr = stateInputHosts[i]->host<float>();
                const auto &s = kStateShapes[i];
                int sz = s.c * s.h * s.w;
                memcpy(bufPtr, stateBuffers[i].data(), sz * sizeof(float));
                stateInputs[i]->copyFromHostTensor(stateInputHosts[i].get());
            }

            // ---- 运行推理 ----
            net->runSession(session);

            // ---- 取出 logits ----
            logitsTensor->copyToHostTensor(logitsHost.get());
            int numClasses = logitsHost->elementSize();
            std::vector<float> logits(numClasses);
            memcpy(logits.data(), logitsHost->host<float>(), numClasses * sizeof(float));

            int idx_ = idx;

            if (SOFTMAX_THRES > 0.0f)
            {
                std::vector<float> probs = logits;
                softmaxInplace(probs);
                float maxProb = *std::max_element(probs.begin(), probs.end());

                if (maxProb > SOFTMAX_THRES)
                {
                    idx_ = argmax(logits);
                }
                else
                {
                    idx_ = idx; // 保持上一帧
                }
            }
            else
            {
                idx_ = argmax(logits);
            }

            // ---- logit history 平滑 ----
            if (HISTORY_LOGIT)
            {
                historyLogit.push_back(logits);
                if ((int)historyLogit.size() > 12)
                {
                    historyLogit.erase(historyLogit.begin(),
                                       historyLogit.begin() + (historyLogit.size() - 12));
                }
                // 求平均 logit
                std::vector<float> avgLogit(numClasses, 0.0f);
                for (const auto &v : historyLogit)
                {
                    for (int k = 0; k < numClasses; ++k)
                    {
                        avgLogit[k] += v[k];
                    }
                }
                for (float &x : avgLogit)
                    x /= historyLogit.size();
                idx_ = argmax(avgLogit);
            }

            // ---- history smoothing （老的 processOutput）----
            idx = processOutput(idx_, history);

            // ===== 新增：最终后处理平滑（最近 POST_SMOOTH_WIN 帧投票） =====
            if (POST_SMOOTH)
            {
                postHistory.push_back(idx);
                if ((int)postHistory.size() > POST_SMOOTH_WIN)
                {
                    postHistory.erase(postHistory.begin(),
                                      postHistory.begin() + (postHistory.size() - POST_SMOOTH_WIN));
                }

                // 统计窗口内各类别出现次数
                std::map<int, int> cnt;
                for (int v : postHistory)
                {
                    cnt[v]++;
                }
                int bestLabel = idx;
                int bestCount = 0;
                for (auto &p : cnt)
                {
                    if (p.second > bestCount)
                    {
                        bestCount = p.second;
                        bestLabel = p.first;
                    }
                }
                displayIdx = bestLabel;
            }
            else
            {
                displayIdx = idx;
            }

            // ---- 更新隐藏状态 buffer ----
            for (int i = 0; i < 10; ++i)
            {
                stateOutputs[i]->copyToHostTensor(stateOutputHosts[i].get());
                float *outPtr = stateOutputHosts[i]->host<float>();
                const auto &s = kStateShapes[i];
                int sz = s.c * s.h * s.w;
                stateBuffers[i].assign(outPtr, outPtr + sz);
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            currentTime = std::chrono::duration<double>(t2 - t1).count();

            std::cout << frameIndex << " " << kCategories[displayIdx] << std::endl;
        }

        // ---- 画 UI ----
        cv::Mat show;
        cv::resize(frame, show, cv::Size(640, 480));
        cv::flip(show, show, 1); // Python 版是 img[:, ::-1]，这里用水平翻转

        int height = show.rows;
        int width = show.cols;

        cv::Mat label(height / 10, width, CV_8UC3, cv::Scalar(255, 255, 255));

        cv::putText(label,
                    "Prediction: " + kCategories[displayIdx], // 使用平滑后的 displayIdx
                    cv::Point(0, height / 16),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 0, 0),
                    2);

        double fps = (currentTime > 0.0) ? (1.0 / currentTime) : 0.0;
        char fpsText[64];
        snprintf(fpsText, sizeof(fpsText), "%.1f Vid/s", fps);

        cv::putText(label,
                    fpsText,
                    cv::Point(width - 170, height / 16),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 0, 0),
                    2);

        cv::Mat cat;
        cv::vconcat(show, label, cat);

        cv::imshow(WINDOW_NAME, cat);

        int key = cv::waitKey(1);
        if ((key & 0xFF) == 'q' || key == 27)
        { // ESC
            break;
        }
        else if ((key & 0xFF) == 'f' || (key & 0xFF) == 'F')
        {
            fullScreen = !fullScreen;
            if (fullScreen)
            {
                cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            }
            else
            {
                cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
