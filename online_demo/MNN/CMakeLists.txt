cmake_minimum_required(VERSION 3.10)
project(gesture_mnn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 1. 找 OpenCV ----
set(OpenCV_DIR "/media/aispeech/2874DB3B74DB0A88/work/smartCar/thirdParty/opencv_3.4.16/linux_x86-64/share/OpenCV" CACHE PATH "Path to OpenCV config directory")

find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ---- 2. 设置 / 查找 MNN 路径 ----
# 手动指定 MNN 根目录：cmake .. -DMNN_ROOT=/path/to/MNN/install
set(MNN_ROOT "/media/aispeech/2874DB3B74DB0A88/work/smartCar/thirdParty/mnn_2.4.0/linux_x86-64" CACHE PATH "Root path of MNN install (contains include/ and lib/)")

# 尝试自动查找头文件和库
if(MNN_ROOT)
    set(_MNN_HINT_INCLUDE "${MNN_ROOT}/include")
    set(_MNN_HINT_LIB "${MNN_ROOT}/lib")
else()
    set(_MNN_HINT_INCLUDE "")
    set(_MNN_HINT_LIB "")
endif()

find_path(MNN_INCLUDE_DIR
    NAMES MNN/Interpreter.hpp
    HINTS ${_MNN_HINT_INCLUDE}
    PATHS /usr/include /usr/local/include
)

find_library(MNN_LIBRARY
    NAMES MNN
    HINTS ${_MNN_HINT_LIB}
    PATHS /usr/lib /usr/local/lib
)

if(NOT MNN_INCLUDE_DIR OR NOT MNN_LIBRARY)
    message(FATAL_ERROR "Cannot find MNN. Set MNN_ROOT or MNN_INCLUDE_DIR/MNN_LIBRARY manually.")
endif()

message(STATUS "MNN include dir: ${MNN_INCLUDE_DIR}")
message(STATUS "MNN library: ${MNN_LIBRARY}")

# ---- 3. 可执行文件 ----
add_executable(gesture_mnn
    main.cpp   # 源码文件名，和你的 cpp 保持一致
)

target_include_directories(gesture_mnn
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${MNN_INCLUDE_DIR}
)

target_link_libraries(gesture_mnn
    PRIVATE
        ${OpenCV_LIBS}
        ${MNN_LIBRARY}
)

# 可选：Release 默认打开优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
