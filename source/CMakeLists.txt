cmake_minimum_required(VERSION 3.4)
project (share_nv12)

option(CLDNN "build CLDNN" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O0")

if (CLDNN)
    message("build clDNN version")
    add_definitions(-DCLDNN_VER)
    set(InferenceEngine_DIR "${CMAKE_CURRENT_LIST_DIR}/../../build")

    find_package(InferenceEngine 2.1 QUIET)
    if (NOT(InferenceEngine_FOUND))
        message(FATAL_ERROR "InferenceEngine not found")
    endif()

    include_directories(
        ${InferenceEngine_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}/../../dldt/inference-engine/thirdparty/clDNN/common/intel_ocl_icd/6.3/linux/include)

    link_directories(/home/fresh/data/work/dldt_media_sharing/source/dldt/bin/intel64/Debug/lib/)

    add_executable(share_nv12 ${CMAKE_CURRENT_LIST_DIR}/main.cpp
                              ${CMAKE_CURRENT_LIST_DIR}/video.h
                              ${CMAKE_CURRENT_LIST_DIR}/classification_results.h)

    target_link_libraries(share_nv12
        ${InferenceEngine_LIBRARIES}
        OpenCL clDNNPlugin va va-drm)
else()
    message("build DNNL version")
    set (DLDT_DIR "${CMAKE_CURRENT_LIST_DIR}/../Linux")
    set (DLDT_INCLUDE_DIR "/opt/intel/openvino/inference_engine")

    set (OpenCV_INCLUDE_DIR ${DLDT_DIR}/opencv/include)
    set (OpenCV_LIB_DIR ${DLDT_DIR}/opencv/lib)
    set (OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_photo opencv_highgui opencv_imgcodecs)

    set (OPENVINO_IE_INCLUDE ${DLDT_INCLUDE_DIR}/include)
    set (OPENVINO_IE_LIB_DIR ${DLDT_DIR}/lib)
    set (OPENVINO_IE_LIBS DNNLPlugin inference_engine inference_engine_legacy)

    include_directories( ${OpenCV_INCLUDE_DIR} ${OPENVINO_IE_INCLUDE})

    link_directories(${OpenCV_LIB_DIR} ${OPENVINO_IE_LIB_DIR})

    add_executable(share_nv12 ${CMAKE_CURRENT_LIST_DIR}/main.cpp 
                              ${CMAKE_CURRENT_LIST_DIR}/video.h
                              ${CMAKE_CURRENT_LIST_DIR}/classification_results.h)

    set_target_properties(share_nv12 PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(share_nv12 PRIVATE ${OpenCV_LIBS} ${OPENVINO_IE_LIBS} OpenCL va va-drm)
endif (CLDNN)
