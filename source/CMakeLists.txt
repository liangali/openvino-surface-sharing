cmake_minimum_required(VERSION 3.4)
project (share_nv12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O0")

set(InferenceEngine_DIR "${CMAKE_CURRENT_LIST_DIR}/../../build")

find_package(InferenceEngine 2.1 QUIET)
if (NOT(InferenceEngine_FOUND))
    message(FATAL_ERROR "InferenceEngine not found")
endif()

include_directories( 
    ${InferenceEngine_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/../../dldt/inference-engine/thirdparty/clDNN/common/intel_ocl_icd/6.3/linux/include )

link_directories(/home/fresh/data/work/dldt_media_sharing/source/dldt/bin/intel64/Debug/lib/)

add_executable(share_nv12 ${CMAKE_CURRENT_LIST_DIR}/main.cpp 
                          ${CMAKE_CURRENT_LIST_DIR}/video.h
                          ${CMAKE_CURRENT_LIST_DIR}/classification_results.h)

target_link_libraries(share_nv12 
    ${InferenceEngine_LIBRARIES}
    OpenCL clDNNPlugin va va-drm
    )


