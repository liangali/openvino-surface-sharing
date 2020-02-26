#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <fcntl.h>
#include <new>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <memory>
#include <iomanip>

#include <inference_engine.hpp>
#include <ie_compound_blob.h>

#include <gpu/gpu_context_api_va.hpp>
#include <cldnn/cldnn_config.hpp>

#include "va/va.h"
#include "va/va_drm.h"

#include "video.h"
#include "classification_results.h"

using namespace InferenceEngine;

VADisplay va_dpy = NULL;
int va_fd = -1;

bool dump_decode_output = false;

const std::string input_model = "/home/fresh/data/model/resnet-50/resnet-50-128.xml";

void setBatchSize(CNNNetwork& network, size_t batch) {
    ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
    for (auto& shape : inputShapes) {
        auto& dims = shape.second;
        if (dims.empty()) {
            throw std::runtime_error("Network's input shapes have empty dimensions");
        }
        dims[0] = batch;
    }
    network.reshape(inputShapes);
}

int initVA()
{
    VADisplay disp;
    VAStatus va_res = VA_STATUS_SUCCESS;
    int major_version = 0, minor_version = 0;

    int adapter_num = 0;
    char adapterpath[256];
    snprintf(adapterpath,sizeof(adapterpath),"/dev/dri/renderD%d", adapter_num + 128);

    va_fd = open(adapterpath, O_RDWR);
    if (va_fd < 0) {
        printf("ERROR: failed to open adapter in %s\n", adapterpath);
        return -1;
    }

    va_dpy = vaGetDisplayDRM(va_fd);
    if (!va_dpy) {
        close(va_fd);
        va_fd = -1;
        printf("ERROR: failed in vaGetDisplayDRM\n");
        return -1;
    }

    va_res = vaInitialize(va_dpy, &major_version, &minor_version);
    if (VA_STATUS_SUCCESS != va_res) {
        close(va_fd);
        va_fd = -1;
        printf("ERROR: failed in vaInitialize with err = %d\n", va_res);
        return -1;
    }

    printf("INFO: vaInitialize done\n");

    return 0;
}

#define CHECK_VASTATUS(va_status,func)                                    \
if (va_status != VA_STATUS_SUCCESS) {                                     \
    fprintf(stderr,"%s:%s (%d) failed, exit\n", __func__, func, __LINE__); \
    exit(1);                                                              \
}

int decodeFrame(VASurfaceID& frame)
{
    VAEntrypoint entrypoints[5];
    int num_entrypoints, vld_entrypoint;
    VAConfigAttrib attrib;
    VAConfigID config_id;
    VASurfaceID surface_id[RT_NUM];
    VAContextID context_id;
    VABufferID pic_param_buf, iqmatrix_buf, slice_param_buf, slice_data_buf;
    int major_ver, minor_ver;
    VAStatus va_status;
    int putsurface=0;

    va_status = vaQueryConfigEntrypoints(va_dpy, VAProfileH264Main, entrypoints, 
                             &num_entrypoints);
    CHECK_VASTATUS(va_status, "vaQueryConfigEntrypoints");

    for	(vld_entrypoint = 0; vld_entrypoint < num_entrypoints; vld_entrypoint++) {
        if (entrypoints[vld_entrypoint] == VAEntrypointVLD)
            break;
    }
    if (vld_entrypoint == num_entrypoints) {
        printf("ERROR: not find AVC VLD entry point\n");
        return -1;
    }

    /* find out the format for the render target */
    attrib.type = VAConfigAttribRTFormat;
    vaGetConfigAttributes(va_dpy, VAProfileH264Main, VAEntrypointVLD, &attrib, 1);
    if ((attrib.value & VA_RT_FORMAT_YUV420) == 0) {
        printf("ERROR: not find desired YUV420 RT format\n");
        return -1;
    }

    va_status = vaCreateConfig(va_dpy, VAProfileH264Main, VAEntrypointVLD, &attrib, 1, &config_id);
    CHECK_VASTATUS(va_status, "vaQueryConfigEntrypoints");

    for (size_t i = 0; i < RT_NUM; i++)
    {
        va_status = vaCreateSurfaces(va_dpy, VA_RT_FORMAT_YUV420, CLIP_WIDTH, CLIP_HEIGHT,  &surface_id[i], 1, NULL, 0 );
        CHECK_VASTATUS(va_status, "vaCreateSurfaces");
    }

    /* Create a context for this decode pipe */
    va_status = vaCreateContext(va_dpy, config_id, CLIP_WIDTH, ((CLIP_HEIGHT+15)/16)*16, VA_PROGRESSIVE, surface_id, RT_NUM, &context_id);
    CHECK_VASTATUS(va_status, "vaCreateContext");

    va_status = vaCreateBuffer(va_dpy, context_id, VAPictureParameterBufferType, pic_size, 1, &pic_param, &pic_param_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VAPictureParameterBufferType");
    va_status = vaCreateBuffer(va_dpy, context_id, VAIQMatrixBufferType, iq_size, 1, &iq_matrix, &iqmatrix_buf );
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VAIQMatrixBufferType");
    va_status = vaCreateBuffer(va_dpy, context_id, VASliceParameterBufferType, slc_size, 1, &slc_param, &slice_param_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VASliceParameterBufferType");
    va_status = vaCreateBuffer(va_dpy, context_id, VASliceDataBufferType, bs_size, 1, bs_data, &slice_data_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VASliceDataBufferType");

    /* send decode workload to GPU */
    va_status = vaBeginPicture(va_dpy, context_id, surface_id[RT_ID]);
    CHECK_VASTATUS(va_status, "vaBeginPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &pic_param_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &iqmatrix_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &slice_param_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &slice_data_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaEndPicture(va_dpy,context_id);
    CHECK_VASTATUS(va_status, "vaEndPicture");

    va_status = vaSyncSurface(va_dpy, surface_id[RT_ID]);
    CHECK_VASTATUS(va_status, "vaSyncSurface");

    frame = surface_id[RT_ID];

    if (dump_decode_output)
    {
        VAImage output_image;
        va_status = vaDeriveImage(va_dpy, surface_id[RT_ID], &output_image);
        CHECK_VASTATUS(va_status, "vaDeriveImage");

        void *out_buf = nullptr;
        va_status = vaMapBuffer(va_dpy, output_image.buf, &out_buf);
        CHECK_VASTATUS(va_status, "vaMapBuffer");

        FILE *fp = fopen("out.nv12", "wb+");
        // dump y_plane
        char *y_buf = (char*)out_buf;
        int y_pitch = output_image.pitches[0];
        for (size_t i = 0; i < CLIP_HEIGHT; i++)
        {
            fwrite(y_buf + y_pitch*i, CLIP_WIDTH, 1, fp);
        }
        // dump uv_plane
        char *uv_buf = (char*)out_buf + output_image.offsets[1];
        int uv_pitch = output_image.pitches[1];
        for (size_t i = 0; i < CLIP_HEIGHT/2; i++)
        {
            fwrite(uv_buf + uv_pitch*i, CLIP_WIDTH, 1, fp);
        }

        // use below command line to convert nv12 surface as bmp image
        /* ffmpeg -s 224x224 -pix_fmt nv12 -f rawvideo -i out.nv12 out.bmp */

        fclose(fp);
    }

    vaDestroyConfig(va_dpy,config_id);
    vaDestroyContext(va_dpy,context_id);

    return 0;
}

int main (int argc, char **argv) {
    if (argc == 2 && *argv[1] == 'd')
    {
        dump_decode_output = true;
    }
    Core ie;
    const std::string device_name = "GPU";
    CNNNetwork network = ie.ReadNetwork(input_model);
    setBatchSize(network, 1);

    // set input info
    if (network.getInputsInfo().empty()) {
        std::cerr << "Network inputs info is empty" << std::endl;
        return -1;
    }
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;
    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);
    input_info->getPreProcess().setColorFormat(ColorFormat::NV12);

    // set output info
    if (network.getOutputsInfo().empty()) {
        std::cerr << "Network outputs info is empty" << std::endl;
        return -1;
    }
    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);

    if (initVA()) {
        printf("ERROR: initVA failed!\n");
        return -1;
    }

    // get NV12 surface from HW decode output
    VASurfaceID va_frame;
    if(decodeFrame(va_frame)) {
        printf("ERROR: decode failed\n");
        return -1;
    }

    auto shared_va_context = gpu::make_shared_context(ie, device_name, va_dpy);
    ExecutableNetwork executable_network = ie.LoadNetwork(network, shared_va_context,
        {{ CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS, PluginConfigParams::YES } });

    InferRequest infer_request = executable_network.CreateInferRequest();

    //wrap decoder output into RemoteBlobs and set it as inference input
    auto nv12_blob = gpu::make_shared_blob_nv12(CLIP_HEIGHT, CLIP_WIDTH, shared_va_context, va_frame);
    infer_request.SetBlob(input_name, nv12_blob);

    // Do inference
    infer_request.Infer();

    Blob::Ptr output = infer_request.GetBlob(output_name);
    ClassificationResult classificationResult(output, {"vaapi-nv12"});
    classificationResult.print();

    printf("done!\n");
    return 0;
}