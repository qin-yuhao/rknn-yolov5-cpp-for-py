#ifndef _rknnPool_H
#define _rknnPool_H

#include <queue>
#include <vector>
#include <iostream>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "postprocess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ThreadPool.hpp"
#include <vector>
using cv::Mat;
using std::queue;
using std::vector;

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
void slice_img(detect_result_group_t detect_result_group, cv::Mat img, std::vector<cv::Mat> &result);


cv::Mat letterbox(cv::Mat &src, int h, int w)
{

    int in_w = src.cols; // width
    int in_h = src.rows; // height
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;

    cv::Mat resize_img;

    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;

    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(128, 128, 128));
    if (resize_img.cols != w or resize_img.rows != h)
    cv::resize(resize_img, resize_img, cv::Size(w, h));
    return resize_img;
}




class rknn_lite
{
private:
    rknn_context rkModel;
    unsigned char *model_data;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    int ret;
    int channel = 3;
    int width = 0;
    int height = 0;
public:
    Mat ori_img;
    std::vector<cv::Mat> interf();
    rknn_lite(char *dst, int n);
    ~rknn_lite();
};

rknn_lite::rknn_lite(char *model_name, int n)
{
    /* Create the neural network */
    printf("Loading mode... %s\n", model_name);
    int model_data_size = 0;
    // 读取模型文件数据
    model_data = load_model(model_name, &model_data_size);
    // 通过模型文件初始化rknn类
    ret = rknn_init(&rkModel, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    // 
    rknn_core_mask core_mask;
    if (n == 0)
        core_mask = RKNN_NPU_CORE_0;
    else if(n == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;
    int ret = rknn_set_core_mask(rkModel, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }

    // 初始化rknn类的版本
    ret = rknn_query(rkModel, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 获取模型的输入参数
    ret = rknn_query(rkModel, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 设置输入数组
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    // 设置输出数组
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // 设置输入参数
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
}

rknn_lite::~rknn_lite()
{
    ret = rknn_destroy(rkModel);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
}

std::vector<cv::Mat> rknn_lite::interf()
{
    cv::Mat img;
    
    //cv::imwrite("letterbox.jpg", ori_img);
    // 获取图像宽高
    int img_width = ori_img.cols;
    int img_height = ori_img.rows;
    if (img_width != 640 || img_height != 640)
    ori_img = letterbox(ori_img, 640, 640);
    cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB);
    

    // init rga context
    // rga是rk自家的绘图库,绘图效率高于OpenCV
    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    // You may not need resize when src resulotion equals to dst resulotion
    void *resize_buf = nullptr;
    // 如果输入图像不是指定格式
    if (img_width !=  width || img_height !=  height)
    {
        resize_buf = malloc( height *  width *  channel);
        memset(resize_buf, 0x00,  height *  width *  channel);

        src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf,  width,  height, RK_FORMAT_RGB_888);
         ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR !=  ret)
        {
            printf("%d, 目标检测check error! %s", __LINE__, imStrError((IM_STATUS) ret));
            exit(-1);
        }
        IM_STATUS STATUS = imresize(src, dst);

        cv::Mat resize_img(cv::Size( width,  height), CV_8UC3, resize_buf);
        inputs[0].buf = resize_buf;
    }
    
    else
         inputs[0].buf = (void *)img.data;
    // 设置rknn的输入数据
    rknn_inputs_set( rkModel,  io_num.n_input,  inputs);

    // 设置输出
    rknn_output outputs[ io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i <  io_num.n_output; i++)
        outputs[i].want_float = 0;
    // 调用npu进行推演
     ret = rknn_run( rkModel, NULL);
    // 获取npu的推演输出结果
     ret = rknn_outputs_get( rkModel,  io_num.n_output, outputs, NULL);

    // 总之就是绘图部分
    // post process
    // width是模型需要的输入宽度, img_width是图片的实际宽度
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    float scale_w = (float) width / img_width;
    float scale_h = (float) height / img_height;
    //printf("rknn_lite::interf 开始后处理\n");
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i <  io_num.n_output; ++i)
    {
        out_scales.push_back( output_attrs[i].scale); // push_back是vector的一个方法,用于向vector中添加元素
        out_zps.push_back( output_attrs[i].zp);
    }
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,  height,  width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    //printf("rknn_lite::interf 开始绘图\n");
    // Draw Objects
    // char text[256];
    // for (int i = 0; i < detect_result_group.count; i++)
    // {
    //     detect_result_t *det_result = &(detect_result_group.results[i]);
    //     sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100); // sprintf是C语言的一个函数,用于格式化字符串
    //     int x1 = det_result->box.left;
    //     int y1 = det_result->box.top;
    //     rectangle(ori_img, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
    //     putText(ori_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    // }
    std::vector<cv::Mat> result;
    //printf("rknn_lite::interf 开始切片\n");
    slice_img(detect_result_group, ori_img, result);

    ret = rknn_outputs_release( rkModel,  io_num.n_output, outputs);
    if (resize_buf)
    {
        free(resize_buf);
    }
    return result;
}

//切片检测到的目标图像
void slice_img(detect_result_group_t detect_result_group, cv::Mat img, std::vector<cv::Mat> &result)
{
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);//left top right bottom分别是左上角和右下角的坐标
        if (det_result->prop < BOX_THRESH)
            continue;
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        
        cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat resu = img(rect);
        //letterbox
        if (resu.cols == 0 or resu.rows == 0)
        {  
            printf("resu.cols == 0 or resu.rows == 0 图像长或宽为0,跳过\n");
            //printf("x1:%d, y1:%d, x2:%d, y2:%d\n", x1, y1, x2, y2);
            //printf("img.cols:%d, img.rows:%d\n", img.cols, img.rows);
            continue;
        }  
        resu = letterbox(resu, 224, 224);
        result.push_back(resu);
        //只取前5个
        if (i == MAX_BOX-1)
            break;
    }
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;
    
    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}


#endif