#include <iostream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <net.h>
#include <layer.h>
#include <omp.h>
#include <include/colornet.h>

class Sig17Slice : public ncnn::Layer
{
public:
    Sig17Slice() {
        one_blob_only = true;
    }

    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;
                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Sig17Slice)

int colorization(const cv::Mat &bgr, const cv::Mat &out_image, const std::string &model_path) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);
    const int W_in = 256;
    const int H_in = 256;
    cv::Mat Base_img, lab, L, input_img;
    if (net.load_param((model_path +
        "/siggraph17_color_sim.param").c_str())) return -1;
    if (net.load_model((model_path +
        "/siggraph17_color_sim.bin").c_str())) return -1;
    Base_img = bgr.clone();
    // normalize levels
    Base_img.convertTo(Base_img, CV_32F, 1.0 / 255);
    // Convert BGR to LAB color space format
    cvtColor(Base_img, lab, cv::COLOR_BGR2Lab);
    // Extract L channel
    cv::extractChannel(lab, L, 0);
    // Resize to input shape 256x256
    resize(L, input_img, cv::Size(W_in, H_in));
    // convert to NCNN::MAT
    ncnn::Mat in_LAB_L(input_img.cols, input_img.rows, 1, (void *)input_img.data);
    in_LAB_L = in_LAB_L.clone();
    ncnn::Extractor ex = net.create_extractor();
    // set input, output lyers
    ex.input("input", in_LAB_L);
    // inference network
    ncnn::Mat out;
    ex.extract("out_ab", out);
    // create LAB material
    cv::Mat colored_LAB(out.h, out.w, CV_32FC2);
    // Extract ab channels from ncnn:Mat out
    memcpy((uchar *)colored_LAB.data, out.data, out.w * out.h * 2 * sizeof(float));
    // get separsted LAB channels a&b
    cv::Mat a(out.h, out.w, CV_32F, (float *)out.data);
    cv::Mat b(out.h, out.w, CV_32F, (float *)out.data + out.w * out.h);
    // Resize a, b channels to original image size
    cv::resize(a, a, Base_img.size());
    cv::resize(b, b, Base_img.size());
    // merge channels, and convert back to BGR
    cv::Mat color, chn[] = {L, a, b};
    cv::merge(chn, 3, lab);
    cvtColor(lab, color, cv::COLOR_Lab2BGR);
    //normalize values to 0->255
    color.convertTo(color, CV_8UC3, 255);
    color.copyTo(out_image);
    return 0;
}