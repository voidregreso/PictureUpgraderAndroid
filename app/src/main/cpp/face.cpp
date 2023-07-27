#include "include/face.h"

namespace wsdsb {

    Face::Face(): prob_threshold(0.5), nms_threshold(0.65) {
        face_template.push_back(cv::Point2f(192.98138, 239.94708));
        face_template.push_back(cv::Point2f(318.90277, 240.1936));
        face_template.push_back(cv::Point2f(256.63416, 314.01935));
        face_template.push_back(cv::Point2f(201.26117, 371.41043));
        face_template.push_back(cv::Point2f(313.08905, 371.15118));
    }
    Face::~Face() {
        net_.clear();
    }

    static inline float intersection_area(const Object_t &a, const Object_t &b) {
        return (a.rect & b.rect).area();
    }

    static void qsort_descent_inplace(std::vector<Object_t>& faceobjects) {
        std::sort(faceobjects.begin(), faceobjects.end(), [](const Object_t& a, const Object_t& b) {
            return a.score > b.score;
        });
    }

    static void nms_sorted_bboxes(const std::vector<Object_t> &faceobjects, std::vector<int> &picked, float nms_threshold) {
        picked.clear();
        const int n = faceobjects.size();
        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) areas[i] = faceobjects[i].rect.area();
        for (int i = 0; i < n; i++) {
            const Object_t &a = faceobjects[i];
            int keep = 1;
            for (int j : picked) {
                const Object_t &b = faceobjects[j];
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[j] - inter_area;
                if (inter_area / union_area > nms_threshold) keep = 0;
            }
            if (keep) picked.push_back(i);
        }
    }

    static inline float sigmoid(float x) {
        return 1.f / (1.f + exp(-x));
    }

    static void generate_proposals(const ncnn::Mat& anchors, int stride, int pad_h, int pad_w, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object_t>& objects) {
        const int num_grid_x = pad_w / stride;
        const int num_grid_y = pad_h / stride;
        const int num_class = feat_blob.w - 15 - 5;
        const int num_anchors = anchors.w / 2;

        for (int q = 0; q < num_anchors; q++) {
            const float anchor_w = anchors[q * 2];
            const float anchor_h = anchors[q * 2 + 1];
            const ncnn::Mat feat = feat_blob.channel(q);
            for (int i = 0; i < num_grid_y; i++) {
                for (int j = 0; j < num_grid_x; j++) {
                    const float* featptr = feat.row(i * num_grid_x + j);
                    float box_confidence = sigmoid(featptr[4]);
                    if (box_confidence >= prob_threshold) {
                        // find class index with max class score
                        float class_score = *std::max_element(featptr + 5, featptr + 5 + num_class);
                        float confidence = box_confidence * sigmoid(class_score);
                        if (confidence >= prob_threshold) {
                            float dx = sigmoid(featptr[0]);
                            float dy = sigmoid(featptr[1]);
                            float dw = sigmoid(featptr[2]);
                            float dh = sigmoid(featptr[3]);
                            float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                            float pb_cy = (dy * 2.f - 0.5f + i) * stride;
                            float pb_w = pow(dw * 2.f, 2) * anchor_w;
                            float pb_h = pow(dh * 2.f, 2) * anchor_h;
                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;
                            Object_t obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.score = confidence;
                            for (int l = 0; l < 5; l++) {
                                float x = (featptr[3 * l + 6] * 2 - 0.5 + j) * stride;
                                float y = (featptr[3 * l + 1 + 6] * 2 - 0.5 + i) * stride;
                                obj.pts.push_back(cv::Point2f(x, y));
                            }
                            objects.push_back(obj);
                        }
                    }
                }
            }
        }
    }

    int Face::Load(const std::string &model_path) {
        std::string net_param_path = model_path + "/yolov7-lite-e.param";
        std::string net_model_path = model_path + "/yolov7-lite-e.bin";
        int ret = net_.load_param(net_param_path.c_str());
        if (ret < 0) {
            fprintf(stderr, "open param file %s failed\n", net_param_path.c_str());
            return -1;
        }
        ret = net_.load_model(net_model_path.c_str());
        if (ret < 0) {
            fprintf(stderr, "open bin file %s failed\n", net_model_path.c_str());
            return -1;
        }
        output_indexes_.resize(3);
        const auto &blobs = net_.blobs();
        for (int i = 0; i != blobs.size(); ++i) {
            const auto &b = blobs[i];
            if (b.name == "stride_8")
                output_indexes_[0] = i;
            if (b.name == "stride_16")
                output_indexes_[1] = i;
            if (b.name == "stride_32")
                output_indexes_[2] = i;
        }
        for(const auto &input : net_.input_indexes()) {
            input_indexes_.push_back(input);
        }
        return 0;
    }


    void Face::PreProcess(const void *input_data, std::vector<Tensor_t> &input_tensor) {
        const int target_size = 640;
        auto *bgr = (cv::Mat *)input_data;
        int img_w = bgr->cols;
        int img_h = bgr->rows;
        // letterbox pad to multiple of 32
        int w = img_w;
        int h = img_h;
        float scale = 1.f;
        if (w > h) {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr->data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
        // pad to target_size rectangle
        int wpad = target_size - w;
        int hpad = target_size - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
        in_pad.substract_mean_normalize(0, norm_vals_);
        Tensor_t tensor_t(in_pad);
        tensor_t.img_h = img_h;
        tensor_t.img_w = img_w;
        tensor_t.pad_h = hpad;
        tensor_t.pad_w = wpad;
        tensor_t.in_h = in_pad.h;
        tensor_t.in_w = in_pad.w;
        tensor_t.scale = scale;
        input_tensor.push_back(tensor_t);
    }

    void Face::Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor) {
        ncnn::Extractor net_ex = net_.create_extractor();
        for(int i = 0; i != input_indexes_.size(); ++i) {
            net_ex.input(input_indexes_[i], input_tensor[i].data);
        }
        for(int output_indexe : output_indexes_) {
            ncnn::Mat out;
            net_ex.extract(output_indexe, out);
            output_tensor.push_back(Tensor_t(out));
        }
    }

    void Face::AlignFace(const cv::Mat &img, Object_t &object) {
        cv::Mat affine_matrix = cv::estimateAffinePartial2D(object.pts, face_template, cv::noArray(), cv::LMEDS);
        cv::Mat cropped_face;
        cv::warpAffine(img, cropped_face, affine_matrix, cv::Size(512, 512), 1, cv::BORDER_CONSTANT, cv::Scalar(135, 133, 132));
        cv::Mat affine_matrix_inv;
        cv::invertAffineTransform(affine_matrix, affine_matrix_inv);
        affine_matrix_inv *= 2;
        affine_matrix_inv.copyTo(object.trans_inv);
        cropped_face.copyTo(object.trans_img);
    }

    void Face::generateAndInsertProposals(const std::vector<float>& anchor_values, int stride, const Tensor_t& input,
                                          const ncnn::Mat& output_data, float prob_threshold, std::vector<Object_t>& proposals) {
        ncnn::Mat anchors(anchor_values.size());
        for (size_t i = 0; i < anchor_values.size(); ++i) {
            anchors[i] = anchor_values[i];
        }
        std::vector<Object_t> objects;
        generate_proposals(anchors, stride, input.in_h, input.in_w, output_data, prob_threshold, objects);
        proposals.insert(proposals.end(), objects.begin(), objects.end());
    }

    void Face::PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result) {
        std::vector<Object_t> proposals;
        generateAndInsertProposals({ 4.f, 5.f, 6.f, 8.f, 10.f, 12.f }, 8, input_tensor[0], output_tensor[0].data, prob_threshold, proposals);
        generateAndInsertProposals({ 15.f, 19.f, 23.f, 30.f, 39.f, 52.f }, 16, input_tensor[0], output_tensor[1].data, prob_threshold, proposals);
        generateAndInsertProposals({ 72.f, 97.f, 123.f, 164.f, 209.f, 297.f }, 32, input_tensor[0], output_tensor[2].data, prob_threshold, proposals);

        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);
        int count = picked.size();
        ((PipeResult_t*)result)->face_count = count;
        for (int i = 0; i != count; ++i) {
            Object_t& obj = ((PipeResult_t*)result)->object[i] = proposals[picked[i]];
            float x0 = (obj.rect.x - (input_tensor[0].pad_w / 2)) / input_tensor[0].scale;
            float y0 = (obj.rect.y - (input_tensor[0].pad_h / 2)) / input_tensor[0].scale;
            float x1 = (obj.rect.x + obj.rect.width - (input_tensor[0].pad_w / 2)) / input_tensor[0].scale;
            float y1 = (obj.rect.y + obj.rect.height - (input_tensor[0].pad_h / 2)) / input_tensor[0].scale;
            for (int j = 0; j < 5; j++) {
                float ptx = (obj.pts[j].x - (input_tensor[0].pad_w / 2)) / input_tensor[0].scale;
                float pty = (obj.pts[j].y - (input_tensor[0].pad_h / 2)) / input_tensor[0].scale;
                obj.pts[j] = cv::Point2f(ptx, pty);
            }
            x0 = (std::max)((std::min)(x0, (float)(input_tensor[0].img_w - 1)), 0.f);
            y0 = (std::max)((std::min)(y0, (float)(input_tensor[0].img_h - 1)), 0.f);
            x1 = (std::max)((std::min)(x1, (float)(input_tensor[0].img_w - 1)), 0.f);
            y1 = (std::max)((std::min)(y1, (float)(input_tensor[0].img_h - 1)), 0.f);
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
        }
    }

    int Face::Process(const cv::Mat &input_img, void *result) {
        std::vector<Tensor_t> input_tensor;
        PreProcess((void *)&input_img, input_tensor);
        std::vector<Tensor_t> output_tensor;
        Run(input_tensor, output_tensor);
        PostProcess(input_tensor, output_tensor, result);
        for(int i = 0; i != ((PipeResult_t *)result)->face_count; ++i) {
            AlignFace(input_img, ((PipeResult_t *)result)->object[i]);
        }
        return 0;
    }
}