#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <vector>
#include <sstream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier{
	public:
		Classifier(const string& model_file,
					const string& traind_file,
					const string& mean_file,
					const string& label_flile);
		// 均值处理
		void SetMean(const string& mean_file);
		std::vector<float> Predict(const cv::Mat& img);
		void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
		void WrapInputLayer(std::vector<cv::Mat>* input_channels);
		std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

	private:
		shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;  // 图像的大小
		int num_channels_ ;       // 通道数
		cv::Mat mean_;
		std::vector<string> labels_;     // 标签
};

Classifier::Classifier(const string& model_file,
						const string& traind_file,
						const string& mean_file,
						const string& label_flile){

	Caffe::set_mode(Caffe::CPU);

	/*初始化网络*/
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(traind_file);

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	std::cout<<input_layer->width()<<std::endl;
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());


	//得到均值图像
	SetMean(mean_file);
	std::ifstream labels(label_flile.c_str());
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));



}
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}
/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}
void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels){
	cv::Mat sample;
	if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
  	else
  		sample = img;

	cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3){
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
	cv::Mat sample_normalized;
    sample_normalized  = sample_float - mean_;// cv::subtract(sample_float, mean_, sample_normalized);
	
	cv::split(sample_normalized, *input_channels);


}
std::vector<float> Classifier::Predict(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();


    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}
void Classifier::SetMean(const string& mean_file){
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
	// 转换为Blob<float>
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	cv::Scalar cs;
	for(int i = 0; i < mean_blob.shape(1); ++i){
		// 提取单通道
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		cs[i] = cv::mean(channel)[0];
		
		// 指针移动 到下个通道
		data += mean_blob.height() * mean_blob.width();
	}

	mean_ = cv::Mat(input_geometry_, CV_32FC3, cs);

}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
    std::vector<float> output = Predict(img);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}
#endif