#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <ext_list.hpp>
//#include <sstream>

using namespace InferenceEngine;

class ObjectDetector {
private:
	struct CnnConfig
	{
		explicit CnnConfig(const std::string& path_to_model,
			const std::string& path_to_weights)
			: path_to_model(path_to_model), path_to_weights(path_to_weights) {}

		std::string path_to_model;
		std::string path_to_weights;
		int max_batch_size{ 1 };
	};

	struct DetectorConfig : public CnnConfig
	{
		explicit DetectorConfig(const std::string& path_to_model,
			const std::string& path_to_weights)
			: CnnConfig(path_to_model, path_to_weights) {}

		float confidence_threshold{ 0.5f };
		float increase_scale_x{ 1.f };
		float increase_scale_y{ 1.f };
		bool is_async = true;
	};

	InferenceEngine::InferRequest::Ptr request;
	DetectorConfig config_;
	InferenceEngine::Core ie_;
	std::string deviceName_;

	InferenceEngine::ExecutableNetwork net_;
	std::string input_name_;
	std::string output_name_;
	float width_ = 0;
	float height_ = 0;

public:
	ObjectDetector(const std::string& path_to_model,
		const std::string & device,
		const std::string& custom_cpu_library,
		const std::string& custom_cldnn_kernels,
		bool should_use_perf_counter) :
		config_(DetectorConfig(path_to_model, fileNameNoExt(path_to_model) + ".bin")),
		deviceName_(device)
	{
		std::set<std::string> loadedDevices;

		if (loadedDevices.find(device) != loadedDevices.end()) {
			return;
		}

		std::cout << "Loading device " << device << std::endl;
		std::cout << ie_.GetVersions(device) << std::endl;

		if ((device.find("CPU") != std::string::npos)) {
			ie_.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

			if (!custom_cpu_library.empty()) {
				auto extension_ptr = make_so_pointer<IExtension>(custom_cpu_library);
				ie_.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr), "CPU");
			}
		}
		else if (!custom_cldnn_kernels.empty()) {
			ie_.SetConfig({ {PluginConfigParams::KEY_CONFIG_FILE, custom_cldnn_kernels} }, "GPU");
		}

		if (device.find("CPU") != std::string::npos) {
			ie_.SetConfig({ {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES} }, "CPU");
		}
		else if (device.find("GPU") != std::string::npos) {
			ie_.SetConfig({ {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES} }, "GPU");
		}

		if (should_use_perf_counter)
			ie_.SetConfig({ {PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES} });

		loadedDevices.insert(device);

		CNNNetReader net_reader;
		net_reader.ReadNetwork(config_.path_to_model);
		net_reader.ReadWeights(config_.path_to_weights);

		InputsDataMap inputInfo(net_reader.getNetwork().getInputsInfo());
		for (const std::pair<std::string, InputInfo::Ptr>& input : inputInfo)
		{
			InputInfo::Ptr inputInfo = input.second;
			if (4 == inputInfo->getTensorDesc().getDims().size())
			{
				inputInfo->setPrecision(Precision::U8);
				inputInfo->getInputData()->setLayout(Layout::NCHW);
				input_name_ = input.first;
			}
		}

		OutputsDataMap outputInfo(net_reader.getNetwork().getOutputsInfo());

		DataPtr& _output = outputInfo.begin()->second;
		output_name_ = outputInfo.begin()->first;

		_output->setPrecision(Precision::FP32);
		_output->setLayout(TensorDesc::getLayoutByDims(_output->getDims()));

		net_ = ie_.LoadNetwork(net_reader.getNetwork(), deviceName_);
	};

	cv::Mat getResult(const cv::Mat &frame)
	{
		if (!request) {
			request = net_.CreateInferRequestPtr();
		}

		width_ = static_cast<float>(frame.cols);
		height_ = static_cast<float>(frame.rows);

		Blob::Ptr inputBlob = request->GetBlob(input_name_);

		matU8ToBlob<uint8_t>(frame, inputBlob);

		if (config_.is_async) {
			request->StartAsync();
		}
		else {
			request->Infer();
		}

		request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

		const float *data = request->GetBlob(output_name_)->buffer().as<float *>();

		int w = 1024;
		int h = 512;

		result_copy.copyTo(result_);

		cv::resize(result_, result_, cv::Size(w, h));

		result_.convertTo(result_, CV_32FC1);

		float* ptr_ = const_cast<float*>(data) + w * h;

		std::memcpy((void*)result_.data, (void*)ptr_, sizeof(float)*w*h);

		cv::threshold(result_, result_, 0, 0, cv::THRESH_TOZERO);
		cv::normalize(result_, result_, 0, 1, cv::NORM_MINMAX);
		cv::convertScaleAbs(result_, result_, 255);
		cv::applyColorMap(result_, result_, cv::COLORMAP_MAGMA);

		cv::resize(result_, result_, cv::Size(width_, height_));

		return result_;
	};

private:
	cv::Mat result_copy = cv::imread("../../models/test_img.jpg", 0);
	cv::Mat result_;
};

int main() {
	std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

	//输入的训练网络，可以替换
	auto det_model = "../../models/lapnet/LapNet_chkpt_better_epoch92618_GPU0.xml";

	//逐张显示
	bool WAIT_TO_SEE = true;

	ObjectDetector object_detector(det_model, "CPU", "", "", false);

	std::stringstream ss;

	for (int i = 1; i < 11; ++i)
	{
		ss << "../../imgs/s" << i << ".jpg";
		cv::Mat frame = cv::imread(ss.str());
		ss.str("");

		cv::Mat result = object_detector.getResult(frame);

		cv::Mat combine;
		cv::hconcat(frame, result, combine);
		cv::imshow("source&&result", combine);

		if (WAIT_TO_SEE)
		{
			cv::waitKey();
		}
		else
		{
			cv::waitKey(1);
		}
	}

	return 0;
}
