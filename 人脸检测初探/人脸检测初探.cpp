// 人脸检测初探.cpp : 定义控制台应用程序的入口点。
//http://blog.csdn.net/zwhlxl/article/details/44238347

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch(src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {


	// Get the path to your CSV.
	string fn_csv = string("at.txt");
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if(images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// The following lines create an Eigenfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// This here is a full PCA, if you just want to keep
	// 10 principal components (read Eigenfaces), then call
	// the factory method like this:
	//
	//      cv::createEigenFaceRecognizer(10);
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0), call it with:
	//
	//      cv::createEigenFaceRecognizer(10, 123.0);
	//
	// If you want to use _all_ Eigenfaces and have a threshold,
	// then call the method like this:
	//
	//      cv::createEigenFaceRecognizer(0, 123.0);
	//
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	return 0;
}

