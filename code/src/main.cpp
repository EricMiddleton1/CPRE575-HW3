// Headers
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <numeric>
#include <utility>

enum class ProjectionType {
	Horizontal,
	Vertical
};

using Path = std::vector<std::vector<cv::Point>>;

void changeColor(const cv::Mat& inframe, cv::Mat& outFrame, const cv::Scalar& oldColorMin,
	const cv::Scalar& oldColorMax, const cv::Scalar& newColor, const std::string& window = {});

template<ProjectionType Type>
std::vector<int> getProjection(const cv::Mat& image);

std::vector<int> getPeaks(const std::vector<int>&, unsigned int minDistance = 1);

template<ProjectionType Dir>
std::vector<std::pair<int, int>> locate(const cv::Mat& mask, int minArea);

void drawPath(cv::Mat& frame, const Path& path, const cv::Scalar& color, int thickness = 3);

int main(int argc, char* argv[]) {
	
	if(argc != 2) {
		std::cerr << "Usage: " << argv[0] << " video_file" << std::endl;
		return 1;
	}

	// Load input video
	//  If your video is in a different source folder than your code, 
	//  make sure you specify the directory correctly!
	cv::VideoCapture input_cap(argv[1]);
	
	// Check validity of target file
	if(!input_cap.isOpened()) {
		std::cerr << "Input video not found." << std::endl;
		return 1;
	}
	
	// Set up target output video
	/*	usage: VideoWriter(filename, encoding, framerate, Size)
	 *		in our case, cv_cap_prop_* means "get property of capture"
	 *	 	we want our output to have the same properties as the input!
	 */
	cv::VideoWriter output_cap("output.avi", 
							input_cap.get(CV_CAP_PROP_FOURCC),
							input_cap.get(CV_CAP_PROP_FPS),
							cv::Size(input_cap.get(CV_CAP_PROP_FRAME_WIDTH),
							input_cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
	
	// Again, check validity of target output file
	if(!output_cap.isOpened()) {
		std::cerr << "Could not create output file." << std::endl;
		return 1;
	}
	
	cv::namedWindow("Frame");

	int pelletCount = -1, lastPelletCount = 0;
	int pelletTime = 6;

	Path playerPath;
	bool pathBreak = true;

	// Loop to read from input one frame at a time, write text on frame, and
	// copy to output video
	cv::Mat frame;
	while(input_cap.read(frame) && (cv::waitKey(10) == -1)) {
		cv::Mat frame_hsv;
		auto frameSize = frame.size();

		//Convert frame to HSV color space
		cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
		
		//Replace border lines with brown
		changeColor(frame_hsv, frame, cv::Scalar(110, 200, 50), cv::Scalar(130, 255, 200),
			cv::Scalar(60, 78, 115));

		//Change Pac-Man color to green
		changeColor(frame_hsv, frame, cv::Scalar(25, 100, 100), cv::Scalar(35, 255, 255),
			cv::Scalar(0, 255, 0));

		//Detect Power Pellets
		cv::Mat pelletMask, pelletMask2;
		cv::inRange(frame_hsv, cv::Scalar(170, 40, 225), cv::Scalar(180, 85, 255), pelletMask);
		cv::inRange(frame_hsv, cv::Scalar(0, 40, 225), cv::Scalar(10, 85, 255), pelletMask2);
		cv::bitwise_or(pelletMask, pelletMask2, pelletMask);
		
		//Locate all blobs of size
		auto blobs = locate<ProjectionType::Horizontal>(pelletMask, 40);
		int curPelletCount = 0;
		for(auto blob : blobs) {
			//Pellet is ~40 pixels. Integer division blobSize/40 gives total pellet count
			curPelletCount += blob.second / 40;
		}
		if(pelletCount == -1) {
			pelletCount = curPelletCount;
		}
		else if(lastPelletCount == 0 && curPelletCount > 0) {
			//Detect that pellets have reappeared on screen
			pelletCount = curPelletCount;
		}
		else if(curPelletCount == 0) {
			pelletTime--;
			if(pelletTime < 0) {
				//If pellets should have appeared again, but have not, then there are no pellets
				pelletCount = 0;
				pelletTime = 6;
			}
		}
		else {
			pelletTime = 6;
		}
		lastPelletCount = curPelletCount;

		cv::putText(frame, std::to_string(pelletCount), cv::Point(210, 10),
			cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255));

		//Detect number of lives
		cv::Mat frameLives = frame_hsv(cv::Rect(0, frameSize.height-20, 100, 20)), livesMask;
		cv::inRange(frameLives, cv::Scalar(20, 100, 200), cv::Scalar(40, 255, 255), livesMask);
		int lives = locate<ProjectionType::Horizontal>(livesMask, 10).size();
		
		cv::putText(frame, std::to_string(lives), cv::Point(75, 285), cv::FONT_HERSHEY_PLAIN, 1.0,
			cv::Scalar(255, 255, 255));

		//Detect Pac-Man and plot his trajectory
		cv::Mat playerMask;
		auto gameWindow = frame_hsv(cv::Rect(0, 0, frameSize.width, frameSize.height - 20));
		cv::inRange(gameWindow, cv::Scalar(25, 100, 100), cv::Scalar(35, 255, 255), playerMask);
		auto playerX = locate<ProjectionType::Horizontal>(playerMask, 10),
			playerY = locate<ProjectionType::Vertical>(playerMask, 10);

		if(playerX.size() != 1 || playerY.size() != 1) {
			//Start new curve in path (needed for when pac-man jumps from left to right)
			pathBreak = true;
		}
		else {
			if(pathBreak == true) {
				//Start new curve
				playerPath.emplace_back();
				pathBreak = false;
			}
			//Place current point into current curve
			playerPath.back().emplace_back(playerX[0].first, playerY[0].first);
		}

		drawPath(frame, playerPath, cv::Scalar(0, 255, 255));

		cv::imshow("Frame", frame);
		output_cap.write(frame);
	}
	
	
	// free the capture objects from memory
	input_cap.release();
	output_cap.release();
	
	return 0;
}

void changeColor(const cv::Mat& inFrame, cv::Mat& outFrame, const cv::Scalar& oldColorMin,
	const cv::Scalar& oldColorMax, const cv::Scalar& newColor, const std::string& window) {
		cv::Mat mask, frameColor{outFrame.size(), CV_8UC3, newColor};

		cv::inRange(inFrame, oldColorMin, oldColorMax, mask);
		frameColor.copyTo(outFrame, mask);

		if(!window.empty()) {
			cv::imshow(window, mask);
		}
}

/*
 * getProjection
 *
 * Calculates horizontal or vertical projection of image
 * and returns as vector<int>
 *
*/

template<ProjectionType Type>
std::vector<int> getProjection(const cv::Mat& image) {
	std::vector<int> projection((Type == ProjectionType::Horizontal) ?
		image.cols : image.rows);

	for(unsigned int i = 0; i < projection.size(); ++i) {
		if(Type == ProjectionType::Horizontal) {
			projection[i] = cv::countNonZero(image(cv::Rect(i, 0, 1, image.rows)));
		}
		else {
			projection[i] = cv::countNonZero(image(cv::Rect(0, i, image.cols, 1)));
		}
	}

	return projection;
}

/*
 * getPeaks
 *
 * Detect peaks from input vector<int> with min distance between
 * and returns as vector<int>
 *
*/

std::vector<int> getPeaks(const std::vector<int>& values, unsigned int minDistance) {
	std::vector<int> peaks;

	int avg = std::accumulate(values.begin(), values.end(), 0) / values.size();
	for(unsigned int i = 0; i < values.size(); ++i) {
		if((peaks.empty() || (i >= (peaks[peaks.size()-1]+minDistance))) && values[i] > avg) {
			peaks.push_back(i);
		}
	}

	return peaks;
}

template<ProjectionType Dir>
std::vector<std::pair<int, int>> locate(const cv::Mat& mask, int minArea) {
	std::vector<std::pair<int, int>> blobs;

	auto proj = getProjection<Dir>(mask);
	
	int area = 0, start = 0;
	for(size_t i = 0; i < proj.size(); ++i) {
		if(proj[i] > 0) {
			if(area == 0) {
				start = i;
			}
			area += proj[i];
		}
		else if(area > 0) {
			if(area >= minArea) {
				blobs.emplace_back((start+i)/2, area);
			}
			area = 0;
		}
	}

	return blobs;
}


void drawPath(cv::Mat& frame, const Path& path, const cv::Scalar& color, int thickness) {
	for(const auto& curve : path) {
		if(curve.size() > 1) {
			for(size_t i = 0; i < (curve.size()-1); ++i) {
				cv::line(frame, curve[i], curve[i+1], color, thickness);
			}
		}
	}
}