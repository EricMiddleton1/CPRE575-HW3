// Headers
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char* argv[]) {
	
	// Load input video
	//  If your video is in a different source folder than your code, 
	//  make sure you specify the directory correctly!
	VideoCapture input_cap("../../input/part1a/input1a1.avi");
	
	// Check validity of target file
	if(!input_cap.isOpened()) {
		std::cout << "Input video not found." << std::endl;
		return -1;
	}
	
	// Set up target output video
	/*	usage: VideoWriter(filename, encoding, framerate, Size)
	 *		in our case, cv_cap_prop_* means "get property of capture"
	 *	 	we want our output to have the same properties as the input!
	 */
	
	VideoWriter output_cap("helloworld.avi", 
							input_cap.get(CV_CAP_PROP_FOURCC),
							input_cap.get(CV_CAP_PROP_FPS),
							Size(input_cap.get(CV_CAP_PROP_FRAME_WIDTH),
							input_cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
	
	// Again, check validity of target output file
	if(!output_cap.isOpened()) {
		std::cout << "Could not create output file." << std::endl;
		return -1;
	}
	
	// Loop to read from input one frame at a time, write text on frame, and
	// copy to output video
	Mat frame;
	while(input_cap.read(frame)) {
		putText(frame, "Hello World!",
				Point(0, 50),
				FONT_HERSHEY_PLAIN,
				1.0,
				Scalar(255, 255, 255));
		
		output_cap.write(frame);
	}
	
	
	// free the capture objects from memory
	input_cap.release();
	output_cap.release();
	
	return 1;
	
}