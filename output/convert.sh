#!/bin/bash

for video_in in `find . -name *.avi`
do
	video_out="${video_in%.*}.mp4"
	
	echo Converting $video_in to $video_out
	ffmpeg -i $video_in -c:a aac -b:a 128k -c:v libx264 -crf 23 $video_out
done
