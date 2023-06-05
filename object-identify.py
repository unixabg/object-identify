# import the necessary packages
import argparse
import datetime
import time
import cv2
import sys

#This is to pull the information about what each object is called
classNames = []
classFile = "Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
	classNames = f.read().rstrip("\n").split("\n")

#This is to pull the information about what each object should look like
configPath = "Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Object_Detection_Files/frozen_inference_graph.pb"

#This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



#This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
	classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
	#Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below
	#print(classIds,bbox)
	if len(objects) == 0:
		objects = classNames

	objectInfo =[]

	if len(classIds) != 0:
		for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
			className = classNames[classId - 1]
			if className in objects:
				objectInfo.append([box,className])
				if (draw):
					cv2.rectangle(img,box,color=(0,255,0),thickness=2)
					cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,.5,(255,255,255),1)
					cv2.putText(img,str(round(confidence*100,2)),(box[0]+100,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,.5,(255,255,255),1)

	return img,objectInfo


#Below determines the size of the live feed window that will be displayed on the Raspberry Pi OS
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", help="path to the video file")
	ap.add_argument("--preview", action="store_true", help="show preview window")
	ap.add_argument("--threshold", type=float, default=0.45, help="set detection threshold")
	args = vars(ap.parse_args())

	if args["video"] is None:
		print("Please either specify a video device, video or image to the --video parameter.");
		print(" --video /dev/video0");
		print(" --video test.mp4");
		print(" --video test.jpg");
		sys.exit()

	# set the detection confidence threshold
	threshold = args["threshold"]
	if threshold > 0.85:
		print("Input threshold set to high, lowering to 0.85")
		threshold = 0.85
	elif threshold < 0.3:
		print("Input threshold set to low, increasing to 0.3")
		threshold = 0.3
	else:
		print("Input threshold is set at "+str(threshold))

	cap = cv2.VideoCapture(args["video"])
	cap.set(3,640)
	cap.set(4,480)


	while True:
		success, img = cap.read()

		# if the frame could not be grabbed, then we have reached the end of the video
		if img is None:
			break

		#Below provides a huge amount of controll. the detection threshold number, the 0.2 number is the nms number)
		result, objectInfo = getObjects(img,threshold,0.2,objects=['person'])
		#print(objectInfo)
		timestr = time.strftime("%Y%m%d-%H%M%S")
		if len(objectInfo) > 0:
			cv2.imwrite(timestr+'test.jpg',img)

		if args["preview"]:
			cv2.imshow("Output",img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	#clean up
	#cap.stop();
	cv2.destroyAllWindows()
