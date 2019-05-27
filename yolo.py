from global_var import *

#Code to track and get the pose of the person
def object_tracking(image,res):

    global track_points, pixel_displacement_value

    box=list(track_points.keys())
    color=list(track_points.values())
    track_points={}

    for i in res:
        detection=i[0].decode('utf-8')        
        if(detection in ('person')):
            flag=0
            upper = (int(i[2][0]-i[2][2]/2),int(i[2][1]-i[2][3]/2))
            lower = (int(i[2][0]+i[2][2]/2),int(i[2][1]+i[2][3]/2))

            if(len(box)!=0):
                ind,j=min(enumerate([sum(np.abs(np.subtract((int(i[2][0]),int(i[2][1])),k))) for k in box]),key=lambda x:x[1])

                if(j<pixel_displacement_value):
                    flag=1

            if(flag==1):   
                track_points[((int(i[2][0]),int(i[2][1])))]=[color[ind][0],color[ind][1],color[ind][1]]

            if(flag==0):
                track_points[((int(i[2][0]),int(i[2][1])))]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            
            cv2.rectangle(image, upper, lower , track_points[((int(i[2][0]),int(i[2][1])))], thickness = 4)
            cv2.putText(image, '{0}'.format(detection),(int(i[2][0]-i[2][2]/2), 
                        int(i[2][1]-i[2][3]/2)  -6),cv2.FONT_HERSHEY_SIMPLEX,0.6, (255, 255, 255),1,cv2.LINE_AA)
                

    return image

#Yolo detection block 
def detect(image, thresh=0.9, hier_thresh=.5, nms=.45):

    global predict_image , ndarray_image, get_network_boxes, net, meta, free_image, free_detections

    data = image.ctypes.data_as(POINTER(c_ubyte))
    im = ndarray_image(data, image.ctypes.shape, image.ctypes.strides)

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): 
        do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])  

    free_image(im) 
    free_detections(dets, num) 

    image=object_tracking(image,res)

    return image