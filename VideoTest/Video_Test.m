function videoTest

    Vehicles=struct([]);
    detector=vision.CascadeObjectDetector('vehicleBackDetector.xml');
    obj = VideoReader('jan28.avi'));
    detector.MergeThreshold = 3;
    detector.ScaleFactor=1.1;
    load('GroundTruth');
    f=1;
    _precision=0;
    _recall=0;
    while hasFrame(obj)
        frame = readFrame(obj);
        detectImg=frame;
        tic

        bbox=detector(frame);
        [bbox,Vehicles]= updateInstances(Vehicles,bbox,4,5,f);
        if size(bbox)>0
            detectImg=insertObjectAnnotation(frame,'rectangle',bbox,'vehicle');
            [precision,recall]=computeaccuracy(bbox,instance(f,:));
            _precision=precision+_precision;
            _recall=recall+_recall;
        end
        subplot(2,1,2);
        imshow(detectImg);
        if f<10
            frame_num=strcat('000',num2str(f));
        elseif f<100
            frame_num=strcat('00',num2str(f));
        elseif f<1000
            frame_num=strcat('0',num2str(f));
        elseif f>=1000
            frame_num=num2str(f);
        end
        name=strcat(frame_num,'.png');
        imwrite(detectImg,name);
        toc;
        f=f+1
        pause(0.01);
    end
    disp('precision=');
    disp(_precision/f);
    disp(' recall=');
    disp(_recall/f);
end
        
function [precision,recall]=computeaccuracy(detections,gtruths
	true_detections=0;
	[~,actual_positives]=size(gtruths);
	[total_detections,~]=size(detections);
	for y=1:actual_positives
    	if(size(gtruths{y})==0)
    	    break
    	end
    	for i=1:total_detections
        	overlap=bboxOverlapRatio(detections(i,:),gtruths{y});
        	if(overlap>0.5)
            	true_detections=true_detections+1;
        	end
    	end
	end
	precision=true_detections/total_detections;
	recall=true_detections/actual_positives;        
end

function [boxes,Vehicles]= updateInstances(Vehicles,bbox,minOccurence,framelimit,f)
    boxes=[];
    overlap=0;
    [bbox,g_size]=groupRectangles(bbox,1);
    [detections,~]=size(bbox);
    for i=1:detections
        [~,s_vehicles]=size(Vehicles);
        for y=1:s_vehicles
            overlap=bboxOverlapRatio(bbox(i,:),Vehicles(y).Position,'min');
            if (overlap>=0.5)
                Vehicles(y).Position=((Vehicles(y).Position*3)+(bbox(i,:)*2))/5;
                Vehicles(y).Occurence(mod(f,framelimit)+1)=g_size(i);
                Vehicles(y).Active=1;
                break;
            end
        end
        if(overlap<0.5)
            id=s_vehicles+1;
            Vehicles(id).Id=id;
            Vehicles(id).Position=bbox(i,:);
            Vehicles(id).Occurence(mod(f,framelimit)+1)=g_size(i);
            Vehicles(id).Active=1;
        end
    end
    for s=1:size(Vehicles')
        if Vehicles(s).Active==0
            Vehicles(s).Occurence(mod(f,framelimit)+1)=0;
        end
        Vehicles(s).Active=0;
        if sum(Vehicles(s).Occurence)>=minOccurence
            boxes=[boxes;Vehicles(s).Position];
        end
    end 
end 

function [rect,g_size]=groupRectangles(rectList,groupThreshold)
rect=[];
g_size=[];
if (isempty(rectList) || groupThreshold<=0)
    disp("Invalid Inputs");
    return
end
for i=1:size(rectList)
   group=rectList(i,:);
   b=i+1;
   for j=i+1:size(rectList)
       overlap=bboxOverlapRatio(rectList(i,:),rectList(b,:),'min');
       if overlap>0.5
           group=[group;rectList(b,:)];
           rectList(b,:)=[];
           b=b-1;
       end
       b=b+1;
   end
   [group_size,~]=size(group);
   if group_size>=groupThreshold
       box(1,1)=floor(median(group(:,1)));
       box(1,2)=floor(median(group(:,2)));
       box(1,3)=floor(median(group(:,1)+group(:,3))-box(1,1));
       box(1,4)=floor(median(group(:,2)+group(:,4))-box(1,2));
       rect=[rect;box];
       g_size=[g_size;group_size];
   end
   [stop,~]=size(rectList);
   if i == stop
       break
   end
end
