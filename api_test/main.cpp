#include <iostream>
#include <dirent.h>  
#include "caffe/caffe.hpp"
#include "Classifier.h"
using namespace caffe; 
using namespace std;
string filePath = "/home/roger/WorkSpace/caffe/have-fun-with-machine-learning/data/untrained-samples";  

int main(){

    Classifier* classifier = new Classifier("/home/roger/Downloads/20170820-114817-f935_epoch_30.0/deploy.prototxt",
                                    "/home/roger/Downloads/20170820-114817-f935_epoch_30.0/snapshot_iter_30.caffemodel",
                                    "/home/roger/Downloads/20170820-114817-f935_epoch_30.0/mean.binaryproto",
                                    "/home/roger/Downloads/20170820-114817-f935_epoch_30.0/labels.txt");


    struct dirent *ptr;      
    DIR *dir;
    dir=opendir(filePath.c_str());  

    vector<string> files;  
    cout << "文件列表: "<< endl;  
    while((ptr=readdir(dir))!=NULL)  
    {     
        //跳过'.'和'..'两个目录  
        if(ptr->d_name[0] == '.')  
            continue;  
        int i = 1;
        while(1){
            if(ptr->d_name[i] == '.')break;
            i++;
        }  
        if(ptr->d_name[i+1] == 'j')
            files.push_back(ptr->d_name);  
    }  
    for (int i = 0; i < files.size(); ++i)  
    {  
        std::cout << files[i] <<std::endl;
        cv::Mat img = cv::imread("/home/roger/WorkSpace/caffe/have-fun-with-machine-learning/data/untrained-samples/"+files[i], -1);
        std::vector<Prediction> predictions = classifier->Classify(img, 2);
        /* Print the top N predictions. */
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            Prediction p = predictions[i];
            std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "      " ;
        }
        std::cout<<std::endl;

    }
    return 0;
}