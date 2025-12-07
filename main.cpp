#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include "functions.h"

using namespace cv;
using namespace std;

double elongation(const Rect& r) {
    double w = r.width;
    double h = r.height;
    if (w == 0 || h == 0) return 0;
    return max(w / h, h / w);
}

pair<Rect, Rect> pickTwoMostElongated(const vector<Rect>& rects) {
    if (rects.size() < 2) 
        return {Rect(), Rect()};

    vector<pair<double, Rect>> scored;
    for (auto& r : rects)
        scored.push_back({ elongation(r), r });

    sort(scored.begin(), scored.end(),
         [](auto& a, auto& b){ return a.first > b.first; });

    return { scored[0].second, scored[1].second };
}

Rect mergeRects(const Rect& a, const Rect& b) {
    int x1 = min(a.x, b.x);
    int y1 = min(a.y, b.y);
    int x2 = max(a.x + a.width, b.x + b.width);
    int y2 = max(a.y + a.height, b.y + b.height);

    return Rect(x1, y1, x2 - x1, y2 - y1);
}



int main() {
    Mat image;
    image = imread("/home/b/Рабочий стол/opencv_test/modelCVtube/13.png"); //подругжаем изображение или видео

    if (image.empty()) {
        std::cerr << "ERROR: IMAGE IS EMPTY!";
        return -1;
    }

    //=========================================================Предобработка изображения====================================================================
   
    cv::Mat predProcessedImage = preprocessingImage (image);

    //=========================================================Работа с контурами========================================================================
   
    Rect firstRectangle = firstPartCannyFind(image, predProcessedImage);
    //rectangle(image, firstRectangle, Scalar(0,0,255), 3); 

   // =========================================================Проверка формы и подтверждение================================================================
    
    Rect secondFindedRect = secondPartCirclesFind(image, predProcessedImage, firstRectangle);
    //rectangle(image, secondFindedRect, Scalar(0,255,0), 3);
   

    //==============================================================Проверка цвета===========================================================================
    
    Rect thirdFindedRect = thirdPartColorFind(image);
    //rectangle(image, thirdFindedRect, Scalar(0,0,0), 3);


    double first = rectangleQuality (firstRectangle);
    double second = rectangleQuality(secondFindedRect);
    double third = rectangleQuality(thirdFindedRect);
    
    std::cout << first << " " << second << " " << third << std::endl;
    
    //=============================================================Сравнение и поиск лучшего=================================================================
    
    std::vector<Rect> findedRects = {firstRectangle, secondFindedRect, thirdFindedRect};

    auto [r1, r2] = pickTwoMostElongated(findedRects);
    Rect result = mergeRects(r1, r2);
    //Rect pipe = pipeRectAuto(findedRects);
    //Rect joint = jointRectAuto(findedRects, pipe);
    //Rect result = buildFinalRect(pipe, joint);

    rectangle(image, result, Scalar(0,0,0), 3);
    
    imshow("image", image);
    cv::waitKey(0);

    return 0;


}


