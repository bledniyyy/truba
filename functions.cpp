#include "functions.h"
#include <iostream>

using namespace cv;

cv::Rect mergeRectanglesWithMorphology(const std::vector<cv::Rect>& rects, const cv::Size& imageSize) {
    
    cv::Mat mask = cv::Mat::zeros(imageSize, CV_8UC1);                  // Создаем маску
    

    for (const auto& rect : rects) {        
        cv::rectangle(mask, rect, cv::Scalar(255), cv::FILLED);        // Рисуем все прямоугольники на маске
    }
    
    // Применяем морфологическое расширение для объединения близких областей
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(35, 35));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // Находим контуры объединенной области
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return cv::Rect();
    
    // Берем самый большой контур
    auto largestContour = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    
    return cv::boundingRect(*largestContour);
}


std::vector<cv::Rect> mergeOverlappingRectangles(const std::vector<cv::Rect>& rects, int maxGap) {
    std::vector<cv::Rect> mergedRects;
    std::vector<bool> merged(rects.size(), false);
    
    for (size_t i = 0; i < rects.size(); i++) {
        if (merged[i]) continue;
        
        cv::Rect current = rects[i];
        bool foundMerge = true;
        
        while (foundMerge) {
            foundMerge = false;
            for (size_t j = i + 1; j < rects.size(); j++) {
                if (merged[j]) continue;
                
                // Расширяем текущий прямоугольник с учетом зазора
                cv::Rect expanded = current;
                expanded.x -= maxGap;
                expanded.y -= maxGap;
                expanded.width += 2 * maxGap;
                expanded.height += 2 * maxGap;
                
                // Если прямоугольники пересекаются или близко
                if ((expanded & rects[j]).area() > 0) {
                    // Объединяем прямоугольники
                    current = current | rects[j];
                    merged[j] = true;
                    foundMerge = true;
                }
            }
        }
        mergedRects.push_back(current);
    }
    
    return mergedRects;
}

cv::Mat preprocessingImage (cv::Mat originalImage) { //cv::Mat CLAHEimage) {
    Mat grayImage, image_lab, noBlics;
    cvtColor(originalImage, grayImage, COLOR_BGR2GRAY);                 //преобразуем изображение в серый цвет
    cvtColor(originalImage, image_lab, COLOR_BGR2Lab);

    GaussianBlur(grayImage, grayImage, Size(5,5), 5, 5);        //применяем размытие гаусса для сглаживания шумов


    //применяем метод CLAHE для повышения контрастности изображение
    std::vector<Mat> lab_planes;                                //создаём вектор для кадров изображения
    split(image_lab, lab_planes);                               //делим изображение на кардры
    Ptr<CLAHE> clahe = createCLAHE();                           //создаём CLAHE
    clahe->setClipLimit(4.0);
    clahe->setTilesGridSize(Size(8,8));

    Mat clahe_output;
    clahe->apply(grayImage, clahe_output);                      //применяем CLAHE
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3)); 
    morphologyEx(clahe_output, noBlics, MORPH_CLOSE, kernel);   // ...

    return noBlics;
}


cv::Rect2i firstPartCannyFind (cv::Mat originalImage ,cv::Mat predprocessedImage) {
    Mat cannyImage;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
    Canny(predprocessedImage, cannyImage, 100, 100);                       //отрисовываем контуры объектов
    morphologyEx(cannyImage, cannyImage, MORPH_CLOSE, kernel);  //слепляем контуры

    std::vector<std::vector<Point>> countours;                  //вектор точек контуров
    std::vector<Vec4i> hierarchy;                               //иерархия контуров
    findContours(cannyImage, countours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE); //ищем контуры для последующей фильтрации и отрисовки

    std::vector<Rect> boundingRects(countours.size());           //вектор ограничивающих прямоугольники для каждого контура
    std::vector<Rect> TubedRects;                                //вектор первично отфильтрованных прямоугольников
   
    for (size_t i = 0; i < countours.size(); i++) {              //фильтруем полученные прямоугольники
        boundingRects[i] = boundingRect(countours[i]);
        if (boundingRects[i].area() >= 2000) {              
            if (boundingRects[i].x > boundingRects[i].y) { 
                TubedRects.push_back(boundingRect(countours[i]));
            }
        }
    }

    //Mat imgcopy;
    //originalImage.copyTo(imgcopy);

    auto merged1 = mergeOverlappingRectangles(TubedRects, 5);
    Rect merged2 = mergeRectanglesWithMorphology(merged1, originalImage.size());

    
    //rectangle(originalImage, merged2, Scalar(0,0,255), 3);              //отрисовка лучшего прямоугольника по результатам первичной фильтрации

    return merged2;
}

cv::Rect2i secondPartCirclesFind (cv::Mat originalImage, cv::Mat predprocessedImage, cv::Rect2i firstFindedRectangle) {
    std::vector<Vec3f> circles;
    std::vector<Vec3i> filtredCircles;
    HoughCircles(predprocessedImage, circles, HOUGH_GRADIENT, 1.5, predprocessedImage.rows / 2, 100, 30, 50, 150);

    // Правильная фильтрация кругов по высоте прямоугольника
    int rectCenterY = firstFindedRectangle.y + firstFindedRectangle.height / 2;
    int tolerance = 200; // Допуск по высоте
    
    for (size_t c = 0; c < circles.size(); c++) {
        Vec3i s = circles[c]; 
        Point center = Point(s[0], s[1]);
        // Круги должны быть примерно на той же высоте, что и прямоугольник
        if (abs(center.y - rectCenterY) <= tolerance) {
            int raduis = s[2];
            filtredCircles.push_back(s);
            //circle(imgcopy, center, raduis, Scalar(255,255,255), 2);
        }
    }

    // Сохраняем исходный прямоугольник для сравнения
    Rect originalRect = firstFindedRectangle;
    
    if (!filtredCircles.empty()) {
        // Сортируем круги по X для определения левого и правого
        std::sort(filtredCircles.begin(), filtredCircles.end(), [](const Vec3i &a, const Vec3i &b){
            return a[0] < b[0];
        });

        // Находим крайние круги
        Vec3i leftCircle = filtredCircles.front();
        Vec3i rightCircle = filtredCircles.back();
        
        // Вычисляем среднюю высоту кругов
        int sumY = 0;
        for (const auto& circle : filtredCircles) {
            sumY += circle[1];
        }
        int avgY = sumY / filtredCircles.size();
        
        // ОЦЕНКА ТОЧНОСТИ:
        // 1. Если круги образуют хорошую линию, используем их для построения прямоугольника
        // 2. Если нет, то просто расширяем существующий прямоугольник
        
        // Проверяем, образуют ли круги хорошую горизонтальную линию
        bool goodLine = true;
        int lineThreshold = 30; // Максимальное отклонение по Y
        
        for (const auto& circle : filtredCircles) {
            if (abs(circle[1] - avgY) > lineThreshold) {
                goodLine = false;
                break;
            }
        }
        
        // Также проверяем покрытие - сколько кругов находятся внутри исходного прямоугольника
        int circlesInside = 0;
        for (const auto& circle : filtredCircles) {
            Point center(circle[0], circle[1]);
            if (originalRect.contains(center)) {
                circlesInside++;
            }
        }
        double coverage = (double)circlesInside / filtredCircles.size();
        
        Rect newRect;
        
        if (goodLine && coverage < 0.7) {
            // Круги образуют хорошую линию и большинство вне прямоугольника - строим новый
            int leftX = leftCircle[0] - leftCircle[2] - 20;
            int rightX = rightCircle[0] + rightCircle[2] + 20;
            int topY = avgY - originalRect.height / 2;
            int bottomY = avgY + originalRect.height / 2;
            
            newRect = Rect(leftX, topY, rightX - leftX, bottomY - topY);
            std::cout << "NEW RECTANGLE BUILT FROM CIRCLES\n";
        } else {
            // Расширяем существующий прямоугольник на основе кругов
            newRect = originalRect;
            
            for (const auto& circle : filtredCircles) {
                Point center(circle[0], circle[1]);
                int radius = circle[2];
                
                Rect circleRect(
                    center.x - radius - 15,
                    center.y - radius - 15, 
                    radius * 2 + 30, 
                    radius * 2 + 30
                );
                
                newRect = newRect | circleRect;
            }
            std::cout << "RECTANGLE EXTENDED WITH CIRCLES\n";
        }
        
        // Обновляем основной прямоугольник
        firstFindedRectangle = newRect;
        
        // Перерисовываем обновленный прямоугольник
        //rectangle(originalImage, firstFindedRectangle, Scalar(0,255,0), 3);
        
        std::cout << "Original: x=" << originalRect.x << " y=" << originalRect.y 
                  << " w=" << originalRect.width << " h=" << originalRect.height << "\n";
        std::cout << "New: x=" << firstFindedRectangle.x << " y=" << firstFindedRectangle.y 
                  << " w=" << firstFindedRectangle.width << " h=" << firstFindedRectangle.height << "\n";
        std::cout << "Circles found: " << filtredCircles.size() 
                  << ", Coverage: " << coverage * 100 << "%\n";

        return newRect;
    } else {
        std::cout << "No aligned circles found - using original rectangle\n";
    }    
}

cv::Rect2i thirdPartColorFind (cv::Mat originalImage) {
    int hmin = 0, hmax = 79, smin = 13, smax = 183, vmin = 0, vmax = 86;

    Mat HSVimage;
    Mat Points;
    GaussianBlur(originalImage, HSVimage, Size(5,5), 5, 5);
    cvtColor(originalImage, HSVimage, COLOR_BGR2HSV);

    std::vector<int> lover_bound = {hmin, smin, vmin};
    std::vector<int> upper_bound = {hmax, smax, vmax};

    inRange(HSVimage, lover_bound, upper_bound, Points);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(6,6));
    dilate(Points, Points, kernel);

    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> contours;

    findContours(Points, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Rect minSrect;
    minSrect.width = 100;
    minSrect.height = 100;

    int val_biggest = contours[0].size();
    int idx_biggest = 0;

    int minS_rect = minSrect.width * minSrect.height;
    for (size_t cntrs = 0; cntrs < contours.size(); cntrs++) {
        if (val_biggest < contours[cntrs].size()) {
                    val_biggest = contours[cntrs].size();
                    idx_biggest = cntrs;
        }
        
        Rect bounding_box = boundingRect(contours[idx_biggest]);

        
        int width = bounding_box.width;
        int height = bounding_box.height;

        if ((width * height) > minS_rect) {
            int x = bounding_box.x;
            int y = bounding_box.y;
            
            //rectangle(originalImage, bounding_box, Scalar(0, 0, 0), 2);
            return bounding_box;
        }
    }
}

PipeOrientation detectOrientation(const cv::Rect& rect) {
    float aspectRatio = (float)rect.width / rect.height;
    
    if (aspectRatio > 1.5) {
        
        return HORIZONTAL;  // Ширина значительно больше высоты
    } else if (aspectRatio < 0.67) {
        
        return VERTICAL;    // Высота значительно больше ширины
    } else {
        
        return UNKNOWN;     // Примерно квадратный
    }
}

double rectangleQuality(const cv::Rect& rect) {
    double aspectRatio = static_cast<double>(rect.width) / rect.height;
    
    // Идеальное соотношение для трубы (можно настроить)
    const double idealAspect = 3.0; // например, 3:1
    
    // Вычисляем насколько близко к идеальному соотношению
    double ratio = std::max(aspectRatio, 1.0 / aspectRatio);
    double quality = 1.0 / (1.0 + std::abs(ratio - idealAspect));
    
    // Учитываем площадь (большие прямоугольники предпочтительнее)
    quality *= std::sqrt(rect.area());
    
    return quality;
}

