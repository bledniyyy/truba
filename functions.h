#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>

//@brief Статусы положения трубы
enum PipeOrientation {
    HORIZONTAL,
    VERTICAL,
    UNKNOWN
};

//@brief Функция для объединения пересекающихся прямоугольников с помощью морфологического закрытия
//@param rects  вектор прямоугольников, 
//@param imageSize размер изображения
cv::Rect mergeRectanglesWithMorphology(const std::vector<cv::Rect>& rects, const cv::Size& imageSize);
 
//@brief Функция для объединения пересекающихся прямоугольников
//@param rects вектор прямоугольников, 
//@param maxGap максимальное расстояние между ними
std::vector<cv::Rect> mergeOverlappingRectangles(const std::vector<cv::Rect>& rects, int maxGap = 10);

//@brief Функция предобработки изображения с применением CLAHE и с последующем убиранием бликов.
//@brief На выходе бинарное изображение.
//@param originalImage оригинальное изображение
cv::Mat preprocessingImage (cv::Mat originalImage);

//@brief Функция реализующая первый этап распознавания трубы на фото. Использует Canny.
//@brief На выходе на изображении рисуется предполгаемая область трубы. Размеры и координаты прямоугольника
//@param originalImage оригинальное изображение
//@param predprocessedImage подготовленное бинарное изображение
cv::Rect2i firstPartCannyFind (cv::Mat originalImage ,cv::Mat predprocessedImage);

//@brief Функция реализовывающая второй этап распознавания трубы на фото, а также подтверждения, выбранного объекта на первом этапе
//@brief Функция работает за счёт того что пытается вписать круги на фото и ищет наиболее подходящие круги по определенным параметрам
//@brief На выходе - обновлённый прямоугольник предполагаемой области трубы. Размеры и координаты прямоугольника
//@param originalImage оригинальное изображение
//@param predprocessedImage подготовленное бинарное изображение
//@param firstFindedRectangle прямоугольник найденный функцией `firstPartCannyFind (cv::Mat predprocessedImage)`
cv::Rect2i secondPartCirclesFind (cv::Mat originalImage, cv::Mat predprocessedImage, cv::Rect2i firstFindedRectangle);

//@brief Функция реализовывающая третий этап распознавания трубы на фото по цвету
//@param originalImage оригинальное изображение
cv::Rect2i thirdPartColorFind (cv::Mat originalImage);

//@brief Функция определяющая положение трубы
//@param rect подопытный прямоугольник
PipeOrientation detectOrientation(const cv::Rect& rect);

//@brief Функция для вычисления "качества" прямоугольника на основе соотношения сторон
//@param rect подопытный прямоугольник
double rectangleQuality(const cv::Rect& rect);



#endif