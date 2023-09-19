#include <iostream>
#include <thread>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include <chrono>
#include <emmintrin.h>



// Функция для конвертации изображения BMP из RGB в YUV420
void rgbToYuv420(const std::vector<uint8_t>& rgb, std::vector<uint8_t>& yuv420, int width, int height, int startRow, int endRow) {
    // Размеры Y-плоскости
    int yPlaneSize = width * height;

    // Размеры U и V плоскостей 
    int uvPlaneSize = yPlaneSize / 4;

    // Разделение Y, U и V плоскостей в выходном векторе
    yuv420.resize(yPlaneSize + 2 * uvPlaneSize);

    // Итераторы для Y, U и V плоскостей
    uint8_t* yPtr = yuv420.data();
    uint8_t* uPtr = yPtr + yPlaneSize;
    uint8_t* vPtr = uPtr + uvPlaneSize;

    for (int row = startRow; row < endRow; row+=2) {
        // Обратный порядок обхода строк (отражение по вертикали)
        int reversedRow = height - 1 - row;
        for (int col = 0; col < width; col+=2) {
            int rgbIndex = (reversedRow * width + col) * 3;
            uint8_t r = rgb[rgbIndex + 2]; 
            uint8_t g = rgb[rgbIndex + 1];
            uint8_t b = rgb[rgbIndex];     
            yPtr[row * width + col] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            uPtr[(row / 2) * (width / 2) + (col / 2)] = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            vPtr[(row / 2) * (width / 2) + (col / 2)] = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
            
            rgbIndex = ((reversedRow-1) * width + col) * 3;
             r = rgb[rgbIndex + 2];
             g = rgb[rgbIndex + 1];
             b = rgb[rgbIndex];
            yPtr[(row+1) * width + col] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
           

            rgbIndex = ((reversedRow - 1) * width + (col+1)) * 3;
             r = rgb[rgbIndex + 2];
             g = rgb[rgbIndex + 1];
             b = rgb[rgbIndex];
            yPtr[(row + 1) * width + (col + 1)] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
          
           
            rgbIndex = (reversedRow  * width + (col + 1)) * 3;
             r = rgb[rgbIndex + 2];
             g = rgb[rgbIndex + 1];
             b = rgb[rgbIndex];
            yPtr[row  * width + (col + 1)] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
          
        }
    }
}

// Функция для наложения изображения на кадр видео в верхнем левом углу
void overlayImageOnFrame(const std::vector<uint8_t>& imageYUV, std::vector<uint8_t>& frameYUV, int frameWidth, int frameHeight, int imageWidth, int imageHeight, int x = 0, int y = 0) {
    int yPlaneSizeVID = frameWidth * frameHeight;
    int yPlaneSizePic = imageWidth * imageHeight;

    // Масштабируем размеры изображения до размеров кадра видео
    int scaledImageWidth = std::min(imageWidth, frameWidth - x);
    int scaledImageHeight = std::min(imageHeight, frameHeight - y);

    // Налагаем компоненту Y (яркость) изображения на компоненту Y кадра видео
    for (int row = 0; row < scaledImageHeight; row++) {
        for (int col = 0; col < scaledImageWidth; col++) {
            int imageIndex = (row * imageWidth + col);
            int frameIndex = ((y + row) * frameWidth + (x + col));
            frameYUV[frameIndex] = imageYUV[imageIndex];//записали y часть
        }
    }

    // Налагаем компоненты U и V (цветовые разности) изображения на кадр видео
    for (int row = 0; row < scaledImageHeight / 2; row++) {
        for (int col = 0; col < scaledImageWidth / 2; col++) {
            int imageIndex = (yPlaneSizePic + row * (imageWidth / 2) + col);
            int frameIndex = (yPlaneSizeVID + (y / 2 + row) * (frameWidth / 2) + (x / 2 + col));
            frameYUV[frameIndex] = imageYUV[imageIndex];//записали U часть (синяя разность)

            imageIndex = (yPlaneSizePic + yPlaneSizePic / 4 + row * (imageWidth / 2) + col);
            frameIndex = (yPlaneSizeVID + yPlaneSizeVID / 4 + (y / 2 + row) * (frameWidth / 2) + (x / 2 + col));
            frameYUV[frameIndex] = imageYUV[imageIndex];//записали V часть (красная разность)
        }
    }
}


// Функция, которую будут выполнять потоки для наложения кадров.
void processImageRows(const std::vector<uint8_t>& imageYUV, std::vector<uint8_t>& frameYUV, int frameWidth, int frameHeight, int imageWidth, int imageHeight, int x, int y, int startY, int endY) {
    int yPlaneSizeVID = frameWidth * frameHeight;
    int yPlaneSizePic = imageWidth * imageHeight;
    int scaledImageWidth = std::min(imageWidth, frameWidth - x);
    int scaledImageHeight = std::min(imageHeight, frameHeight - y);



    // Оптимизация с использованием SIMD инструкций SSE2 для компоненты Y
        for (int row = startY; row < endY; row++) {
            for (int col = 0; col < imageWidth; col += 16) {
                // Загрузка 16 значений изображения
                __m128i imgY = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&imageYUV[(row * imageWidth + col)]));

                // Сохранение 
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&frameYUV[((y + row) * frameWidth + (x + col))]), imgY);
            }
        }

        for (int row = startY / 2; row < endY / 2; row++) {
            for (int col = 0; col < scaledImageWidth / 2; col+=16) {

                int imageIndex = (yPlaneSizePic + row * (imageWidth / 2) + col);
                int frameIndex = (yPlaneSizeVID + (y / 2 + row) * (frameWidth / 2) + (x / 2 + col));

                __m128i imgU = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&imageYUV[imageIndex]));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&frameYUV[frameIndex]), imgU);



                imageIndex = (yPlaneSizePic + yPlaneSizePic / 4 + row * (imageWidth / 2) + col);
                frameIndex = (yPlaneSizeVID + yPlaneSizeVID / 4 + (y / 2 + row) * (frameWidth / 2) + (x / 2 + col));

                __m128i imgV = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&imageYUV[imageIndex]));

                _mm_storeu_si128(reinterpret_cast<__m128i*>(&frameYUV[frameIndex]), imgV);
            }
        }

}

void overlayImageOnFrameParallel(const std::vector<uint8_t>& imageYUV, std::vector<uint8_t>& frameYUV, int frameWidth, int frameHeight, int imageWidth, int imageHeight, int x = 0, int y = 0) {
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        numThreads = 1; // Если не удается получить количество ядер, используем один поток
    }

    // Создаем вектор потоков
    std::vector<std::thread> threads;

    // Разбиваем область обработки на части для каждого потока
    int rowsPerThread = imageHeight / numThreads;
    int startY = 0;
    int endY = 0;


    for (int i = 0; i < numThreads; i++) {
        startY = endY;
        endY = (i == numThreads - 1) ? imageHeight : startY + rowsPerThread;

        // Создаем поток и передаем ему свою область обработки
        threads.emplace_back(processImageRows, std::ref(imageYUV), std::ref(frameYUV), frameWidth, frameHeight, imageWidth, imageHeight, x, y, startY, endY);
    }


    // Дожидаемся завершения всех потоков
    for (std::thread& thread : threads) {
        thread.join();
    }
}


void rgbToYuv420Parallel(const std::vector<uint8_t>& rgb, std::vector<uint8_t>& yuv420, int width, int height) {
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        numThreads = 1; // Если не удается получить количество ядер, используем один поток
    }

    // Определение количества полных пар строк для каждого потока
    int fullPairsPerThread = height / (2 * numThreads);
    int remainingRows = height % (2 * numThreads);

    // Создаем вектор потоков
    std::vector<std::thread> threads;

    int startY = 0;
    int endY = 0;

    for (int i = 0; i < numThreads; i++) {
        startY = endY;
        int fullRows = fullPairsPerThread * 2;

        // Если остались нечетные строки, добавляем их к последнему потоку
        if (i == numThreads - 1 && remainingRows > 0) {
            fullRows += remainingRows;
        }

        endY = startY + fullRows;

        // Создаем поток и передаем ему свою область обработки
        threads.emplace_back(rgbToYuv420, std::ref(rgb), std::ref(yuv420), width, height, startY, endY);
    }

    // Дожидаемся завершения всех потоков
    for (std::thread& thread : threads) {
        thread.join();
    }
}


int main(void) {
    std::string inputImagePath, inputVideoPath, outputVideoName, outputImageName;
    int width = 0, height = 0, PicWigth = 0, PicHeight = 0, YuvWigth = 1280, YuvHeight = 720;
    // Запрос пути к исходному изображению BMP
    std::cout << "Input bmp path: ";
    std::cin >> inputImagePath;

    // Загрузка изображения BMP
    std::ifstream imageFile(inputImagePath, std::ios::binary);
    if (!imageFile.is_open()) {
        std::cerr << "Error" << std::endl;
        return 1;
    }

    // Определение размеров изображения BMP
    imageFile.seekg(18); // Перемещение к 18 байту, где хранятся размеры

    imageFile.read(reinterpret_cast<char*>(&width), sizeof(int));
    imageFile.read(reinterpret_cast<char*>(&height), sizeof(int));

    // Чтение данных изображения BMP
    std::vector<uint8_t> rgba(width * height * 4);
    imageFile.seekg(54); // Перемещение к данным пикселей
    imageFile.read(reinterpret_cast<char*>(rgba.data()), rgba.size());
    imageFile.close();

    // Запрос пути к видео в формате YUV
    std::cout << "Input video path: ";
    std::cin >> inputVideoPath;

    // Определение пути к папке из пути к исходному видео
    size_t lastSlashPos = inputVideoPath.find_last_of("/\\");
    std::string outputVideoPath = inputVideoPath.substr(0, lastSlashPos + 1);


    // Запрос нового имени файла для сохранения видео
    std::cout << "Input name video: ";
    std::cin >> outputVideoName;

    // Открытие видео в формате YUV для чтения
    std::ifstream videoFile(inputVideoPath, std::ios::binary);
    if (!videoFile.is_open()) {
        std::cerr << "Error." << std::endl;
        return 1;
    }




    // Открытие нового файла для записи видео
    std::ofstream outputVideo(outputVideoPath + outputVideoName + ".yuv", std::ios::binary);
    if (!outputVideo.is_open()) {
        std::cerr << "Error" << std::endl;
        return 1;
    }



    // Конвертация изображения BMP в YUV420



    std::vector<uint8_t> yuv420;
    rgbToYuv420Parallel(rgba, yuv420, width, height);




    // Чтение и обработка кадров видео в формате YUV
    auto time = std::chrono::steady_clock::now();
    {
        std::vector<uint8_t> yuvFrame(YuvWigth * YuvHeight * 3 / 2); // YUV420
        while (videoFile.read(reinterpret_cast<char*>(yuvFrame.data()), yuvFrame.size())) {

            // Наложение изображения на кадр видео
            overlayImageOnFrameParallel(yuv420, yuvFrame, YuvWigth, YuvHeight, width, height);


            // Запись обработанного кадра в выходное видео
            outputVideo.write(reinterpret_cast<char*>(yuvFrame.data()), yuvFrame.size());
        }
    }
    //отсечка времени цикла.
    auto  duration = std::chrono::steady_clock::now()- time;
    std::cout << "time=" << duration.count() << std::endl;



    videoFile.close();
    outputVideo.close();
    std::cout << "YuvWigth" << YuvWigth << std::endl << "YuvHeight" << YuvHeight << std::endl << "width" << width << std::endl << "height" << height << std::endl;

    std::cout << "Success as " << outputVideoPath << outputVideoName << ".yuv" << std::endl;

    // Запрос нового имени файла для сохранения изображения
    std::cout << "Input name image: ";
    std::cin >> outputImageName;

    // Создание пути к сохраняемому изображению
    std::string outputImagePath = inputVideoPath.substr(0, lastSlashPos + 1) + outputImageName + ".yuv";

    // Сохранение изображения BMP после конвертации в YUV420
    std::ofstream outputImage(outputImagePath, std::ios::binary);
    if (!outputImage.is_open()) {
        std::cerr << "Error" << std::endl;
        return 1;
    }
    outputImage.write(reinterpret_cast<char*>(yuv420.data()), yuv420.size());
    outputImage.close();
    std::cout << "Image saved as " << outputImagePath << std::endl;

    return 0;
}