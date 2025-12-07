cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Print OpenCV info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Set source files
set(SOURCES
    main.cpp
    functions.cpp
    # Добавьте здесь другие .cpp файлы, если они есть
)

# Set header files
set(HEADERS
    functions.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link OpenCV
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Additional include directories if you have headers in subfolders
# target_include_directories(${PROJECT_NAME} PRIVATE include)
# target_include_directories(${PROJECT_NAME} PRIVATE src)