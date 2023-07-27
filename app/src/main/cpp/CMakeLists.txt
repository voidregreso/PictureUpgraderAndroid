# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
set(target pictureupgrader)
project(${target})

# Import OpenCV dependencies
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-4.6.0-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

# Import NCNN dependencies
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-custom-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Set key properties
set_target_properties(
    ncnn PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-frtti;-fexceptions"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frtti -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()

# Define source files
file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

add_library( # Sets the name of the library.
        ${target}
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${srcs} ${hdrs})

#find_library(log-lib log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

include_directories("${CMAKE_CURRENT_LIST_DIR}")

target_link_libraries( # Specifies the target library. ${log-lib}
        ${target} ncnn ${OpenCV_LIBS} ${ANDROID_OPENCV_COMPONENTS})