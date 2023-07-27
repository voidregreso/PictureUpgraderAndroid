# Colorize and Super-Resolution Android App

Colorize your black and white photos and perform super-resolution enhancement in just a few easy steps on your Android phone. The application is powered by [Tencent's NCNN](https://github.com/Tencent/ncnn) and [OpenCV](https://opencv.org/) library and written both in Kotlin and Native C++.

## Features

* **Colorize B/W Photos:** The app allows you to easily colorize your black and white photos.
* **Super-Resolution:** Enhance the resolution of your photos with the super-resolution feature.
* **Down sampling:** Compresses the selected image resolution to reduce the pressure of super-resolution.
* **Ease of Use:** User-friendly interface, which enables you to use the app easily without any technical skills.

## Requirements

* Android SDK version 23 or higher
* NDK 24 or higher with CMake support
* OpenCV & NCNN libraries

## Installation

1. Clone the project
    ```
    git clone https://github.com/uvoidregreso/PictureUpgraderAndroid.git
    ```
    
2. Download *NCNN & OpenCV prebuilt libraries* and *model assets* from releases of this repository.

3. Open the project with Android Studio.

4. Unzip and place the downloaded *OpenCV & NCNN prebuilt libraries* under `app/src/main/cpp`, and as for the downloaded *model assets*, do in the same way under `app/src/main/assets`.

5. Configure the path to the header files and libraries of OpenCV & NCNN in `CMakeLists.txt`. NCNN with Vulkan acceleration are recommended to fully utilize the phone's GPU resources and significantly improve inference performance.

6. Build and run the project.

## How to Use

1. Open the app, then click on 'Select' button and select a photo from your gallery.

2. Click on the 'Colorization' button to colorize your B/W photo.

3. If you want to enhance the resolution of your photo (especially for portrait restoration), click on the 'Super-Resolution' button.

4. At last, if the resolution of the original image is too large, but the details are blurred, you can click the "Reduce Resolution" button to reduce the resolution of the image, and then proceed to the super-resolution operation to prevent the program from crashing or taking a long time to perform it.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/voidregreso/PictureUpgraderAndroid/issues).
