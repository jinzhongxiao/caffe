cmake_minimum_required(VERSION 2.8.3)
project(test)

   SET(CMAKE_BUILD_TYPE "Debug") 

   SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

       SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_definitions(-DCPU_ONLY=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CAFFE_INCLUDEDIR /home/roger/catkin_ws/src/ros_caffe/caffe/include /home/roger/catkin_ws/src/ros_caffe/caffe/distribute/include)
set(CAFFE_LINK_LIBRARAY /home/roger/catkin_ws/src/ros_caffe/caffe/build/lib)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
include_directories(
  ${CAFFE_INCLUDEDIR}
  ${OpenCV_INCLUDE_DIRS}
)
link_directories(
  /usr/local/lib
  /usr/lib
  ${CAFFE_LINK_LIBRARAY}
)
add_executable(main main.cpp)
target_link_libraries(main  ${OpenCV_LIBRARIES} caffe glog)