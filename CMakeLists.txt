cmake_minimum_required(VERSION 3.4...3.18)
project(objdetecter)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# set C++ standard
set(CMAKE_CXX_STANDARD 11)

# set build type
# if (NOT CMAKE_BUILD_TYPE)
# #set(CMAKE_BUILD_TYPE Release)
# endif ()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# set install prefix
# install target and libraries
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
#当前文件夹
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# set install rpath
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "lib")

# add pybind11 subdirectory
set(MY_PYBIND ${CMAKE_SOURCE_DIR}/pybind11)
add_subdirectory(${MY_PYBIND})

# set demo sources
set(DEMO_SOURCES ${CMAKE_SOURCE_DIR}/src/objdetecter.cc  ${CMAKE_SOURCE_DIR}/src/postprocess.cc ${CMAKE_SOURCE_DIR}/src/mat_warper.cpp)

# add demo module
pybind11_add_module(objdetecter ${DEMO_SOURCES})

# set demo include directories
# target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(objdetecter PRIVATE ${CMAKE_SOURCE_DIR}/include)

# find opencv package
find_package(OpenCV 4.5.3 REQUIRED core imgproc highgui)

# link opencv libraries
target_link_libraries(objdetecter PRIVATE ${OpenCV_LIBS})

# set rknn api path
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)

# set lib arch
set(LIB_ARCH aarch64)

# set rknn rt lib
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# set rknn api include directory
target_include_directories(objdetecter PRIVATE ${RKNN_API_PATH}/include)

# set 3rdparty include directory
target_include_directories(objdetecter PRIVATE ${CMAKE_SOURCE_DIR}/../3rdparty)

# link rknn rt lib
target_link_libraries(objdetecter PRIVATE ${RKNN_RT_LIB})

# set rga path
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga/RK3588)

# set rga lib
set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)

# set rga include directory
target_include_directories(objdetecter PRIVATE ${RGA_PATH}/include)

# link rga lib
target_link_libraries(objdetecter PRIVATE ${RGA_LIB})

# set version info
target_compile_definitions(objdetecter PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} PUBLIC)

# set library output directory
#set_target_properties(demo PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# install target and libraries
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
#install(TARGETS demo DESTINATION ./)

#install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
#install(PROGRAMS ${RGA_LIB} DESTINATION lib)
#install(DIRECTORY model DESTINATION ./)