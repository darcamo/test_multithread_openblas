cmake_minimum_required(VERSION 3.14)
project(test_multithread_openblas)

set(CMAKE_CXX_STANDARD 17)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# If the automatic support using conan-cmake does not work because you don't have a system-wide conan instalation (you
# have installed conan in a virtual environment) then use the two lines below and manually call "conan install .." from
# the build folder.
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
  conan_basic_setup(TARGETS)
else()
  message(FATAL_ERROR "The file conanbuildinfo.cmake doesn't exist! Run the command below from each build folder
    conan install .. --build=missing -s build_type=Release")
endif()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

add_executable(test_multithread_openblas
  main.cpp
  ThreadPool.h
  )
target_link_libraries(test_multithread_openblas CONAN_PKG::armadillo)
target_link_libraries(test_multithread_openblas CONAN_PKG::gsl_microsoft)
target_link_libraries(test_multithread_openblas pthread)
