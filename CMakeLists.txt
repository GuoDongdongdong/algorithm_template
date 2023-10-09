cmake_minimum_required(VERSION 3.16)

project(leetcode
    VERSION 0.1.0
    LANGUAGES CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
target_compile_features(main PRIVATE cxx_std_17)