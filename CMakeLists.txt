cmake_minimum_required(VERSION 2.8)

project(credit LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

add_library(creditcu
    simulation.cu
    credit.cu)

set_target_properties(creditcu PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(creditcu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

project(creditcu LANGUAGES CXX)

add_executable(credit
    main.cpp)

target_link_libraries(credit creditcu)


