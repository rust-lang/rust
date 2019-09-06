project(Enzyme)

set(CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 3.10)

set(ENZYME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ENZYME_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")
message( LLVM_SHLIBEXT = ${LLVM_SHLIBEXT} )


find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

add_subdirectory(Enzyme)
add_subdirectory(test)
