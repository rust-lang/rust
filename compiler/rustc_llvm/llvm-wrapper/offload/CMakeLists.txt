cmake_minimum_required(VERSION 3.20)
project(RustOffload LANGUAGES CXX)

find_package(LLVM CONFIG REQUIRED)

add_library(RustOffload-${LLVM_VERSION_MAJOR} SHARED
    OffloadWrapper.cpp
)

target_include_directories(RustOffload-${LLVM_VERSION_MAJOR} PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(RustOffload-${LLVM_VERSION_MAJOR} PRIVATE
    LLVM
)

if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(
        RustOffload-${LLVM_VERSION_MAJOR}
        PRIVATE -fno-rtti
    )
endif()

install(TARGETS RustOffload-${LLVM_VERSION_MAJOR}
    LIBRARY DESTINATION lib
)
