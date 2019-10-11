project(Enzyme)

SET(CMAKE_CXX_FLAGS "-Wall")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")

SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fno-omit-frame-pointer")

#SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fno-omit-frame-pointer -fsanitize=address")
#SET(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 3.5)

set(ENZYME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ENZYME_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")
message( LLVM_SHLIBEXT = ${LLVM_SHLIBEXT} )

get_filename_component(LLVM_ABSOLUTE_DIR
    "${LLVM_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

set(LLVM_DIR "${LLVM_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)

message("found llvm dir " ${LLVM_DIR})

get_filename_component(LLVM_ABSOLUTE_LIT
                       "${LLVM_EXTERNAL_LIT}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

                   set(LLVM_EXTERNAL_LIT "${LLVM_ABSOLUTE_LIT}" CACHE FILEPATH "a" FORCE)
message("found llvm lit " ${LLVM_EXTERNAL_LIT})


list(INSERT CMAKE_PREFIX_PATH 0 "${LLVM_DIR}")
message("CMAKE_PREFIX_PATH " ${CMAKE_PREFIX_PATH})
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
message("found llvm include directory here " ${LLVM_INCLUDE_DIRS})
message("found llvm definitions " ${LLVM_DEFINITIONS})
message("found llvm version " ${LLVM_VERSION_MAJOR})

add_subdirectory(Enzyme)
add_subdirectory(test)
add_subdirectory(functional_tests_c)
