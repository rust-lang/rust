project(Enzyme)

cmake_minimum_required(VERSION 3.9)

SET(CMAKE_CXX_FLAGS "-Wall -fPIC -fno-rtti")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")

SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -ggdb -fno-omit-frame-pointer")

#SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fno-omit-frame-pointer -fsanitize=address")
#SET(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(ENZYME_EXTERNAL_SHARED_LIB "Build external shared library" OFF)
set(ENZYME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ENZYME_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${ENZYME_SOURCE_DIR}/cmake/modules")

set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")
if (APPLE)
    set(LLVM_SHLIBEXT ".dylib")
endif()
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
message("LLVM_INSTALL_PREFIX: ${LLVM_INSTALL_PREFIX}")
message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("found llvm definitions " ${LLVM_DEFINITIONS})
message("found llvm version " ${LLVM_VERSION_MAJOR})

list(GET LLVM_INCLUDE_DIRS 0 LLVM_IDIR)
message("first llvm include directory" ${LLVM_IDIR})

file(READ ${LLVM_IDIR}/llvm/Analysis/ScalarEvolution.h INPUT_TEXT)
string(REPLACE private public INPUT_TEXT "${INPUT_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolution.h" "${INPUT_TEXT}")


file(READ ${LLVM_IDIR}/llvm/Analysis/TargetLibraryInfo.h INPUT_TEXT)
string(REPLACE "class TargetLibraryInfo {" "class TargetLibraryInfo {public:" INPUT_TEXT "${INPUT_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/TargetLibraryInfo.h" "${INPUT_TEXT}")

if (${LLVM_VERSION_MAJOR} GREATER_EQUAL 11)
file(READ ${LLVM_IDIR}/llvm/Transforms/Utils/ScalarEvolutionExpander.h INPUT_TEXT)
else()
file(READ ${LLVM_IDIR}/llvm/Analysis/ScalarEvolutionExpander.h INPUT_TEXT)
endif()

find_package(MPI)
if (${MPI_FOUND})
    add_definitions(-DBUILDMPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
else()
    set(MPI_C_LIBRARIES "")
endif()

if (${LLVM_VERSION_MAJOR} LESS 12)
string(REPLACE "#define LLVM_ANALYSIS_SCALAREVOLUTIONEXPANDER_H" "#define LLVM_ANALYSIS_SCALAREVOLUTIONEXPANDER_H\n#include \"SCEV/ScalarEvolution.h\""  INPUT_TEXT "${INPUT_TEXT}")
string(REPLACE "LLVM_ANALYSIS" "FAKELLVM_ANALYSIS" INPUT_TEXT "${INPUT_TEXT}")
string(REPLACE "class SCEVExpander " "namespace fake {\n  class SCEVExpander " INPUT_TEXT "${INPUT_TEXT}")
string(REPLACE "struct SCEVOperand " "namespace fake {\n  struct SCEVOperand " INPUT_TEXT "${INPUT_TEXT}")
string(REPLACE "};\n}" "};\n}}" INPUT_TEXT "${INPUT_TEXT}")
string(REPLACE "const SCEV* S;\n};\n" "const SCEV* S;\n};\n}\n" INPUT_TEXT "${INPUT_TEXT}")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolutionExpander.h" "${INPUT_TEXT}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

add_subdirectory(Enzyme)
add_subdirectory(test)

# The benchmarks data are not in git-exported source archives to minimize size.
# Only add the benchmarks if the directory exists.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()
