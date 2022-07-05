cmake_minimum_required(VERSION 3.9)
project(Enzyme)

include(CMakePackageConfigHelpers)


set(ENZYME_MAJOR_VERSION 0)
set(ENZYME_MINOR_VERSION 0)
set(ENZYME_PATCH_VERSION 34)
set(ENZYME_VERSION
  ${ENZYME_MAJOR_VERSION}.${ENZYME_MINOR_VERSION}.${ENZYME_PATCH_VERSION})


SET(CMAKE_CXX_FLAGS "-Wall -fPIC -fno-rtti ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")

SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -ggdb -fno-omit-frame-pointer")

#SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fno-omit-frame-pointer -fsanitize=address")
#SET(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENZYME_CLANG "Build enzyme clang plugin" ON)
option(ENZYME_EXTERNAL_SHARED_LIB "Build external shared library" OFF)
set(ENZYME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ENZYME_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${ENZYME_SOURCE_DIR}/cmake/modules")

set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")
if (APPLE)
    set(LLVM_SHLIBEXT ".dylib")
endif()
message( LLVM_SHLIBEXT = ${LLVM_SHLIBEXT} )

set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

get_filename_component(LLVM_ABSOLUTE_DIR
    "${LLVM_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

set(LLVM_DIR "${LLVM_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)

message("CMAKE_PREFIX_PATH " ${CMAKE_PREFIX_PATH})

find_package(LLVM REQUIRED CONFIG)

if (NOT DEFINED LLVM_EXTERNAL_LIT)
  if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
      message("found llvm match ${CMAKE_MATCH_1} dir ${LLVM_DIR}")
    if (EXISTS ${LLVM_DIR}/../../../bin/llvm-lit) 
      set(LLVM_EXTERNAL_LIT ${LLVM_DIR}/../../../bin/llvm-lit) 
    endif()
  else()
    if (EXISTS ${LLVM_DIR}/bin/llvm-lit) 
      set(LLVM_EXTERNAL_LIT ${LLVM_DIR}/bin/llvm-lit) 
    endif()
  endif()
endif()

get_filename_component(LLVM_ABSOLUTE_LIT
                       "${LLVM_EXTERNAL_LIT}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

                   set(LLVM_EXTERNAL_LIT "${LLVM_ABSOLUTE_LIT}" CACHE FILEPATH "a" FORCE)
message("found llvm lit " ${LLVM_EXTERNAL_LIT})

list(INSERT CMAKE_PREFIX_PATH 0 "${LLVM_DIR}")
message("LLVM dir ${LLVM_DIR}")
if (ENZYME_CLANG)
if (DEFINED Clang_DIR)
    get_filename_component(Clang_ABSOLUTE_DIR
      "${Clang_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(Clang_DIR "${Clang_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)
  list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
  message("clang dir defined ${Clang_DIR}")
else()
  if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
    if (EXISTS ${LLVM_DIR}/../clang/../../libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR}/../clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/?$")
    if (EXISTS ${LLVM_DIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR}/lib/cmake/clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/cmake/?$")
      get_filename_component(LLDIR "${LLVM_DIR}" DIRECTORY)
    if (EXISTS ${LLDIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLDIR}/lib/cmake/clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  else()
    if (EXISTS ${LLVM_DIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR})
      list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  endif()
  message("clang dir from llvm ${Clang_DIR}")
endif()
else()
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

message("clang dir ${Clang_DIR}")

if (DEFINED Clang_DIR)
    find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} NO_DEFAULT_PATH)
if (${Clang_FOUND})
    include_directories(${CLANG_INCLUDE_DIRS})
    message("clang inc dir ${CLANG_INCLUDE_DIRS}")
    add_definitions(${CLANG_DEFINITIONS})
endif()
else()
    set(Clang_FOUND 0) 
endif()
message("found ${Clang_FOUND}")
# include(AddClang)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
message("LLVM_INSTALL_PREFIX: ${LLVM_INSTALL_PREFIX}")
message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("found llvm definitions " ${LLVM_DEFINITIONS})
message("found llvm version " ${LLVM_VERSION_MAJOR})

option(ENZYME_FLANG "Build for non-version compliant FLANG" OFF)
if (ENZYME_FLANG)
  add_definitions(-DFLANG=1)
endif()

option(ENZYME_ROCM "Build for non-version compliant ROCM" OFF)
if (ENZYME_ROCM)
  add_definitions(-DROCM=1)
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/Enzyme)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")


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

add_subdirectory(tools)
add_subdirectory(Enzyme)
add_subdirectory(BCLoad)
add_subdirectory(test)

# The benchmarks data are not in git-exported source archives to minimize size.
# Only add the benchmarks if the directory exists.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()


# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
set(var INSTALL_${p}_DIR)
if(NOT IS_ABSOLUTE "${${var}}")
  set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
endif()
endforeach()

export(TARGETS LLVMEnzyme-${LLVM_VERSION_MAJOR}
  FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")

if (${Clang_FOUND})
export(TARGETS ClangEnzyme-${LLVM_VERSION_MAJOR}
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
endif()

export(TARGETS LLDEnzyme-${LLVM_VERSION_MAJOR}
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")

export(PACKAGE Enzyme)

set(CONF_LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
set(CONF_LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR})
set(CONF_LLVM_VERSION_PATCH ${LLVM_VERSION_PATCH})
set(CONF_LLVM_DIR ${LLVM_DIR})
set(CONF_LLVM_BINARY_DIR ${LLVM_BINARY_DIR})

set(CONF_INCLUDE_DIRS lib)
configure_file(cmake/EnzymeConfig.cmake.in
     "${PROJECT_BINARY_DIR}/EnzymeConfig.cmake" @ONLY)
   
set(CONF_INCLUDE_DIRS lib)
configure_file(cmake/EnzymeConfig.cmake.in
     "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EnzymeConfig.cmake" @ONLY)

configure_file(cmake/EnzymeConfigVersion.cmake.in
     "${PROJECT_BINARY_DIR}/EnzymeConfigVersion.cmake" @ONLY)
   
install(FILES
     "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EnzymeConfig.cmake"
     "${PROJECT_BINARY_DIR}/EnzymeConfigVersion.cmake"
     DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
   
install(EXPORT EnzymeTargets DESTINATION
    "${INSTALL_CMAKE_DIR}" COMPONENT dev)
   
