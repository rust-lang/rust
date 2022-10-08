set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BC_LOAD_FLAGS "" CACHE STRING "")
set(BC_LOAD_HEADER "" CACHE STRING "")

if (NOT ("${BC_LOAD_HEADER}" STREQUAL ""))
    add_custom_target(blasheaders mkdir -p "${CMAKE_CURRENT_BINARY_DIR}/gsl" && cp ${BC_LOAD_HEADER} "${CMAKE_CURRENT_BINARY_DIR}/gsl/blas_headers.h")
set_target_properties(blasheaders PROPERTIES EXCLUDE_FROM_ALL TRUE)
elseif (${Clang_FOUND})
include(ExternalProject)
ExternalProject_Add(gsl
    GIT_REPOSITORY https://github.com/EnzymeAD/gsl
    GIT_TAG 5805b8323f75834c121fd65464cc55f2828f5b05
    PREFIX gsl
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/gsl/install
    CONFIGURE_COMMAND sh -c "cp sys/gsl_sys.h . && cp cblas/gsl_cblas.h . && rm cblas/test*.c && echo > config.h"
    BUILD_COMMAND  sh -c "$<TARGET_FILE:clang> cblas/*.c ${BC_LOAD_FLAGS} -I . -I .. -S -emit-llvm -O1"
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TEST_COMMAND      ""
)
set_target_properties(gsl PROPERTIES EXCLUDE_FROM_ALL TRUE)


ExternalProject_Add(openblas
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS
    GIT_TAG 18b19d135b0e80219531219a90c1b78dd9249ad0
    PREFIX openblas
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/openblas/install
    CONFIGURE_COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/fixopenblas.sh $<TARGET_FILE:clang> 
    BUILD_COMMAND  sh -c "NO_CBLAS=1 GEMM_MULTITHREAD_THRESHOLD=4 COMMON_OPT=\"\" CC=\"$<TARGET_FILE:clang> -DDTB_DEFAULT_ENTRIES=64 -DGEMM_DEFAULT_ALIGN=\"0UL\" -DGEMM_DEFAULT_OFFSET_A=0 -DGEMM_DEFAULT_OFFSET_B=0 -DGEMM_P=0 -DGEMM_Q=0 -DGEMM_MULTITHREAD_THRESHOLD=4 -Dnthreads=1 -S -emit-llvm -O1\" AR=\"echo\" make -C interface VERBOSE=1 NUM_THREADS=1"
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TEST_COMMAND      ""
)
set_target_properties(openblas PROPERTIES EXCLUDE_FROM_ALL TRUE)

ExternalProject_Add(fblas
    GIT_REPOSITORY https://github.com/EnzymeAD/clapack
    GIT_TAG 42603bf49ac2b27a7c00c53f92fa648748476510
    PREFIX fblas
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/openblas/install
    CONFIGURE_COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/fixwrapblas.sh
    BUILD_COMMAND sh -c "$<TARGET_FILE:clang> ${BC_LOAD_FLAGS} -S -emit-llvm -O1 BLAS/WRAP/bclib32.c BLAS/WRAP/bclib64.c -I ../../INCLUDE"
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TEST_COMMAND      ""
)
set_target_properties(fblas PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_custom_target(blasheaders cp "${CMAKE_CURRENT_SOURCE_DIR}/makeblas.cmake" "${CMAKE_CURRENT_BINARY_DIR}/gsl/CMakeLists.txt" && cd "${CMAKE_CURRENT_BINARY_DIR}/gsl" && ${CMAKE_COMMAND} . DEPENDS gsl fblas ${CMAKE_CURRENT_SOURCE_DIR}/makeblas.cmake)
set_target_properties(blasheaders PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if ((NOT ("${BC_LOAD_HEADER}" STREQUAL "")) OR ${Clang_FOUND})
if (${LLVM_VERSION_MAJOR} LESS 8)
    add_llvm_loadable_module( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
else()
# on windows `PLUGIN_TOOL` doesn't link against LLVM.dll
if ((WIN32 OR CYGWIN) AND LLVM_LINK_LLVM_DYLIB)
    add_llvm_library( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
	LINK_COMPONENTS
	LLVM
    BUILDTREE_ONLY
    )
else()
    add_llvm_library( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
        BUILDTREE_ONLY
    )
endif()
endif()

add_dependencies(BCPass-${LLVM_VERSION_MAJOR} blasheaders)
target_include_directories(BCPass-${LLVM_VERSION_MAJOR} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gsl)

if (APPLE)
# Darwin-specific linker flags for loadable modules.
set_target_properties(BCPass-${LLVM_VERSION_MAJOR} PROPERTIES
    LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

set_target_properties(BCPass-${LLVM_VERSION_MAJOR} PROPERTIES EXCLUDE_FROM_ALL TRUE)

if (${ENZYME_EXTERNAL_SHARED_LIB})
    add_library( EnzymeBCLoad-${LLVM_VERSION_MAJOR}
        SHARED
        BCLoader.cpp 
    )
    add_dependencies(EnzymeBCLoad-${LLVM_VERSION_MAJOR} blasheaders)
    target_include_directories(EnzymeBCLoad-${LLVM_VERSION_MAJOR} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gsl)

    target_link_libraries(EnzymeBCLoad-${LLVM_VERSION_MAJOR} LLVM)
    install(TARGETS EnzymeBCLoad-${LLVM_VERSION_MAJOR}
        EXPORT EnzymeTargets
        LIBRARY DESTINATION lib COMPONENT shlib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Enzyme"
        COMPONENT dev)
endif()
endif()
