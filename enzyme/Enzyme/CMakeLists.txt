#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

file(GLOB ENZYME_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cpp"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND ENZYME_SRC SCEV/ScalarEvolutionExpander.cpp)

list(APPEND ENZYME_SRC  TypeAnalysis/TypeTree.cpp TypeAnalysis/TypeAnalysis.cpp TypeAnalysis/TypeAnalysisPrinter.cpp)

message("found enzyme sources " ${ENZYME_SRC})

if (${LLVM_VERSION_MAJOR} LESS 8)
    add_llvm_loadable_module( LLVMEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    #LINK_COMPONENTS
    #Core
    #Analysis
    #ScalarOpts
    )
    add_llvm_loadable_module( ClangEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymeClang.cpp
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    #LINK_COMPONENTS
    #Core
    #Analysis
    #ScalarOpts
    )
else()
# on windows `PLUGIN_TOOL` doesn't link against LLVM.dll
if ((WIN32 OR CYGWIN) AND LLVM_LINK_LLVM_DYLIB)
    add_llvm_library( LLVMEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        MODULE
        DEPENDS
        intrinsics_gen
	LINK_COMPONENTS
	LLVM
    )
    add_llvm_library( ClangEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymeClang.cpp
        MODULE
        DEPENDS
        intrinsics_gen
	LINK_COMPONENTS
	LLVM
    )
else()
    #set(LLVM_LINK_COMPONENTS Core Analysis ScalarOpts)
    add_llvm_library( LLVMEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        MODULE
        DEPENDS
        intrinsics_gen
        #LLVMCore
        PLUGIN_TOOL
        opt
    #LINK_COMPONENTS
    #Core
    #Analysis
    #ScalarOpts
    )
    add_llvm_library( ClangEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymeClang.cpp
        MODULE
        DEPENDS
        intrinsics_gen
        #LLVMCore
        PLUGIN_TOOL
        opt
    #LINK_COMPONENTS
    #Core
    #Analysis
    #ScalarOpts
    )
endif()
endif()

if (APPLE)
# Darwin-specific linker flags for loadable modules.
set_target_properties(LLVMEnzyme-${LLVM_VERSION_MAJOR} PROPERTIES
    LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()