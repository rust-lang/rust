#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

set(Enzyme_TABLEGEN_EXE enzyme-tblgen)

function(enzyme_tablegen ofn)
   tablegen(Enzyme ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

set(LLVM_TARGET_DEFINITIONS InstructionDerivatives.td)
enzyme_tablegen(InstructionDerivatives.inc -gen-derivatives)
add_public_tablegen_target(MLIRArithmeticCanonicalizationIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_LINK_COMPONENTS Demangle)

file(GLOB ENZYME_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cpp"
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND ENZYME_SRC SCEV/ScalarEvolutionExpander.cpp)
list(APPEND ENZYME_SRC  TypeAnalysis/TypeTree.cpp TypeAnalysis/TypeAnalysis.cpp TypeAnalysis/TypeAnalysisPrinter.cpp TypeAnalysis/RustDebugInfo.cpp)

if (${LLVM_VERSION_MAJOR} LESS 8)
    add_llvm_loadable_module( LLVMEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
if (${Clang_FOUND})
    add_llvm_loadable_module( ClangEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymeClang.cpp
        Clang/EnzymePassLoader.cpp
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
endif()
add_llvm_loadable_module( LLDEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymePassLoader.cpp
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
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
if (${Clang_FOUND})
    add_llvm_library( ClangEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymeClang.cpp
        Clang/EnzymePassLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
	LINK_COMPONENTS
	LLVM
    )
endif()
        add_llvm_library( LLDEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymePassLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
	LINK_COMPONENTS
	LLVM
    )
else()
    add_llvm_library( LLVMEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        MODULE
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
if (${Clang_FOUND})
    add_llvm_library( ClangEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymeClang.cpp
        Clang/EnzymePassLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
endif()
    add_llvm_library( LLDEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC} Clang/EnzymePassLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
endif()
endif()

if (${ENZYME_EXTERNAL_SHARED_LIB})
    add_library( Enzyme-${LLVM_VERSION_MAJOR}
        SHARED
        ${ENZYME_SRC}
    )
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} intrinsics_gen)
    target_link_libraries(Enzyme-${LLVM_VERSION_MAJOR} LLVM)
    install(TARGETS Enzyme-${LLVM_VERSION_MAJOR}
        EXPORT EnzymeTargets
        LIBRARY DESTINATION lib COMPONENT shlib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Enzyme"
        COMPONENT dev)
endif()

if (APPLE)
# Darwin-specific linker flags for loadable modules.
set_target_properties(LLVMEnzyme-${LLVM_VERSION_MAJOR} PROPERTIES
    LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
if (${Clang_FOUND})
set_target_properties(ClangEnzyme-${LLVM_VERSION_MAJOR} PROPERTIES
        LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()
set_target_properties(LLDEnzyme-${LLVM_VERSION_MAJOR} PROPERTIES
        LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

install(TARGETS LLVMEnzyme-${LLVM_VERSION_MAJOR}
    EXPORT EnzymeTargets
    LIBRARY DESTINATION lib COMPONENT shlib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Enzyme"
    COMPONENT dev)

if (${Clang_FOUND})
install(TARGETS ClangEnzyme-${LLVM_VERSION_MAJOR}
    EXPORT EnzymeTargets
    LIBRARY DESTINATION lib COMPONENT shlib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Enzyme"
    COMPONENT dev)
endif()
install(TARGETS LLDEnzyme-${LLVM_VERSION_MAJOR}
    EXPORT EnzymeTargets
    LIBRARY DESTINATION lib COMPONENT shlib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Enzyme"
    COMPONENT dev)
