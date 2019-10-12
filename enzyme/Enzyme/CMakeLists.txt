#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

file(GLOB ENZYME_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cpp"
)

list(APPEND ENZYME_SRC SCEV/ScalarEvolutionExpander.cpp)

message("found enzyme sources " ${ENZYME_SRC})

if (${LLVM_VERSION_MAJOR} LESS 8)
    add_llvm_loadable_module( LLVMEnzyme-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
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

endif()
