#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

file(GLOB ENZYME_SRC
    "*.cpp"
)

list(APPEND ENZYME_SRC SCEV/ScalarEvolutionExpander.cpp)

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
