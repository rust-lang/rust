#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

if (${LLVM_VERSION_MAJOR} LESS 8)
    add_llvm_loadable_module( LLVMEnzyme-${LLVM_VERSION_MAJOR}
    Enzyme.cpp
    Utils.cpp
    SCEV/ScalarEvolutionExpander.cpp
    DEPENDS
    intrinsics_gen
    PLUGIN_TOOL
    opt
  )
else()
    add_llvm_library( LLVMEnzyme-${LLVM_VERSION_MAJOR}
    Enzyme.cpp
    Utils.cpp
    SCEV/ScalarEvolutionExpander.cpp
    MODULE
    DEPENDS
    intrinsics_gen
    PLUGIN_TOOL
    opt
  )

endif()
