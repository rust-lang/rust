#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

if (${LLVM_VERSION_MAJOR} LESS_EQUAL 7)
add_llvm_loadable_module( LLVMEnzyme
    Enzyme.cpp
    #    DEPENDS
    #intrinsics_gen
    #PLUGIN_TOOL
    #opt
  )
else()
add_llvm_library( LLVMEnzyme
    Enzyme.cpp
    #    DEPENDS
    #intrinsics_gen
    #PLUGIN_TOOL
    #opt
  )

endif()
