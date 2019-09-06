#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

add_llvm_loadable_module( LLVMEnzyme
    Enzyme.cpp
    #    DEPENDS
    #intrinsics_gen
    #PLUGIN_TOOL
    #opt
  )
