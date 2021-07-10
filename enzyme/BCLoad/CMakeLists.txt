set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    )
else()
    add_llvm_library( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
endif()
endif()

if (APPLE)
# Darwin-specific linker flags for loadable modules.
set_target_properties(BCPass-${LLVM_VERSION_MAJOR} PROPERTIES
    LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()
