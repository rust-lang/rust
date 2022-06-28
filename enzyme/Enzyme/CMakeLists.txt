#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

set(Enzyme_TABLEGEN_EXE enzyme-tblgen)

function(enzyme_tablegen ofn)
  if (${LLVM_VERSION_MAJOR} GREATER 11)
    if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
      message("Using fallback code. Please switch to a newer cmake")
      tablegenSubstitute(Enzyme ${ARGV})
    else()
      tablegen(Enzyme ${ARGV})
    endif()
  else()
    tablegen(Enzyme ${ARGV})
  endif()


  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(tablegenSubstitute project ofn)
  # Validate calling context.
  if(NOT ${project}_TABLEGEN_EXE)
    message(FATAL_ERROR "${project}_TABLEGEN_EXE not set")
  endif()

  # Use depfile instead of globbing arbitrary *.td(s) for Ninja.
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    # Make output path relative to build.ninja, assuming located on
    # ${CMAKE_BINARY_DIR}.
    # CMake emits build targets as relative paths but Ninja doesn't identify
    # absolute path (in *.d) as relative path (in build.ninja)
    # Note that tblgen is executed on ${CMAKE_BINARY_DIR} as working directory.
    file(RELATIVE_PATH ofn_rel
      ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
    set(additional_cmdline
      -o ${ofn_rel}
      -d ${ofn_rel}.d
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.d
      )
    set(local_tds)
    set(global_tds)
  else()
    file(GLOB local_tds "*.td")
    file(GLOB_RECURSE global_tds "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")
    set(additional_cmdline
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      )
  endif()

  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()
  if (LLVM_ENABLE_DAGISEL_COV)
    list(FIND ARGN "-gen-dag-isel" idx)
    if( NOT idx EQUAL -1 )
      list(APPEND LLVM_TABLEGEN_FLAGS "-instrument-coverage")
    endif()
  endif()
  if (LLVM_ENABLE_GISEL_COV)
    list(FIND ARGN "-gen-global-isel" idx)
    if( NOT idx EQUAL -1 )
      list(APPEND LLVM_TABLEGEN_FLAGS "-instrument-gisel-coverage")
      list(APPEND LLVM_TABLEGEN_FLAGS "-gisel-coverage-file=${LLVM_GISEL_COV_PREFIX}all")
    endif()
  endif()
  # Comments are only useful for Debug builds. Omit them if the backend
  # supports it.
  if (NOT (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR
           uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO"))
    list(FIND ARGN "-gen-dag-isel" idx)
    if (NOT idx EQUAL -1)
      list(APPEND LLVM_TABLEGEN_FLAGS "-omit-comments")
    endif()
  endif()

  # MSVC can't support long string literals ("long" > 65534 bytes)[1], so if there's
  # a possibility of generated tables being consumed by MSVC, generate arrays of
  # char literals, instead. If we're cross-compiling, then conservatively assume
  # that the source might be consumed by MSVC.
  # [1] https://docs.microsoft.com/en-us/cpp/cpp/compiler-limits?view=vs-2017
  if (MSVC AND project STREQUAL LLVM)
    list(APPEND LLVM_TABLEGEN_FLAGS "--long-string-literals=0")
  endif()
  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio has problems with llvm-tblgen's native --write-if-changed
    # behavior. Since it doesn't do restat optimizations anyway, just don't
    # pass --write-if-changed there.
    set(tblgen_change_flag)
  else()
    set(tblgen_change_flag "--write-if-changed")
  endif()

  # We need both _TABLEGEN_TARGET and _TABLEGEN_EXE in the  DEPENDS list
  # (both the target and the file) to have .inc files rebuilt on
  # a tablegen change, as cmake does not propagate file-level dependencies
  # of custom targets. See the following ticket for more information:
  # https://cmake.org/Bug/view.php?id=15858
  # The dependency on both, the target and the file, produces the same
  # dependency twice in the result file when
  # ("${${project}_TABLEGEN_TARGET}" STREQUAL "${${project}_TABLEGEN_EXE}")
  # but lets us having smaller and cleaner code here.
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)

  # we are here since the used cmake doesn't support the TRANSFORM command
  # so we manually implement it
  #list(TRANSFORM tblgen_includes PREPEND -I)
  foreach(incl ${tblgen_includes})
          list(APPEND tblgen_includes_prefixed "-I ${incl}")
  endforeach()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND ${${project}_TABLEGEN_EXE} ${ARGN} -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${tblgen_includes_prefixed}
    ${LLVM_TABLEGEN_FLAGS}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${tblgen_change_flag}
    ${additional_cmdline}
    # The file in LLVM_TARGET_DEFINITIONS may be not in the current
    # directory and local_tds may not contain it, so we must
    # explicitly list it here:
    DEPENDS ${${project}_TABLEGEN_TARGET} ${${project}_TABLEGEN_EXE}
      ${local_tds} ${global_tds}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    COMMENT "Building ${ofn}..."
    )

  # `make clean' must remove all those generated files:
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})

  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES
    GENERATED 1)
endfunction()


set(LLVM_TARGET_DEFINITIONS InstructionDerivatives.td)
enzyme_tablegen(InstructionDerivatives.inc -gen-derivatives)
add_public_tablegen_target(InstructionDerivativesIncGen)

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
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} InstructionDerivativesIncGen)
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
