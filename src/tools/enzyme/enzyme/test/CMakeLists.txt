configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_TEST_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR})

add_subdirectory(ActivityAnalysis)
add_subdirectory(TypeAnalysis)
add_subdirectory(Enzyme)
if (${Clang_FOUND})
add_subdirectory(Integration)
endif()
add_subdirectory(BCLoader)
if (ENZYME_MLIR)
    add_subdirectory(MLIR)
endif()
