configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_TEST_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR})

add_subdirectory(ActivityAnalysis)
add_subdirectory(TypeAnalysis)
add_subdirectory(Enzyme/ReverseMode)
add_subdirectory(Enzyme/ForwardMode)
add_subdirectory(Integration/ReverseMode)
add_subdirectory(Integration/ForwardMode)

add_custom_target(check-enzyme DEPENDS check-enzyme-reverse check-enzyme-forward)
add_custom_target(check-enzyme-integration DEPENDS check-enzyme-integration-reverse check-enzyme-integration-forward)