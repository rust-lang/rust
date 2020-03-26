configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_TEST_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR})

add_subdirectory(Enzyme)
add_subdirectory(Integration)

if(Julia_FOUND)
  add_custom_target(check-enzyme-julia
      COMMAND
          env ENZYME_PATH=${ENZYME_BINARY_DIR}/Enzyme ${Julia_EXECUTABLE} --project=${ENZYME_SOURCE_DIR}/julia -e "using Pkg; pkg\"test\""
      DEPENDS
      LLVMEnzyme-${LLVM_VERSION_MAJOR}
      VERBATIM
  )
endif()