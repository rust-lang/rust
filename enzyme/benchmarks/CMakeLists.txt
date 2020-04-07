configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_BENCH_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR} adept2 tapenade)

# Run regression and unit tests
add_lit_testsuite(bench-enzyme "Running enzyme benchmarks tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_BENCH_DEPS}
    ARGS -v
)

set_target_properties(bench-enzyme PROPERTIES FOLDER "bench Tests")

add_subdirectory(nn)
add_subdirectory(taylorlog)
add_subdirectory(logsumexp)
add_subdirectory(matdescent)
