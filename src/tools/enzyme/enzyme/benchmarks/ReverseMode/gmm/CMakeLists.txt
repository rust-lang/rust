# Run regression and unit tests
add_lit_testsuite(bench-gmm-reverse "Running enzyme benchmarks tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_BENCH_DEPS}
    ARGS -v
)
