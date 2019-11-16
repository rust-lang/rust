# Run regression and unit tests
add_lit_testsuite(bench-taylorlog "Running enzyme bencmarks tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)
