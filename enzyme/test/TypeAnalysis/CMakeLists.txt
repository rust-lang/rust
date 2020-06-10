# Run regression and unit tests
add_lit_testsuite(check-typeanalysis "Running enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-typeanalysis PROPERTIES FOLDER "Tests")
