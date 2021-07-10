# Run regression and unit tests
add_lit_testsuite(check-bcpass "Running BCPass regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-bcpass PROPERTIES FOLDER "Tests")

