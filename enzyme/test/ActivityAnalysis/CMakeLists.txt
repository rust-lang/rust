# Run regression and unit tests
add_lit_testsuite(check-activityanalysis "Running enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-activityanalysis PROPERTIES FOLDER "Tests")
