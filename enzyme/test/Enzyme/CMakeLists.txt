add_subdirectory(ReverseMode)
add_subdirectory(ForwardMode)

# Run regression and unit tests
add_lit_testsuite(check-enzyme "Running enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme PROPERTIES FOLDER "Tests")