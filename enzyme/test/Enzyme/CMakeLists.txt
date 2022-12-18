add_subdirectory(ReverseMode)
add_subdirectory(ReverseModeVector)
add_subdirectory(ForwardMode)
add_subdirectory(ForwardModeSplit)
add_subdirectory(ForwardModeVector)
add_subdirectory(BatchMode)

# Run regression and unit tests
add_lit_testsuite(check-enzyme "Running enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

add_lit_testsuite(check-enzyme-bench "Benchmarking enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v --time-tests -o results.json
)


set_target_properties(check-enzyme PROPERTIES FOLDER "Tests")
set_target_properties(check-enzyme-bench PROPERTIES FOLDER "Tests")