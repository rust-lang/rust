# Run regression and unit tests
if (${Clang_FOUND})
add_lit_testsuite(check-bcpass "Running BCPass regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS BCPass-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-bcpass PROPERTIES FOLDER "Tests")
endif()
