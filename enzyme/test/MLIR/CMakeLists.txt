# Run regression and unit tests
add_lit_testsuite(check-enzymemlir "Running MLIR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir PROPERTIES FOLDER "Tests")
