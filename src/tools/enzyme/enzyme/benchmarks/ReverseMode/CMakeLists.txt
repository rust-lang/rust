# Run regression and unit tests
add_lit_testsuite(bench-enzyme-reverse "Running enzyme benchmarks tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_BENCH_DEPS}
    ARGS -v -j 1
)

set_target_properties(bench-enzyme-reverse PROPERTIES FOLDER "bench Tests")

add_subdirectory(nn)
add_subdirectory(taylorlog)
add_subdirectory(logsumexp)
add_subdirectory(matdescent)
add_subdirectory(ode)
add_subdirectory(ode-const)
add_subdirectory(ode-real)
add_subdirectory(fft)

add_subdirectory(gmm)
add_subdirectory(ba)
add_subdirectory(hand)
add_subdirectory(lstm)

add_subdirectory(library)
