include(ExternalProject)
ExternalProject_Add(adept2
    GIT_REPOSITORY https://github.com/rjhogan/Adept-2
    GIT_TAG 4dc3d72c25d120a80745c44599dc8602a4f8fdeb
    PREFIX adept2
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/adept2/install
    CONFIGURE_COMMAND autoreconf -i && ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/adept2/install 
    UPDATE_COMMAND ""
    TEST_COMMAND      ""
)
ExternalProject_Add(tapenade
    URL ${CMAKE_CURRENT_SOURCE_DIR}/ADFirstAidKit.tar
    PREFIX tapenade
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make adBinomial.o adBuffer.o adContext.o adProfile.o adStack.o
    INSTALL_COMMAND ""
    TEST_COMMAND      ""
)


set(BENCH_FLAGS "-I${CMAKE_CURRENT_BINARY_DIR}/adept2/src/adept2/include -I${CMAKE_CURRENT_BINARY_DIR}/tapenade/src/tapenade")
set(BENCH_LINK "${CMAKE_CURRENT_BINARY_DIR}/tapenade/src/tapenade/adBuffer.o ${CMAKE_CURRENT_BINARY_DIR}/tapenade/src/tapenade/adStack.o -L${CMAKE_CURRENT_BINARY_DIR}/adept2/install/lib -ladept")
set(BENCH_LDPATH "${CMAKE_CURRENT_BINARY_DIR}/adept2/install/lib")
message("found bench flags: " ${BENCH_FLAGS})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_BENCH_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR} adept2 tapenade)

# Run regression and unit tests
add_lit_testsuite(bench-enzyme "Running enzyme benchmarks tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_BENCH_DEPS}
    ARGS -v
)

set_target_properties(bench-enzyme PROPERTIES FOLDER "bench Tests")

add_subdirectory(nn)
add_subdirectory(taylorlog)
add_subdirectory(logsumexp)
add_subdirectory(matdescent)

add_subdirectory(gmm)
add_subdirectory(ba)
add_subdirectory(hand)
