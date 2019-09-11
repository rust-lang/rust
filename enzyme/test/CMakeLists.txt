configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_TEST_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR})

set(ENZYME_TESTSUITES)
list(APPEND ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg)

# Run regression and unit tests
add_lit_testsuite(check-enzyme "Running enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme PROPERTIES FOLDER "Tests")

#add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
    #    DEPENDS ${ENZYME_TEST_DEPS}
#)
