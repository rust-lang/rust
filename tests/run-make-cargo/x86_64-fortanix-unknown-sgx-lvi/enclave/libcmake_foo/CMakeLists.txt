enable_language(C CXX ASM)

set(C_SOURCES
    src/foo.c
    )

set_source_files_properties(${C_SOURCES}
                            PROPERTIES
                            LANGUAGE C)

set(CXX_SOURCES
    src/foo_cxx.cpp
    )

set_source_files_properties(${CXX_SOURCES}
                            PROPERTIES
                            LANGUAGE CXX)

set(ASM_SOURCES
    src/foo_asm.s
    )

set_source_files_properties(${ASM_SOURCES}
                            PROPERTIES
                            LANGUAGE ASM)

set(SOURCES
    ${C_SOURCES}
    ${CXX_SOURCES}
    ${ASM_SOURCES})
        
add_library(cmake_foo STATIC
    ${SOURCES})
