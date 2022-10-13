cmake_minimum_required(VERSION 3.9)
project(BLASHeader)

file(GLOB BLAS_LL "${CMAKE_CURRENT_SOURCE_DIR}/src/gsl/*.ll")
set (NEED_COMMA FALSE)
list(FILTER BLAS_LL EXCLUDE REGEX ".*test.*")

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "")
foreach(file ${BLAS_LL})
    get_filename_component(variableName ${file} NAME_WE)

    file(READ ${file} hexString HEX)

    set(hexString "${hexString}00")

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arrayValues ${hexString})
    string(REGEX REPLACE ",$" "" arrayValues ${arrayValues})

    file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "const char __data_${variableName}[] = {${arrayValues}};\n")
endforeach()

file(GLOB BLAS64_LL "${CMAKE_CURRENT_SOURCE_DIR}/../gsl64/src/gsl64/*.ll")
list(FILTER BLAS64_LL EXCLUDE REGEX ".*test.*")

foreach(file ${BLAS64_LL})
    get_filename_component(variableName ${file} NAME_WE)

    file(READ ${file} hexString HEX)

    set(hexString "${hexString}00")

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arrayValues ${hexString})
    string(REGEX REPLACE ",$" "" arrayValues ${arrayValues})

    file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "const char __data_${variableName}64_[] = {${arrayValues}};\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "std::map<std::string, const char*> DATA = {\n")
foreach(file ${BLAS_LL})
    get_filename_component(variableName ${file} NAME_WE)
    # declares byte array and the length variables
    if (${NEED_COMMA})
        file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h ",\n")
    endif()
    set(arrayDefinition "{ \"cblas_${variableName}\",  __data_${variableName} }")
    file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "${arrayDefinition}")
    set (NEED_COMMA TRUE)
endforeach()
foreach(file ${BLAS64_LL})
    get_filename_component(variableName ${file} NAME_WE)
    # declares byte array and the length variables
    if (${NEED_COMMA})
        file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h ",\n")
    endif()
    set(arrayDefinition "{ \"cblas_${variableName}64_\",  __data_${variableName}64_ }")
    file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "${arrayDefinition}")
    set (NEED_COMMA TRUE)
endforeach()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/blas_headers.h "\n};\n")

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../fblas/src/fblas/bclib32.ll" hexString HEX)
set(hexString "${hexString}00")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arrayValues ${hexString})
string(REGEX REPLACE ",$" "" arrayValues ${arrayValues})
file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "const char __data_fblas32[] = {${arrayValues}};\n")

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../fblas/src/fblas/bclib64.ll" hexString HEX)
set(hexString "${hexString}00")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arrayValues ${hexString})
string(REGEX REPLACE ",$" "" arrayValues ${arrayValues})
file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.h "const char __data_fblas64[] = {${arrayValues}};\n")

