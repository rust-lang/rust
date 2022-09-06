include(TableGen)

set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(enzyme-tblgen Enzyme
  enzyme-tblgen.cpp
  )

get_target_property(TBL_LINKED_LIBS LLVMSupport INTERFACE_LINK_LIBRARIES)
if (NOT TBL_LINKED_LIBS)
    target_link_libraries(enzyme-tblgen PUBLIC "stdc++")
endif()

set_target_properties(enzyme-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(enzyme-tblgen
  PRIVATE)
set(Enzyme_CONFIG_TABLEGEN_EXE enzyme-tblgen)
