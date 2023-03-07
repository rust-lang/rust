add_mlir_library(MLIREnzymeImplementations
  ArithAutoDiffOpInterfaceImpl.cpp
  LLVMAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
)
