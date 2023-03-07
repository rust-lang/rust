add_mlir_interface(AutoDiffOpInterface)

set(LLVM_TARGET_DEFINITIONS AutoDiffTypeInterface.td)
mlir_tablegen(AutoDiffTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(AutoDiffTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRAutoDiffTypeInterfaceIncGen)

add_mlir_library(MLIREnzymeAutoDiffInterface
  AutoDiffOpInterface.cpp
  AutoDiffTypeInterface.cpp
  GradientUtils.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  MLIRAutoDiffTypeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
