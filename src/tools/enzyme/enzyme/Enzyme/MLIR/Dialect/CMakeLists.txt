add_mlir_dialect(EnzymeOps enzyme)
add_mlir_doc(EnzymeDialect -gen-dialect-doc EnzymeDialect Enzyme/)
add_mlir_doc(EnzymeOps -gen-op-doc EnzymeOps Enzyme/)

set(LLVM_TARGET_DEFINITIONS EnzymeOps.td)
mlir_tablegen(EnzymeEnums.h.inc -gen-enum-decls)
mlir_tablegen(EnzymeEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(EnzymeAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=enzyme)
mlir_tablegen(EnzymeAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=enzyme)
add_public_tablegen_target(MLIREnzymeEnumsIncGen)

add_mlir_dialect_library(MLIREnzyme
Dialect.cpp
Ops.cpp

ADDITIONAL_HEADER_DIRS

DEPENDS
MLIREnzymeOpsIncGen
MLIREnzymeEnumsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
)
