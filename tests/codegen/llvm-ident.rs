// Verifies that the `!llvm.ident` named metadata is emitted.
//
// revisions: NONE OPT DEBUG
//
// [OPT] compile-flags: -Copt-level=2
// [DEBUG] compile-flags: -Cdebuginfo=2

// The named metadata should contain a single metadata node (see
// `LLVMRustPrepareThinLTOImport` for details).
// CHECK: !llvm.ident = !{![[ID:[0-9]+]]}

// In addition, check that the metadata node has the expected content.
// CHECK: ![[ID]] = !{!"rustc version 1.{{.*}}"}

fn main() {}
