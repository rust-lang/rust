//@ only-x86_64

//@ revisions: NOMODEL MODEL-SMALL MODEL-KERNEL MODEL-MEDIUM MODEL-LARGE
// [NOMODEL] no compile-flags
//@[MODEL-SMALL] compile-flags: -C code-model=small
//@[MODEL-KERNEL] compile-flags: -C code-model=kernel
//@[MODEL-MEDIUM] compile-flags: -C code-model=medium
//@[MODEL-LARGE] compile-flags: -C code-model=large

#![crate_type = "lib"]

// MODEL-SMALL: !llvm.module.flags = !{{{.*}}}
// MODEL-SMALL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 1}
// MODEL-KERNEL: !llvm.module.flags = !{{{.*}}}
// MODEL-KERNEL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 2}
// MODEL-MEDIUM: !llvm.module.flags = !{{{.*}}}
// MODEL-MEDIUM: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 3}
// MODEL-LARGE: !llvm.module.flags = !{{{.*}}}
// MODEL-LARGE: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 4}
// NOMODEL-NOT: Code Model
