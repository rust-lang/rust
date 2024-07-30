//@ only-x86_64

//@ revisions: nomodel model-small model-kernel model-medium model-large
//@[nomodel] compile-flags:
//@[model-small] compile-flags: -C code-model=small
//@[model-kernel] compile-flags: -C code-model=kernel
//@[model-medium] compile-flags: -C code-model=medium
//@[model-large] compile-flags: -C code-model=large

#![crate_type = "lib"]

// CHECK-MODEL-SMALL: !llvm.module.flags = !{{{.*}}}
// CHECK-MODEL-SMALL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 1}
// CHECK-MODEL-KERNEL: !llvm.module.flags = !{{{.*}}}
// CHECK-MODEL-KERNEL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 2}
// CHECK-MODEL-MEDIUM: !llvm.module.flags = !{{{.*}}}
// CHECK-MODEL-MEDIUM: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 3}
// CHECK-MODEL-LARGE: !llvm.module.flags = !{{{.*}}}
// CHECK-MODEL-LARGE: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 4}
// CHECK-NOMODEL-NOT: Code Model
