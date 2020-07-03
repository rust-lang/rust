// revisions: NOMODEL MODEL-TINY MODEL-SMALL MODEL-KERNEL MODEL-MEDIUM MODEL-LARGE
//[NOMODEL] compile-flags:
//[MODEL-TINY] compile-flags: --target=riscv32i-unknown-none-elf -C code-model=tiny
//[MODEL-SMALL] compile-flags: -C code-model=small
//[MODEL-KERNEL] compile-flags: --target=x86_64-unknown-linux-gnu -C code-model=kernel
//[MODEL-MEDIUM] compile-flags: --target=x86_64-unknown-linux-gnu -C code-model=medium
//[MODEL-LARGE] compile-flags: -C code-model=large

#![crate_type = "lib"]

// MODEL-TINY: !llvm.module.flags = !{{{.*}}}
// MODEL-TINY: !(([0-9]+)) = !(i32 1, !"Code-Model", i32 0)
// MODEL-SMALL: !llvm.module.flags = !{{{.*}}}
// MODEL-SMALL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 1}
// MODEL-KERNEL: !llvm.module.flags = !{{{.*}}}
// MODEL-KERNEL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 2}
// MODEL-MEDIUM: !llvm.module.flags = !{{{.*}}}
// MODEL-MEDIUM: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 3}
// MODEL-LARGE: !llvm.module.flags = !{{{.*}}}
// MODEL-LARGE: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 4}
// NOMODEL-NOT: Code Model
