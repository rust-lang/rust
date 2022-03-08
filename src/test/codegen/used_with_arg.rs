#![crate_type = "lib"]
#![feature(used_with_arg)]

// CHECK: @llvm.used = appending global [1 x i8*]{{.*}}USED_LINKER
#[used(linker)]
static mut USED_LINKER: [usize; 1] = [0];

// CHECK-NEXT: @llvm.compiler.used = appending global [1 x i8*]{{.*}}USED_COMPILER
#[used(compiler)]
static mut USED_COMPILER: [usize; 1] = [0];
