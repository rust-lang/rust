// compile-flags: -O

#![crate_type = "lib"]
#![feature(used_with_arg)]

// CHECK: @llvm.used = appending global [1 x i8*]
#[used(linker)]
static mut USED_LINKER: [usize; 1] = [0];

// CHECK-NEXT: @llvm.compiler.used = appending global [1 x i8*]
#[used(compiler)]
static mut USED_COMPILER: [usize; 1] = [0];
