// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(used_with_arg)]
#![feature(used_on_fn_def)]

// we explicitly don't add #[no_mangle] here as
// that puts the function as in the reachable set

// CHECK: @llvm.used = appending global [1 x i8*]{{.*}}used_linker
#[used(linker)]
fn used_linker() {}

// CHECK: @llvm.compiler.used = appending global [1 x i8*]{{.*}}used_compiler
#[used(compiler)]
fn used_compiler() {}

