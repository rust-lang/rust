// This test makes sure that functions get annotated with the proper
// "target-cpu" attribute in LLVM.

//@ no-prefer-dynamic
//@ only-msvc
//@ compile-flags: -C linker-plugin-lto

#![crate_type = "rlib"]

// CHECK-NOT: @{{.*}}__imp_{{.*}}GLOBAL{{.*}} = global i8*

pub static GLOBAL: u32 = 0;
pub static mut GLOBAL2: u32 = 0;
