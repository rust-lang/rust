//@ add-minicore
//@ compile-flags: -C no-prepopulate-passes
//@ revisions: riscv32gc riscv64gc
//@ [riscv32gc] compile-flags: --target riscv32gc-unknown-linux-gnu
//@ [riscv32gc] needs-llvm-components: riscv
//@ [riscv64gc] compile-flags: --target riscv64gc-unknown-linux-gnu
//@ [riscv64gc] needs-llvm-components: riscv

#![feature(link_llvm_intrinsics)]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    #[link_name = "llvm.sqrt.f32"]
    fn sqrt(x: f32) -> f32;
}

pub fn do_call() {
    unsafe {
        // Ensure that we `call` LLVM intrinsics instead of trying to `invoke` them
        // CHECK: call float @llvm.sqrt.f32(float 4.000000e+00)
        sqrt(4.0);
    }
}
