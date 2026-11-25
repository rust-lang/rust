//! Ensure ABI-incompatible features cannot be enabled via `-Ctarget-feature`.
// These are just warnings for now.
//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: x86 riscv
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=+avx512f
//@[x86] needs-llvm-components: x86
//@[riscv] compile-flags: --target=riscv64im-unknown-none-elf -Ctarget-feature=+d
//@[riscv] needs-llvm-components: riscv
//@ add-minicore

#![feature(no_core, riscv_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

// Ensure we can "see" the enabled target feature, and its implications.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx2"),
    all(target_arch = "riscv64", target_feature = "d", target_feature = "f")
)))]
compile_error!("target feature has not been set in cfg");
