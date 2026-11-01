//@ build-pass
//@ add-minicore
//@ min-llvm-version: 23
//@ ignore-backends: gcc
//
//@ revisions: riscv opt-0-bitcode-no opt-0 opt-s-bitcode-no
//
//@[riscv] compile-flags: --target riscv64gc-unknown-linux-gnu -Clto=thin
//@[riscv] needs-llvm-components: riscv
//
//@[opt-0-bitcode-no] compile-flags: --target armv7r-none-eabihf -Copt-level=0 -Cembed-bitcode=no
//@[opt-0-bitcode-no] needs-llvm-components: arm
//
//@[opt-0] compile-flags: --target armv7r-none-eabihf -Copt-level=0
//@[opt-0] needs-llvm-components: arm
//
//@[opt-s-bitcode-no] compile-flags: --target armv7r-none-eabihf -Copt-level=s -Cembed-bitcode=no
//@[opt-s-bitcode-no] needs-llvm-components: arm

// Regression test for
//
// - https://github.com/llvm/llvm-project/issues/61991
// - https://github.com/rust-lang/rust/issues/80608
// - https://github.com/rust-lang/rust/issues/127269
//
// Since LLVM 23 target features are taken into account for module-level assembly.

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[cfg(target_arch = "riscv64")]
global_asm!("fld f0, 0(sp)");

#[cfg(target_arch = "arm")]
global_asm!(
    r#"
.section .text.startup
.global _start
.code 32
.align 0

_start:
    vmsr fpexc, r0
"#
);
