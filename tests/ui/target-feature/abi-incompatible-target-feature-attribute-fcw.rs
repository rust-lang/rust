//@ compile-flags: --crate-type=lib
//@ revisions: aarch64 x86_64
//@[aarch64] compile-flags: --target=aarch64-unknown-none-softfloat
//@[aarch64] needs-llvm-components: aarch64
//@[x86_64] compile-flags: --target=x86_64-unknown-none
//@[x86_64] needs-llvm-components: x86
//@ add-minicore
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]
#![deny(aarch64_softfloat_neon, x86_softfloat_sse)]

extern crate minicore;
use minicore::*;

#[cfg_attr(aarch64, target_feature(enable = "neon"))]
//[aarch64]~^ERROR: enabling the `neon` target feature on the current target is unsound
//[aarch64]~|WARN: previously accepted
#[cfg_attr(x86_64, target_feature(enable = "avx"))]
//[x86_64]~^ERROR: enabling the `sse` target feature on the current target is unsupported
//[x86_64]~|WARN: previously accepted
pub unsafe fn my_fun() {}
