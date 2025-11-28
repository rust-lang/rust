//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=aarch64-unknown-none-softfloat
//@ needs-llvm-components: aarch64
//@ add-minicore
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]
#![deny(aarch64_softfloat_neon)]

extern crate minicore;
use minicore::*;

#[target_feature(enable = "neon")]
//~^ERROR: enabling the `neon` target feature on the current target is unsound
//~|WARN: previously accepted
pub unsafe fn my_fun() {}
