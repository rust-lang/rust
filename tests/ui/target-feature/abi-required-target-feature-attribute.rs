//! Enabling a target feature that is anyway required changes nothing, so this is allowed
//! for `#[target_feature]`.
//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ build-pass
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core, x87_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

#[target_feature(enable = "x87")]
pub unsafe fn my_fun() {}
