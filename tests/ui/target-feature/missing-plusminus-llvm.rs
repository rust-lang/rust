//@ compile-flags: -Tllvm-target-feature=banana --crate-type=rlib -Zunstable-options
//@ build-pass

//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core, intrinsics, rustc_attrs)]
#![no_core]

extern crate minicore;
use minicore::*;

//~? WARN ignoring feature with missing prefix in `-Zllvm-target-feature`: `banana`
