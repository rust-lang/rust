//@ add-minicore
//@ build-fail
//@ compile-flags: -Copt-level=0 --target=aarch64-unknown-linux-gnu
//@ dont-check-compiler-stderr
//@ failure-status: 101
//@ ignore-backends: gcc
//@ needs-llvm-components: aarch64
#![feature(no_core, rustc_attrs)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
#![allow(internal_features)]

extern crate minicore;

#[rustc_scalable_vector(4)]
pub struct Sv(f32);

#[target_feature(enable = "sve")]
pub fn field(x: Sv) -> f32 {
    x.0
    //~^ ERROR broken MIR in Item
    //~| ERROR Projecting into SIMD type Sv is banned by MCP#838
}
