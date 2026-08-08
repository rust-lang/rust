//@ add-minicore
//@ aux-build: simple.rs
//@ compile-flags: -Copt-level=0
//@ check-fail
//@ only-aarch64
#![feature(no_core, rustc_attrs)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
#![allow(internal_features)]

extern crate minicore;
extern crate simple;

pub use simple::Sv;

#[target_feature(enable = "sve")]
pub fn field(x: Sv) -> f32 {
    x.0
    //~^ ERROR: field `0` of struct `Sv` is private
}
