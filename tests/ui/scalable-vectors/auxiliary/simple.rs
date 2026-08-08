//@ add-minicore
//@ compile-flags: -Copt-level=0
//@ only-aarch64
#![feature(no_core, rustc_attrs)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
#![allow(internal_features)]

extern crate minicore;

#[rustc_scalable_vector(4)]
pub struct Sv(f32);
