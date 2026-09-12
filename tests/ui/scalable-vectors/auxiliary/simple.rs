//@ compile-flags: -Copt-level=0
//@ only-aarch64
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![crate_type = "rlib"]

#[allow(unused)] // Only used on aarch64-unknown-linux-gnu.
#[rustc_scalable_vector(4)]
pub struct Sv(f32);
