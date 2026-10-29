//@ check-pass
//@ only-aarch64
#![allow(nonstandard_style)]
#![crate_type = "lib"]
#![deny(dead_code)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(4)]
pub struct svint32_t(i32);
