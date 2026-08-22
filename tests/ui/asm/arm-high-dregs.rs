//@ add-minicore
//@ check-fail
//@ only-arm
//@ only-eabihf
//@ ignore-backends: gcc

// As well as the error message, this also tests that d32 is not enabled by default on arm hardfloat
// targets

#![feature(f16, no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn high(x: f64) {
    asm!("vmov.f64 d16, d0", in("d16") x);
    //~^ ERROR register class `dreg` requires at least one of the following target features: d32, neon
}
