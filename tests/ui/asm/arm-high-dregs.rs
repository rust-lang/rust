//@ add-minicore
//@ only-arm
//@ only-eabihf
//@ ignore-backends: gcc
//@ revisions: baseline target-cpu
//@ [baseline] check-fail
//@ [target-cpu] check-pass
//@ [target-cpu] compile-flags: -Ctarget-cpu=cortex-a5

// As well as the error message, this also tests that d32 is not enabled by default on arm hardfloat
// targets and that it can be re-enabled by target-cpu.

#![feature(f16, no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn high(x: f64) {
    let y: f64;
    asm!("vmov.f64 d16, d0", in("d0") x, out("d16") y);
    //[baseline]~^ ERROR register class `dreg` requires at least one of the following target features: d32, neon
}
