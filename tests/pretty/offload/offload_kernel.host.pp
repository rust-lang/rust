#![feature(prelude_import)]
#![no_std]
//@ only-nightly
//@ revisions: host device

//@ pretty-mode:expanded
//@ pretty-compare-only
//@[host] pp-exact:offload_kernel.host.pp
//@[device] pp-exact:offload_kernel.device.pp

//@[device] compile-flags: -Zunstable-options -Zoffload=Device

#![feature(gpu_offload)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

use std::offload::offload_kernel;

#[unsafe(no_mangle)]
#[inline(never)]
fn foo(_: &[f32], _: &[f32], _: *mut f32) {

    ::core::panicking::panic("not implemented")
}
fn main() {}
