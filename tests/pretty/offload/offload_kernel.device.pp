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

#[rustc_offload_kernel]
#[unsafe(no_mangle)]
unsafe extern "gpu-kernel" fn foo(a: &[f32], b: &[f32], c: *mut f32) {
    *c = a[0] + b[0];
}

fn main() {}
