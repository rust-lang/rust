#![feature(prelude_import)]
#![no_std]
//@ only-nightly
//@ revisions: host device

//@ pretty-mode:expanded
//@ pretty-compare-only
//@[host] pp-exact:offload_kernel_nested.host.pp
//@[device] pp-exact:offload_kernel_nested.device.pp

//@[device] compile-flags: -Zunstable-options -Zoffload=Device

#![feature(gpu_offload)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

use std::offload::offload_kernel;

fn kernel() {
    #[rustc_offload_kernel]
    #[inline(never)]
    fn inner_kernel() {

        ::core::panicking::panic("not implemented")
    }
}
fn main() {}
