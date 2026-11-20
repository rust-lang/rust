// Check we error before unsupported ABIs reach codegen stages.

//@ edition: 2018
//@ compile-flags: --crate-type=lib
#![feature(rustc_attrs)]

use core::mem;

fn anything() {
    let a = unsafe { mem::transmute::<usize, extern "rust-invalid" fn(i32)>(4) }(2);
    //~^ ERROR: is not a supported ABI for the current target [E0570]
}
