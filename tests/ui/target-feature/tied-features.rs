//@ add-core-stubs
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn main() {
    #[target_feature(enable = "pacg")]
    //~^ ERROR must all be either enabled or disabled together
    unsafe fn inner() {}

    unsafe {
        foo();
        bar();
        baz();
        inner();
    }
}

#[target_feature(enable = "paca")]
//~^ ERROR must all be either enabled or disabled together
unsafe fn foo() {}

#[target_feature(enable = "paca,pacg")]
unsafe fn bar() {}

#[target_feature(enable = "paca")]
#[target_feature(enable = "pacg")]
unsafe fn baz() {}

// Confirm that functions which do not end up collected for monomorphisation will still error.

#[target_feature(enable = "paca")]
//~^ ERROR must all be either enabled or disabled together
unsafe fn unused() {}
