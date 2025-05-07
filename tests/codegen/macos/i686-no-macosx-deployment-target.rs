// Checks that we leave the target alone MACOSX_DEPLOYMENT_TARGET is unset.
// See issue #60235.

//@ add-core-stubs
//@ compile-flags: -Copt-level=3 --target=i686-apple-darwin --crate-type=rlib
//@ needs-llvm-components: x86
//@ unset-rustc-env:MACOSX_DEPLOYMENT_TARGET
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Bool {
    b: bool,
}

// CHECK: target triple = "i686-apple-macosx10.12.0"
#[no_mangle]
pub extern "C" fn structbool() -> Bool {
    Bool { b: true }
}
