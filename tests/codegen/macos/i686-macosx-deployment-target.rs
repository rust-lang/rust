//
// Checks that we correctly modify the target when MACOSX_DEPLOYMENT_TARGET is set.
// See issue #60235.

//@ compile-flags: -Copt-level=3 --target=i686-apple-darwin --crate-type=rlib
//@ needs-llvm-components: x86
//@ rustc-env:MACOSX_DEPLOYMENT_TARGET=10.14
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

#[repr(C)]
pub struct Bool {
    b: bool,
}

// CHECK: target triple = "i686-apple-macosx10.14.0"
#[no_mangle]
pub extern "C" fn structbool() -> Bool {
    Bool { b: true }
}
