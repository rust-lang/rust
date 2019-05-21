//
// Checks that we leave the target alone MACOSX_DEPLOYMENT_TARGET is unset.
// See issue #60235.

// compile-flags: -O --target=i686-apple-darwin --crate-type=rlib
// unset-rustc-env:MACOSX_DEPLOYMENT_TARGET
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="freeze"]
trait Freeze { }
#[lang="copy"]
trait Copy { }

#[repr(C)]
pub struct Bool {
    b: bool,
}

// CHECK: target triple = "i686-apple-macosx10.7.0"
#[no_mangle]
pub extern "C" fn structbool() -> Bool {
    Bool { b: true }
}
