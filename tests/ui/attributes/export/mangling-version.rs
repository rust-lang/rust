//@ build-fail
//@ compile-flags: -Zunstable-options -Csymbol-mangling-version=legacy
#![crate_type = "dylib"]
#![feature(export)]

#[export]
pub extern "C" fn foo() -> i32 { 0 }
//~^ ERROR `#[export]` attribute is only usable with `v0` mangling scheme
