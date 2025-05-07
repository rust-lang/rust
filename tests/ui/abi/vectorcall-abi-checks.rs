//@ add-core-stubs
//@ compile-flags: --crate-type=rlib --target=i586-unknown-linux-gnu -C target-feature=-sse,-sse2
//@ build-fail
//@ ignore-pass (test emits codegen-time errors)
//@ needs-llvm-components: x86
#![feature(no_core, abi_vectorcall)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub extern "vectorcall" fn f() {
    //~^ ERROR ABI "vectorcall" which requires the `sse2` target feature
}

#[no_mangle]
pub fn call_site() {
    f();
    //~^ ERROR ABI "vectorcall" which requires the `sse2` target feature
}
