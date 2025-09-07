//@ compile-flags:-C panic=unwind
//@ no-prefer-dynamic
//@ add-core-stubs

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;

extern "C-unwind" fn foo() {}

#[inline]
fn bar() {
    let ptr: extern "C-unwind" fn() = foo;
    ptr();
}
