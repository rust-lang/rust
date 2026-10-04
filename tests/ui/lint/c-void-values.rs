//@ revisions: direct alias

#![allow(unused)]
#![deny(c_void_values)]
#![allow(c_void_statics)]

#[cfg(direct)]
use std::ffi::c_void;
#[cfg(alias)]
#[expect(non_camel_case_types)]
type c_void = std::ffi::c_void;

const FOO: c_void = unsafe { std::mem::transmute(0u8) };
//~^ ERROR c_void
static BAR: c_void = unsafe { std::mem::transmute(0u8) };
//~^ ERROR c_void

fn foo(r: *mut c_void) {
    unsafe { r.read() };
    //~^ ERROR c_void
}

fn main() {}
