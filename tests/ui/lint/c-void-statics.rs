//@ revisions: direct alias

#![allow(unused)]
#![deny(c_void_statics)]
#![allow(c_void_values)]

#[cfg(direct)]
use std::ffi::c_void;
#[cfg(alias)]
#[expect(non_camel_case_types)]
type c_void = std::ffi::c_void;

const FOO: c_void = unsafe { std::mem::transmute(0u8) };
//~^ ERROR c_void
static BAR: c_void = unsafe { std::mem::transmute(0u8) };
//~^ ERROR c_void

unsafe extern "C" {
    safe static BAZ: c_void;
    //~^ ERROR c_void
}

trait Trait {
    const FOO: c_void;
    //~^ ERROR c_void
}

fn main() {}
