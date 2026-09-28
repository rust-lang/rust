//@ compile-flags: -Znext-solver
//@ aux-build: unsafe-binder-macros.rs

#![feature(unsafe_binders, builtin_syntax)]
#![allow(incomplete_features)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate unsafe_binder_macros;

use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn safe_wrap(x: u8) -> unsafe<> u8 {
    wrap_binder!(x)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}
fn safe_unwrap(b: unsafe<> u8) -> u8 {
    unwrap_binder!(b)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}

unsafe fn unsafe_fn_wrap(x: u8) -> unsafe<> u8 {
    wrap_binder!(x)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}
unsafe fn unsafe_fn_unwrap(b: unsafe<> u8) -> u8 {
    unwrap_binder!(b)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}
unsafe fn unsafe_fn_unwrap_place(p: *const unsafe<> u8) -> *const u8 {
    unsafe { &raw const unwrap_binder!(*p) }
}

macro_rules! local_unwrap {
    ($e:expr) => {
        unwrap_binder!($e)
        //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
    };
}
unsafe fn unsafe_fn_local_macro(b: unsafe<> u8) -> u8 {
    local_unwrap!(b)
}

unsafe fn unsafe_fn_builtin(b: unsafe<> u8) -> u8 {
    builtin # unwrap_binder(b)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}

macro_rules! local_builtin_unwrap {
    ($e:expr) => {
        builtin # unwrap_binder($e)
    };
}
unsafe fn unsafe_fn_local_builtin_macro(b: unsafe<> u8) -> u8 {
    local_builtin_unwrap!(b)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}

unsafe fn unsafe_fn_extern_macro(b: unsafe<> u8) -> u8 {
    unsafe_binder_macros::extern_unwrap!(b)
}
unsafe fn unsafe_fn_extern_builtin_macro(b: unsafe<> u8) -> u8 {
    unsafe_binder_macros::extern_builtin_unwrap!(b)
    //~^ ERROR unsafe binder cast is unsafe and requires unsafe block
}

unsafe fn unsafe_fn_block(b: unsafe<> u8) -> u8 {
    unsafe { unwrap_binder!(b) }
}

fn main() {}
