//@aux-build:proc_macro_derive.rs

#![warn(clippy::std_instead_of_core)]
#![allow(unused_imports, deprecated)]

extern crate alloc;

#[macro_use]
extern crate proc_macro_derive;

fn std_instead_of_core() {
    // Regular import
    use std::hash::Hasher;
    //~^ ERROR: used import from `std` instead of `core`
    // Absolute path
    use ::std::hash::Hash;
    //~^ ERROR: used import from `std` instead of `core`
    // Don't lint on `env` macro
    use std::env;

    // Multiple imports
    use std::fmt::{Debug, Result};
    //~^ ERROR: used import from `std` instead of `core`

    // Multiple imports multiline
    #[rustfmt::skip]
    use std::{
        //~^ ERROR: used import from `std` instead of `core`
        fmt::Write as _,
        ptr::read_unaligned,
    };

    // Function calls
    let ptr = std::ptr::null::<u32>();
    //~^ ERROR: used import from `std` instead of `core`
    let ptr_mut = ::std::ptr::null_mut::<usize>();
    //~^ ERROR: used import from `std` instead of `core`

    // Types
    let cell = std::cell::Cell::new(8u32);
    //~^ ERROR: used import from `std` instead of `core`
    let cell_absolute = ::std::cell::Cell::new(8u32);
    //~^ ERROR: used import from `std` instead of `core`

    let _ = std::env!("PATH");

    use std::error::Error;
    //~^ ERROR: used import from `std` instead of `core`

    // lint items re-exported from private modules, `core::iter::traits::iterator::Iterator`
    use std::iter::Iterator;
    //~^ ERROR: used import from `std` instead of `core`
}

#[warn(clippy::std_instead_of_alloc)]
fn std_instead_of_alloc() {
    // Only lint once.
    use std::vec;
    //~^ ERROR: used import from `std` instead of `alloc`
    use std::vec::Vec;
    //~^ ERROR: used import from `std` instead of `alloc`
}

#[warn(clippy::alloc_instead_of_core)]
fn alloc_instead_of_core() {
    use alloc::slice::from_ref;
    //~^ ERROR: used import from `alloc` instead of `core`
}

mod std_in_proc_macro_derive {
    #[warn(clippy::alloc_instead_of_core)]
    #[allow(unused)]
    #[derive(ImplStructWithStdDisplay)]
    struct B {}
}

// Some intrinsics are usable on stable but live in an unstable module, but should still suggest
// replacing std -> core
fn intrinsic(a: *mut u8, b: *mut u8) {
    unsafe {
        std::intrinsics::copy(a, b, 1);
        //~^ std_instead_of_core
    }
}

#[clippy::msrv = "1.76"]
fn msrv_1_76(_: std::net::IpAddr) {}

#[clippy::msrv = "1.77"]
fn msrv_1_77(_: std::net::IpAddr) {}
//~^ std_instead_of_core

#[warn(clippy::alloc_instead_of_core)]
fn issue15579() {
    use std::alloc;

    let layout = alloc::Layout::new::<u8>();
}
