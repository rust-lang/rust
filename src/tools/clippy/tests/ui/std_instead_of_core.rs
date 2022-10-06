#![warn(clippy::std_instead_of_core)]
#![allow(unused_imports)]

extern crate alloc;

#[warn(clippy::std_instead_of_core)]
fn std_instead_of_core() {
    // Regular import
    use std::hash::Hasher;
    // Absolute path
    use ::std::hash::Hash;
    // Don't lint on `env` macro
    use std::env;

    // Multiple imports
    use std::fmt::{Debug, Result};

    // Function calls
    let ptr = std::ptr::null::<u32>();
    let ptr_mut = ::std::ptr::null_mut::<usize>();

    // Types
    let cell = std::cell::Cell::new(8u32);
    let cell_absolute = ::std::cell::Cell::new(8u32);

    let _ = std::env!("PATH");

    // do not lint until `error_in_core` is stable
    use std::error::Error;

    // lint items re-exported from private modules, `core::iter::traits::iterator::Iterator`
    use std::iter::Iterator;
}

#[warn(clippy::std_instead_of_alloc)]
fn std_instead_of_alloc() {
    // Only lint once.
    use std::vec;
    use std::vec::Vec;
}

#[warn(clippy::alloc_instead_of_core)]
fn alloc_instead_of_core() {
    use alloc::slice::from_ref;
}

fn main() {
    std_instead_of_core();
    std_instead_of_alloc();
    alloc_instead_of_core();
}
