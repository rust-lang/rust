#![warn(clippy::std_instead_of_core)]
#![allow(unused_imports)]

extern crate alloc;

#[warn(clippy::std_instead_of_core)]
fn std_instead_of_core() {
    // Regular import
    use std::hash::Hasher;
    // Absolute path
    use ::std::hash::Hash;

    // Multiple imports
    use std::fmt::{Debug, Result};

    // Function calls
    let ptr = std::ptr::null::<u32>();
    let ptr_mut = ::std::ptr::null_mut::<usize>();

    // Types
    let cell = std::cell::Cell::new(8u32);
    let cell_absolute = ::std::cell::Cell::new(8u32);
}

#[warn(clippy::std_instead_of_alloc)]
fn std_instead_of_alloc() {
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
