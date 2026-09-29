//@ compile-flags: -Znext-solver

// Unsafe binders do not have an unsize impl, regardless of whether the inner type does.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;

fn slice(b: Box<unsafe<'a> MD<[&'a u8; 2]>>) -> Box<unsafe<'a> MD<[&'a u8]>> {
    b //~ ERROR mismatched types
}

fn dyn_(b: Box<unsafe<'a> MD<&'a u8>>) -> Box<unsafe<'a> MD<dyn Debug + 'a>> {
    b //~ ERROR mismatched types
}

fn main() {}
