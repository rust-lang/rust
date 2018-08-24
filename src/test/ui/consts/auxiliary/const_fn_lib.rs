// Crate that exports a const fn. Used for testing cross-crate.

#![crate_type="rlib"]
#![feature(const_fn)]

pub const fn foo() -> usize { 22 } //~ ERROR const fn is unstable
