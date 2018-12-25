// Crate that exports a const fn. Used for testing cross-crate.

#![crate_type="rlib"]

pub const fn foo() -> usize { 22 }
