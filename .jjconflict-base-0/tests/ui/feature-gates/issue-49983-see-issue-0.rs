extern crate core;

// error should not say "(see issue #0)"
#[allow(unused_imports)] use core::ptr::Unique; //~ ERROR use of unstable library feature

fn main() {}
