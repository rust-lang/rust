#![feature(const_generics)] //~ WARN the feature `const_generics` is incomplete

pub const fn func_name<const X: *const u32>() {}
//~^ ERROR using raw pointers

fn main() {}
