#![feature(unsized_tuple_coercion)]
#![feature(unsized_fn_params)]

pub extern "C" fn declare_bad(_x: str) {} //~ERROR: cannot be known at compilation time

fn main() {}
