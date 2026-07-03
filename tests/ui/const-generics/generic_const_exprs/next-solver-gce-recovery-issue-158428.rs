//@ compile-flags: -Znext-solver=globally

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_exprs)]
//~^ ERROR `-Znext-solver=globally` and `generic_const_exprs` are incompatible

use std::mem::size_of;

union AsBytes<T> {
    as_bytes: [u8; const { size_of::<T>() }],
    //~^ ERROR type mismatch resolving
    //~| ERROR the type `[u8; const { size_of::<T>() }]` is not well-formed
}

fn main() {}
