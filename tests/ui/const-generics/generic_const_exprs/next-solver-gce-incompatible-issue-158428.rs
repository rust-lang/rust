//@ compile-flags: -Znext-solver=globally

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_exprs)]
//~^ ERROR `-Znext-solver=globally` and `generic_const_exprs` are incompatible
//@ normalize-stderr: "(--> ).*/tests/ui/const-generics/generic_const_exprs" -> "$1$$DIR"

use std::mem::size_of;

union AsBytes<T> {
    as_bytes: [u8; const { size_of::<T>() }],
}

fn main() {}
