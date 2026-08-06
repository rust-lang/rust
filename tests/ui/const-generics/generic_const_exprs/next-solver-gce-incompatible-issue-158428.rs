//@ compile-flags: -Znext-solver=globally

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_exprs)]
//~^ WARN: `-Znext-solver=globally` is disabled because `generic_const_exprs` is enabled
//@ normalize-stderr: "(--> ).*/tests/ui/const-generics/generic_const_exprs" -> "$1$$DIR"

use std::mem::size_of;

union AsBytes<T> {
    as_bytes: [u8; const { size_of::<T>() }],
    //~^ ERROR: overly complex generic constant
}

fn main() {}
