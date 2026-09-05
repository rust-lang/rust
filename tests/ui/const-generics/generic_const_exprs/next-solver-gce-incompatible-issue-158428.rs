//@ compile-flags: -Znext-solver=globally

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
//~^ ERROR:`generic_const_args` requires -Znext-solver=globally to be enabled
#![feature(generic_const_exprs)]
//~^ WARN: `feature(generic_const_exprs)` is not supported with the next-generation trait solver
//@ normalize-stderr: "(--> ).*/tests/ui/const-generics/generic_const_exprs" -> "$1$$DIR"

use std::mem::size_of;

union AsBytes<T> {
    as_bytes: [u8; { size_of::<T>() }],
}

fn main() {}
