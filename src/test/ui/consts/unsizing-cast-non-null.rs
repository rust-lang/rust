// Regression test for #75118.

use std::ptr::NonNull;

pub const fn dangling_slice<T>() -> NonNull<[T]> {
    NonNull::<[T; 0]>::dangling()
    //~^ ERROR: unsizing casts to types besides slices
}

fn main() {}
