// Regression test for #75118.

use std::ptr::NonNull;

pub const fn dangling_slice<T>() -> NonNull<[T]> {
    NonNull::<[T; 0]>::dangling()
    //~^ ERROR: unsizing casts are only allowed for references right now
}

fn main() {}
