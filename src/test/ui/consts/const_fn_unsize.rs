// check-pass

use std::ptr::NonNull;

const fn test() {
    let _x = NonNull::<[i32; 0]>::dangling() as NonNull<[i32]>;
}

// Regression test for #75118.
pub const fn dangling_slice<T>() -> NonNull<[T]> {
    NonNull::<[T; 0]>::dangling()
}

fn main() {}
