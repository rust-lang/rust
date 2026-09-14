//! Ensure that we refuse to run a do_not_const_check function, even if the body *would* const-check
//! at the moment.
#![feature(rustc_attrs, intrinsics)]

#[rustc_do_not_const_check]
const fn mostly_harmless() {}

const _: () = {
    mostly_harmless(); //~ERROR: calling non-const function
};

// Also ensure the same happens with intrinsics.
// Here we need some intrinsic that the interpreter does *not* have a native implementation for.
// Let's hope nobody adds one...
#[rustc_intrinsic]
#[rustc_do_not_const_check]
pub const fn integer_min<T: Copy>(a: T, b: T) -> T {
    a
}

const _: () = {
    integer_min(0, 1); //~ERROR: calling non-const function
};

fn main() {}
