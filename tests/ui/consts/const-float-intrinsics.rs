//@ check-pass

// Check that the float intrinsics carrying `#[rustc_do_not_const_check]` can actually be called in
// a const context, for every float width. Their bodies are never const-checked and never run by
// const-eval, which has to implement each of these intrinsics itself.

#![feature(core_intrinsics, f16, f128)]

use std::intrinsics::{copysign, fabs, maximum, maximum_number_nsz, minimum, minimum_number_nsz};

macro_rules! check {
    ($ty:ident) => {
        const _: () = {
            assert!(fabs(-2.5 as $ty) == 2.5);
            assert!(copysign(2.5 as $ty, -1.0) == -2.5);

            assert!(minimum(1.0 as $ty, 2.0) == 1.0);
            assert!(maximum(1.0 as $ty, 2.0) == 2.0);
            assert!(minimum_number_nsz($ty::NAN, 2.0) == 2.0);
            assert!(maximum_number_nsz($ty::NAN, 2.0) == 2.0);
        };
    };
}

check!(f16);
check!(f32);
check!(f64);
check!(f128);

fn main() {}
