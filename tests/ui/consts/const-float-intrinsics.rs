//@ check-pass

// Check that the float intrinsics carrying `#[rustc_do_not_const_check]` can actually be called in
// a const context, for every float width. Their fallback bodies call into `libm`, but const-eval
// overrides them.

#![feature(core_intrinsics, f16, f128)]

use std::intrinsics::{ceil, floor, fma, round, round_ties_even, trunc};

macro_rules! check {
    ($ty:ident) => {
        const _: () = {
            assert!(floor(-2.5 as $ty) == -3.0);
            assert!(ceil(-2.5 as $ty) == -2.0);
            assert!(trunc(-2.5 as $ty) == -2.0);
            assert!(round_ties_even(2.5 as $ty) == 2.0);
            assert!(round(2.5 as $ty) == 3.0);
            assert!(fma(3.0 as $ty, 4.0 as $ty, 5.0 as $ty) == 17.0);
        };
    };
}

check!(f16);
check!(f32);
check!(f64);
check!(f128);

fn main() {}
