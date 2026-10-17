//@ check-pass

// Check that the float intrinsics carrying `#[rustc_do_not_const_check]` can actually be called in
// a const context, for every float width. Their fallback bodies call into `libm`, but const-eval
// overrides them.

#![feature(core_intrinsics, f16, f128)]

use std::intrinsics::{ilogb, scalbn};

macro_rules! check {
    ($ty:ident) => {
        const _: () = {
            assert!(scalbn(3.0 as $ty, 2i32) == 12.0);
            assert!(scalbn(1024 as $ty, -10i32) == 1.0);
            assert!(ilogb(1.0 as $ty) == 0);
            assert!(ilogb(4.0 as $ty) == 2);
            assert!(ilogb(0.25 as $ty) == -2);
        };
    };
}

check!(f16);
check!(f32);
check!(f64);
check!(f128);

fn main() {}
