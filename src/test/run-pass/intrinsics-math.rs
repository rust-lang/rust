
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs, macro_rules)]

macro_rules! assert_approx_eq(
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
)

mod rusti {
    extern "rust-intrinsic" {
        pub fn sqrtf32(x: f32) -> f32;
        pub fn sqrtf64(x: f64) -> f64;
        pub fn powif32(a: f32, x: i32) -> f32;
        pub fn powif64(a: f64, x: i32) -> f64;
        pub fn sinf32(x: f32) -> f32;
        pub fn sinf64(x: f64) -> f64;
        pub fn cosf32(x: f32) -> f32;
        pub fn cosf64(x: f64) -> f64;
        pub fn powf32(a: f32, x: f32) -> f32;
        pub fn powf64(a: f64, x: f64) -> f64;
        pub fn expf32(x: f32) -> f32;
        pub fn expf64(x: f64) -> f64;
        pub fn exp2f32(x: f32) -> f32;
        pub fn exp2f64(x: f64) -> f64;
        pub fn logf32(x: f32) -> f32;
        pub fn logf64(x: f64) -> f64;
        pub fn log10f32(x: f32) -> f32;
        pub fn log10f64(x: f64) -> f64;
        pub fn log2f32(x: f32) -> f32;
        pub fn log2f64(x: f64) -> f64;
        pub fn fmaf32(a: f32, b: f32, c: f32) -> f32;
        pub fn fmaf64(a: f64, b: f64, c: f64) -> f64;
        pub fn fabsf32(x: f32) -> f32;
        pub fn fabsf64(x: f64) -> f64;
        pub fn floorf32(x: f32) -> f32;
        pub fn floorf64(x: f64) -> f64;
        pub fn ceilf32(x: f32) -> f32;
        pub fn ceilf64(x: f64) -> f64;
        pub fn truncf32(x: f32) -> f32;
        pub fn truncf64(x: f64) -> f64;
    }
}

pub fn main() {
    unsafe {
        use rusti::*;

        use std::f32;
        use std::f64;

        assert_approx_eq!(sqrtf32(64f32), 8f32);
        assert_approx_eq!(sqrtf64(64f64), 8f64);

        assert_approx_eq!(powif32(25f32, -2i32), 0.0016f32);
        assert_approx_eq!(powif64(23.2f64, 2i32), 538.24f64);

        assert_approx_eq!(sinf32(0f32), 0f32);
        assert_approx_eq!(sinf64(f64::consts::PI / 2f64), 1f64);

        assert_approx_eq!(cosf32(0f32), 1f32);
        assert_approx_eq!(cosf64(f64::consts::PI * 2f64), 1f64);

        assert_approx_eq!(powf32(25f32, -2f32), 0.0016f32);
        assert_approx_eq!(powf64(400f64, 0.5f64), 20f64);

        assert_approx_eq!(fabsf32(expf32(1f32) - f32::consts::E), 0f32);
        assert_approx_eq!(expf64(1f64), f64::consts::E);

        assert_approx_eq!(exp2f32(10f32), 1024f32);
        assert_approx_eq!(exp2f64(50f64), 1125899906842624f64);

        assert_approx_eq!(fabsf32(logf32(f32::consts::E) - 1f32), 0f32);
        assert_approx_eq!(logf64(1f64), 0f64);

        assert_approx_eq!(log10f32(10f32), 1f32);
        assert_approx_eq!(log10f64(f64::consts::E), f64::consts::LOG10_E);

        assert_approx_eq!(log2f32(8f32), 3f32);
        assert_approx_eq!(log2f64(f64::consts::E), f64::consts::LOG2_E);

        assert_approx_eq!(fmaf32(1.0f32, 2.0f32, 5.0f32), 7.0f32);
        assert_approx_eq!(fmaf64(0.0f64, -2.0f64, f64::consts::E), f64::consts::E);

        assert_approx_eq!(fabsf32(-1.0f32), 1.0f32);
        assert_approx_eq!(fabsf64(34.2f64), 34.2f64);

        assert_approx_eq!(floorf32(3.8f32), 3.0f32);
        assert_approx_eq!(floorf64(-1.1f64), -2.0f64);

        // Causes linker error
        // undefined reference to llvm.ceil.f32/64
        //assert!((ceilf32(-2.3f32) == -2.0f32));
        //assert!((ceilf64(3.8f64) == 4.0f64));

        // Causes linker error
        // undefined reference to llvm.trunc.f32/64
        //assert!((truncf32(0.1f32) == 0.0f32));
        //assert!((truncf64(-0.1f64) == 0.0f64));
    }

}
