// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(globs)];

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

        assert!((sqrtf32(64f32).approx_eq(&8f32)));
        assert!((sqrtf64(64f64).approx_eq(&8f64)));

        assert!((powif32(25f32, -2i32).approx_eq(&0.0016f32)));
        assert!((powif64(23.2f64, 2i32).approx_eq(&538.24f64)));

        assert!((sinf32(0f32).approx_eq(&0f32)));
        assert!((sinf64(f64::consts::pi / 2f64).approx_eq(&1f64)));

        assert!((cosf32(0f32).approx_eq(&1f32)));
        assert!((cosf64(f64::consts::pi * 2f64).approx_eq(&1f64)));

        assert!((powf32(25f32, -2f32).approx_eq(&0.0016f32)));
        assert!((powf64(400f64, 0.5f64).approx_eq(&20f64)));

        assert!((fabsf32(expf32(1f32) - f32::consts::e).approx_eq(&0f32)));
        assert!((expf64(1f64).approx_eq(&f64::consts::e)));

        assert!((exp2f32(10f32).approx_eq(&1024f32)));
        assert!((exp2f64(50f64).approx_eq(&1125899906842624f64)));

        assert!((fabsf32(logf32(f32::consts::e) - 1f32).approx_eq(&0f32)));
        assert!((logf64(1f64).approx_eq(&0f64)));

        assert!((log10f32(10f32).approx_eq(&1f32)));
        assert!((log10f64(f64::consts::e).approx_eq(&f64::consts::log10_e)));

        assert!((log2f32(8f32).approx_eq(&3f32)));
        assert!((log2f64(f64::consts::e).approx_eq(&f64::consts::log2_e)));

        assert!((fmaf32(1.0f32, 2.0f32, 5.0f32).approx_eq(&7.0f32)));
        assert!((fmaf64(0.0f64, -2.0f64, f64::consts::e).approx_eq(&f64::consts::e)));

        assert!((fabsf32(-1.0f32).approx_eq(&1.0f32)));
        assert!((fabsf64(34.2f64).approx_eq(&34.2f64)));

        assert!((floorf32(3.8f32).approx_eq(&3.0f32)));
        assert!((floorf64(-1.1f64).approx_eq(&-2.0f64)));

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
