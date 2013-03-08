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

extern mod std;

use std::cmp::FuzzyEq;

mod rusti {
    #[abi = "rust-intrinsic"]  
    pub extern {
        fn sqrtf32(x: f32) -> f32;
        fn sqrtf64(x: f64) -> f64;
        fn powif32(a: f32, x: i32) -> f32;
        fn powif64(a: f64, x: i32) -> f64;
        fn sinf32(x: f32) -> f32;
        fn sinf64(x: f64) -> f64;
        fn cosf32(x: f32) -> f32;
        fn cosf64(x: f64) -> f64;
        fn powf32(a: f32, x: f32) -> f32;
        fn powf64(a: f64, x: f64) -> f64;
        fn expf32(x: f32) -> f32;
        fn expf64(x: f64) -> f64;
        fn exp2f32(x: f32) -> f32;
        fn exp2f64(x: f64) -> f64;
        fn logf32(x: f32) -> f32;
        fn logf64(x: f64) -> f64;
        fn log10f32(x: f32) -> f32;
        fn log10f64(x: f64) -> f64;
        fn log2f32(x: f32) -> f32;
        fn log2f64(x: f64) -> f64;
        fn fmaf32(a: f32, b: f32, c: f32) -> f32;
        fn fmaf64(a: f64, b: f64, c: f64) -> f64;
        fn fabsf32(x: f32) -> f32;
        fn fabsf64(x: f64) -> f64;
        fn floorf32(x: f32) -> f32;
        fn floorf64(x: f64) -> f64;
        fn ceilf32(x: f32) -> f32;
        fn ceilf64(x: f64) -> f64;
        fn truncf32(x: f32) -> f32;
        fn truncf64(x: f64) -> f64;
    }
}

pub fn main() {
    unsafe {
        use rusti::*;

        fail_unless!((sqrtf32(64f32).fuzzy_eq(&8f32)));
        fail_unless!((sqrtf64(64f64).fuzzy_eq(&8f64)));

        fail_unless!((powif32(25f32, -2i32).fuzzy_eq(&0.0016f32)));
        fail_unless!((powif64(23.2f64, 2i32).fuzzy_eq(&538.24f64)));

        fail_unless!((sinf32(0f32).fuzzy_eq(&0f32)));
        fail_unless!((sinf64(f64::consts::pi / 2f64).fuzzy_eq(&1f64)));

        fail_unless!((cosf32(0f32).fuzzy_eq(&1f32)));
        fail_unless!((cosf64(f64::consts::pi * 2f64).fuzzy_eq(&1f64)));

        fail_unless!((powf32(25f32, -2f32).fuzzy_eq(&0.0016f32)));
        fail_unless!((powf64(400f64, 0.5f64).fuzzy_eq(&20f64)));

        fail_unless!((fabsf32(expf32(1f32) - f32::consts::e).fuzzy_eq(&0f32)));
        fail_unless!((expf64(1f64).fuzzy_eq(&f64::consts::e)));

        fail_unless!((exp2f32(10f32).fuzzy_eq(&1024f32)));
        fail_unless!((exp2f64(50f64).fuzzy_eq(&1125899906842624f64)));

        fail_unless!((fabsf32(logf32(f32::consts::e) - 1f32).fuzzy_eq(&0f32)));
        fail_unless!((logf64(1f64).fuzzy_eq(&0f64)));

        fail_unless!((log10f32(10f32).fuzzy_eq(&1f32)));
        fail_unless!((log10f64(f64::consts::e).fuzzy_eq(&f64::consts::log10_e)));

        fail_unless!((log2f32(8f32).fuzzy_eq(&3f32)));
        fail_unless!((log2f64(f64::consts::e).fuzzy_eq(&f64::consts::log2_e)));
      
        fail_unless!((fmaf32(1.0f32, 2.0f32, 5.0f32).fuzzy_eq(&7.0f32)));
        fail_unless!((fmaf64(0.0f64, -2.0f64, f64::consts::e).fuzzy_eq(&f64::consts::e)));

        fail_unless!((fabsf32(-1.0f32).fuzzy_eq(&1.0f32)));
        fail_unless!((fabsf64(34.2f64).fuzzy_eq(&34.2f64)));

        fail_unless!((floorf32(3.8f32).fuzzy_eq(&3.0f32)));
        fail_unless!((floorf64(-1.1f64).fuzzy_eq(&-2.0f64)));

        // Causes linker error
        // undefined reference to llvm.ceil.f32/64
        //fail_unless!((ceilf32(-2.3f32) == -2.0f32));
        //fail_unless!((ceilf64(3.8f64) == 4.0f64));
      
        // Causes linker error
        // undefined reference to llvm.trunc.f32/64
        //fail_unless!((truncf32(0.1f32) == 0.0f32));
        //fail_unless!((truncf64(-0.1f64) == 0.0f64));
    }

}
