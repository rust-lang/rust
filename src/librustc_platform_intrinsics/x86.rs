// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Intrinsic, i, f, v};
use rustc::middle::ty;

macro_rules! p {
    ($name: expr, ($($inputs: tt),*) -> $output: tt) => {
        plain!(concat!("llvm.x86.", $name), ($($inputs),*) -> $output)
    }
}

pub fn find<'tcx>(_tcx: &ty::ctxt<'tcx>, name: &str) -> Option<Intrinsic> {
    Some(match name {
        "mm_movemask_ps" => p!("sse.movmsk.ps", (f32x4) -> i32),
        "mm_movemask_pd" => p!("sse2.movmsk.pd", (f64x2) -> i32),
        "mm_movemask_epi8" => p!("sse2.pmovmskb.128", (i8x16) -> i32),

        "mm_rsqrt_ps" => p!("sse.rsqrt.ps", (f32x4) -> f32x4),

        "mm_sqrt_ps" => plain!("llvm.sqrt.v4f32", (f32x4) -> f32x4),
        "mm_sqrt_pd" => plain!("llvm.sqrt.v2f64", (f64x2) -> f64x2),

        "mm_max_ps" => p!("sse.max.ps", (f32x4, f32x4) -> f32x4),
        "mm_max_pd" => p!("sse2.max.pd", (f64x2, f64x2) -> f64x2),

        "mm_min_ps" => p!("sse.min.ps", (f32x4, f32x4) -> f32x4),
        "mm_min_pd" => p!("sse2.min.pd", (f64x2, f64x2) -> f64x2),
        _ => return None
    })
}
