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
        plain!(concat!("llvm.aarch64.neon.", $name), ($($inputs),*) -> $output)
    }
}
pub fn find<'tcx>(_tcx: &ty::ctxt<'tcx>, name: &str) -> Option<Intrinsic> {
    Some(match name["aarch64_".len()..] {
        "vmaxvq_u8" => p!("umaxv.i8.v16i8", (i8x16) -> i8),
        "vmaxvq_u16" => p!("umaxv.i16.v8i16", (i16x8) -> i16),
        "vmaxvq_u32" => p!("umaxv.i32.v4i32", (i32x4) -> i32),

        "vmaxvq_s8" => p!("smaxv.i8.v16i8", (i8x16) -> i8),
        "vmaxvq_s16" => p!("smaxv.i16.v8i16", (i16x8) -> i16),
        "vmaxvq_s32" => p!("smaxv.i32.v4i32", (i32x4) -> i32),

        "vminvq_u8" => p!("uminv.i8.v16i8", (i8x16) -> i8),
        "vminvq_u16" => p!("uminv.i16.v8i16", (i16x8) -> i16),
        "vminvq_u32" => p!("uminv.i32.v4i32", (i32x4) -> i32),
        "vminvq_s8" => p!("sminv.i8.v16i8", (i8x16) -> i8),
        "vminvq_s16" => p!("sminv.i16.v8i16", (i16x8) -> i16),
        "vminvq_s32" => p!("sminv.i32.v4i32", (i32x4) -> i32),

        "vsqrtq_f32" => plain!("llvm.sqrt.v4f32", (f32x4) -> f32x4),
        "vsqrtq_f64" => plain!("llvm.sqrt.v2f64", (f64x2) -> f64x2),

        "vrsqrteq_f32" => p!("frsqrte.v4f32", (f32x4) -> f32x4),
        "vrsqrteq_f64" => p!("frsqrte.v2f64", (f64x2) -> f64x2),
        "vrecpeq_f32" => p!("frecpe.v4f32", (f32x4) -> f32x4),
        "vrecpeq_f64" => p!("frecpe.v2f64", (f64x2) -> f64x2),

        "vmaxq_f32" => p!("fmax.v4f32", (f32x4, f32x4) -> f32x4),
        "vmaxq_f64" => p!("fmax.v2f64", (f64x2, f64x2) -> f64x2),

        "vminq_f32" => p!("fmin.v4f32", (f32x4, f32x4) -> f32x4),
        "vminq_f64" => p!("fmin.v2f64", (f64x2, f64x2) -> f64x2),

        "vqtbl1q_u8" => p!("tbl1.v16i8", (i8x16, i8x16) -> i8x16),
        "vqtbl1q_s8" => p!("tbl1.v16i8", (i8x16, i8x16) -> i8x16),
        _ => return None,
    })
}
