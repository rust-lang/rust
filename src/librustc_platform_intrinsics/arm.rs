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
        plain!(concat!("llvm.arm.neon.", $name), ($($inputs),*) -> $output)
    }
}
pub fn find<'tcx>(_tcx: &ty::ctxt<'tcx>, name: &str) -> Option<Intrinsic> {
    if !name.starts_with("v") { return None }
    Some(match &name["v".len()..] {
        "maxq_f32" => p!("vmaxs.v4f32", (f32x4, f32x4) -> f32x4),
        "minq_f32" => p!("vmins.v4f32", (f32x4, f32x4) -> f32x4),
        "pmax_s16" => p!("vpmaxs.v4i16", (i16x4, i16x4) -> i16x4),
        "pmax_s32" => p!("vpmaxs.v2i32", (i32x2, i32x2) -> i32x2),
        "pmax_s8" => p!("vpmaxs.v8i8", (i8x8, i8x8) -> i8x8),
        "pmax_u16" => p!("vpmaxu.v4i16", (i16x4, i16x4) -> i16x4),
        "pmax_u32" => p!("vpmaxu.v2i32", (i32x2, i32x2) -> i32x2),
        "pmax_u8" => p!("vpmaxu.v8i8", (i8x8, i8x8) -> i8x8),
        "pmin_s16" => p!("vpmins.v4i16", (i16x4, i16x4) -> i16x4),
        "pmin_s32" => p!("vpmins.v2i32", (i32x2, i32x2) -> i32x2),
        "pmin_s8" => p!("vpmins.v8i8", (i8x8, i8x8) -> i8x8),
        "pmin_u16" => p!("vpminu.v4i16", (i16x4, i16x4) -> i16x4),
        "pmin_u32" => p!("vpminu.v2i32", (i32x2, i32x2) -> i32x2),
        "pmin_u8" => p!("vpminu.v8i8", (i8x8, i8x8) -> i8x8),
        "recpeq_f32" => p!("vrecpe.v4f32", (f32x4) -> f32x4),
        "rsqrteq_f32" => p!("vrsqrte.v4f32", (f32x4) -> f32x4),
        "rsqrteq_f64" => p!("vrsqrte.v2f64", (f64x2) -> f64x2),
        "sqrtq_f32" => plain!("llvm.sqrt.v4f32", (f32x4) -> f32x4),
        "sqrtq_f64" => plain!("llvm.sqrt.v2f64", (f64x2) -> f64x2),
        _ => return None,
    })
}
