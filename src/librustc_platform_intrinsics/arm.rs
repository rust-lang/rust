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
    Some(match name {
        "vpmax_u8" => p!("vpmaxu.v8i8", (i8x8, i8x8) -> i8x8),
        "vpmax_s8" => p!("vpmaxs.v8i8", (i8x8, i8x8) -> i8x8),
        "vpmax_u16" => p!("vpmaxu.v4i16", (i16x4, i16x4) -> i16x4),
        "vpmax_s16" => p!("vpmaxs.v4i16", (i16x4, i16x4) -> i16x4),
        "vpmax_u32" => p!("vpmaxu.v2i32", (i32x2, i32x2) -> i32x2),
        "vpmax_s32" => p!("vpmaxs.v2i32", (i32x2, i32x2) -> i32x2),

        "vpmin_u8" => p!("vpminu.v8i8", (i8x8, i8x8) -> i8x8),
        "vpmin_s8" => p!("vpmins.v8i8", (i8x8, i8x8) -> i8x8),
        "vpmin_u16" => p!("vpminu.v4i16", (i16x4, i16x4) -> i16x4),
        "vpmin_s16" => p!("vpmins.v4i16", (i16x4, i16x4) -> i16x4),
        "vpmin_u32" => p!("vpminu.v2i32", (i32x2, i32x2) -> i32x2),
        "vpmin_s32" => p!("vpmins.v2i32", (i32x2, i32x2) -> i32x2),

        "vsqrtq_f32" => plain!("llvm.sqrt.v4f32", (f32x4) -> f32x4),
        "vsqrtq_f64" => plain!("llvm.sqrt.v2f64", (f64x2) -> f64x2),

        "vrecpeq_f32" => p!("vrecpe.v4f32", (f32x4) -> f32x4),
        "vrsqrteq_f32" => p!("vrsqrte.v4f32", (f32x4) -> f32x4),
        "vrsqrteq_f64" => p!("vrsqrte.v2f64", (f64x2) -> f64x2),

        "vmaxq_f32" => p!("vmaxs.v4f32", (f32x4, f32x4) -> f32x4),

        "vminq_f32" => p!("vmins.v4f32", (f32x4, f32x4) -> f32x4),
        _ => return None,
    })
}
