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
    if !name.starts_with("mm_") { return None }

    Some(match &name["mm_".len()..] {
        "sqrt_ps" => plain!("llvm.sqrt.v4f32", (f32x4) -> f32x4),
        "sqrt_pd" => plain!("llvm.sqrt.v2f64", (f64x2) -> f64x2),

        "movemask_ps" => p!("sse.movmsk.ps", (f32x4) -> i32),
        "max_ps" => p!("sse.max.ps", (f32x4, f32x4) -> f32x4),
        "min_ps" => p!("sse.min.ps", (f32x4, f32x4) -> f32x4),
        "rsqrt_ps" => p!("sse.rsqrt.ps", (f32x4) -> f32x4),
        "rcp_ps" => p!("sse.rcp.ps", (f32x4) -> f32x4),

        "adds_epi16" => p!("sse2.padds.w", (i16x8, i16x8) -> i16x8),
        "adds_epi8" => p!("sse2.padds.b", (i8x16, i8x16) -> i8x16),
        "adds_epu16" => p!("sse2.paddus.w", (i16x8, i16x8) -> i16x8),
        "adds_epu8" => p!("sse2.paddus.b", (i8x16, i8x16) -> i8x16),
        "avg_epu16" => p!("sse2.pavg.w", (i16x8, i16x8) -> i16x8),
        "avg_epu8" => p!("sse2.pavg.b", (i8x16, i8x16) -> i8x16),
        "madd_epi16" => p!("sse2.pmadd.wd", (i16x8, i16x8) -> i32x4),
        "max_epi16" => p!("sse2.pmaxs.w", (i16x8, i16x8) -> i16x8),
        "max_epu8" => p!("sse2.pmaxu.b", (i8x16, i8x16) -> i8x16),
        "max_pd" => p!("sse2.max.pd", (f64x2, f64x2) -> f64x2),
        "min_epi16" => p!("sse2.pmins.w", (i16x8, i16x8) -> i16x8),
        "min_epu8" => p!("sse2.pminu.b", (i8x16, i8x16) -> i8x16),
        "min_pd" => p!("sse2.min.pd", (f64x2, f64x2) -> f64x2),
        "movemask_pd" => p!("sse2.movmsk.pd", (f64x2) -> i32),
        "movemask_epi8" => p!("sse2.pmovmskb.128", (i8x16) -> i32),
        "mul_epu32" => p!("sse2.pmulu.dq", (i32x4, i32x4) -> i64x2),
        "mulhi_epi16" => p!("sse2.pmulh.w", (i8x16, i8x16) -> i8x16),
        "mulhi_epu16" => p!("sse2.pmulhu.w", (i8x16, i8x16) -> i8x16),
        "packs_epi16" => p!("sse2.packsswb.128", (i16x8, i16x8) -> i8x16),
        "packs_epi32" => p!("sse2.packssdw.128", (i32x4, i32x4) -> i16x8),
        "packus_epi16" => p!("sse2.packuswb.128", (i16x8, i16x8) -> i8x16),
        "sad_epu8" => p!("sse2.psad.bw", (i8x16, i8x16) -> i64x2),
        "subs_epi16" => p!("sse2.psubs.w", (i16x8, i16x8) -> i16x8),
        "subs_epi8" => p!("sse2.psubs.b", (i8x16, i8x16) -> i8x16),
        "subs_epu16" => p!("sse2.psubus.w", (i16x8, i16x8) -> i16x8),
        "subs_epu8" => p!("sse2.psubus.b", (i8x16, i8x16) -> i8x16),

        "addsub_pd" => p!("sse3.addsub.pd", (f64x2, f64x2) -> f64x2),
        "addsub_ps" => p!("sse3.addsub.ps", (f32x4, f32x4) -> f32x4),
        "hadd_pd" => p!("sse3.hadd.pd", (f64x2, f64x2) -> f64x2),
        "hadd_ps" => p!("sse3.hadd.ps", (f32x4, f32x4) -> f32x4),
        "hsub_pd" => p!("sse3.hsub.pd", (f64x2, f64x2) -> f64x2),
        "hsub_ps" => p!("sse3.hsub.ps", (f32x4, f32x4) -> f32x4),

        "abs_epi16" => p!("ssse3.pabs.w.128", (i16x8) -> i16x8),
        "abs_epi32" => p!("ssse3.pabs.d.128", (i32x4) -> i32x4),
        "abs_epi8" => p!("ssse3.pabs.b.128", (i8x16) -> i8x16),
        "hadd_epi16" => p!("ssse3.phadd.w.128", (i16x8, i16x8) -> i16x8),
        "hadd_epi32" => p!("ssse3.phadd.d.128", (i32x4, i32x4) -> i32x4),
        "hadds_epi16" => p!("ssse3.phadd.sw.128", (i16x8, i16x8) -> i16x8),
        "hsub_epi16" => p!("ssse3.phsub.w.128", (i16x8, i16x8) -> i16x8),
        "hsub_epi32" => p!("ssse3.phsub.d.128", (i32x4, i32x4) -> i32x4),
        "hsubs_epi16" => p!("ssse3.phsub.sw.128", (i16x8, i16x8) -> i16x8),
        "maddubs_epi16" => p!("ssse3.pmadd.ub.sw.128", (i8x16, i8x16) -> i16x8),
        "mulhrs_epi16" => p!("ssse3.pmul.hr.sw.128", (i16x8, i16x8) -> i16x8),
        "shuffle_epi8" => p!("ssse3.pshuf.b.128", (i8x16, i8x16) -> i8x16),
        "sign_epi16" => p!("ssse3.psign.w.128", (i16x8, i16x8) -> i16x8),
        "sign_epi32" => p!("ssse3.psign.d.128", (i32x4, i32x4) -> i32x4),
        "sign_epi8" => p!("ssse3.psign.b.128", (i8x16, i8x16) -> i8x16),
        _ => return None
    })
}
