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
    if name.starts_with("mm_") {
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

            "max_epi32" => p!("sse41.pmaxsd", (i32x4, i32x4) -> i32x4),
            "max_epi8" => p!("sse41.pmaxsb", (i8x16, i8x16) -> i8x16),
            "max_epu16" => p!("sse41.pmaxuw", (i16x8, i16x8) -> i16x8),
            "max_epu32" => p!("sse41.pmaxud", (i32x4, i32x4) -> i32x4),
            "min_epi32" => p!("sse41.pminsd", (i32x4, i32x4) -> i32x4),
            "min_epi8" => p!("sse41.pminsb", (i8x16, i8x16) -> i8x16),
            "min_epu16" => p!("sse41.pminuw", (i16x8, i16x8) -> i16x8),
            "min_epu32" => p!("sse41.pminud", (i32x4, i32x4) -> i32x4),
            "minpos_epu16" => p!("sse41.phminposuw", (i16x8) -> i16x8),
            "mul_epi32" => p!("sse41.muldq", (i32x4, i32x4) -> i64x2),
            "packus_epi32" => p!("sse41.packusdw", (i32x4, i32x4) -> i16x8),
            "testc_si128" => p!("sse41.ptestc", (i64x2, i64x2) -> i32),
            "testnzc_si128" => p!("sse41.ptestnzc", (i64x2, i64x2) -> i32),
            "testz_si128" => p!("sse41.ptestz", (i64x2, i64x2) -> i32),

            "permutevar_pd" => p!("avx.vpermilvar.pd", (f64x2, i64x2) -> f64x2),
            "permutevar_ps" => p!("avx.vpermilvar.ps", (f32x4, i32x4) -> f32x4),
            "testc_pd" => p!("avx.vtestc.pd", (f64x2, f64x2) -> i32),
            "testc_ps" => p!("avx.vtestc.ps", (f32x4, f32x4) -> i32),
            "testnzc_pd" => p!("avx.vtestnzc.pd", (f64x2, f64x2) -> i32),
            "testnzc_ps" => p!("avx.vtestnzc.ps", (f32x4, f32x4) -> i32),
            "testz_pd" => p!("avx.vtestz.pd", (f64x2, f64x2) -> i32),
            "testz_ps" => p!("avx.vtestz.ps", (f32x4, f32x4) -> i32),

            _ => return None
        })
    } else if name.starts_with("mm256_") {
        Some(match &name["mm256_".len()..] {
            "addsub_pd" => p!("avx.addsub.pd.256", (f64x4, f64x4) -> f64x4),
            "addsub_ps" => p!("avx.addsub.ps.256", (f32x8, f32x8) -> f32x8),
            "hadd_pd" => p!("avx.hadd.pd.256", (f64x4, f64x4) -> f64x4),
            "hadd_ps" => p!("avx.hadd.ps.256", (f32x8, f32x8) -> f32x8),
            "hsub_pd" => p!("avx.hsub.pd.256", (f64x4, f64x4) -> f64x4),
            "hsub_ps" => p!("avx.hsub.ps.256", (f32x8, f32x8) -> f32x8),
            "max_pd" => p!("avx.max.pd.256", (f64x4, f64x4) -> f64x4),
            "max_ps" => p!("avx.max.ps.256", (f32x8, f32x8) -> f32x8),
            "min_pd" => p!("avx.min.pd.256", (f64x4, f64x4) -> f64x4),
            "min_ps" => p!("avx.min.ps.256", (f32x8, f32x8) -> f32x8),
            "permutevar_pd" => p!("avx.vpermilvar.pd.256", (f64x4, i64x4) -> f64x4),
            "permutevar_ps" => p!("avx.vpermilvar.ps.256", (f32x8, i32x8) -> f32x8),
            "rcp_ps" => p!("avx.rcp.ps.256", (f32x8) -> f32x8),
            "rsqrt_ps" => p!("avx.rsqrt.ps.256", (f32x8) -> f32x8),
            "sqrt_pd" => p!("llvm.sqrt.v4f64", (f64x4) -> f64x4),
            "sqrt_ps" => p!("llvm.sqrt.v8f32", (f32x8) -> f32x8),
            "testc_pd" => p!("avx.vtestc.pd.256", (f64x4, f64x4) -> i32),
            "testc_ps" => p!("avx.vtestc.ps.256", (f32x8, f32x8) -> i32),
            "testnzc_pd" => p!("avx.vtestnzc.pd.256", (f64x4, f64x4) -> i32),
            "testnzc_ps" => p!("avx.vtestnzc.ps.256", (f32x8, f32x8) -> i32),
            "testz_pd" => p!("avx.vtestz.pd.256", (f64x4, f64x4) -> i32),
            "testz_ps" => p!("avx.vtestz.ps.256", (f32x8, f32x8) -> i32),

            "abs_epi16" => p!("avx2.pabs.w", (i16x16) -> i16x16),
            "abs_epi32" => p!("avx2.pabs.d", (i32x8) -> i32x8),
            "abs_epi8" => p!("avx2.pabs.b", (i8x32) -> i8x32),
            "adds_epi16" => p!("avx2.padds.w", (i16x16, i16x16) -> i16x16),
            "adds_epi8" => p!("avx2.padds.b", (i8x32, i8x32) -> i8x32),
            "adds_epu16" => p!("avx2.paddus.w", (i16x16, i16x16) -> i16x16),
            "adds_epu8" => p!("avx2.paddus.b", (i8x32, i8x32) -> i8x32),
            "avg_epu16" => p!("avx2.pavg.w", (i16x16, i16x16) -> i16x16),
            "avg_epu8" => p!("avx2.pavg.b", (i8x32, i8x32) -> i8x32),
            "hadd_epi16" => p!("avx2.phadd.w", (i16x16, i16x16) -> i16x16),
            "hadd_epi32" => p!("avx2.phadd.d", (i32x8, i32x8) -> i32x8),
            "hadds_epi16" => p!("avx2.phadd.sw", (i16x16, i16x16) -> i16x16),
            "hsub_epi16" => p!("avx2.phsub.w", (i16x16, i16x16) -> i16x16),
            "hsub_epi32" => p!("avx2.phsub.d", (i32x8, i32x8) -> i32x8),
            "hsubs_epi16" => p!("avx2.phsub.sw", (i16x16, i16x16) -> i16x16),
            "madd_epi16" => p!("avx2.pmadd.wd", (i16x16, i16x16) -> i32x8),
            "maddubs_epi16" => p!("avx2.pmadd.ub.sw", (i8x32, i8x32) -> i16x16),
            "max_epi16" => p!("avx2.pmaxs.w", (i16x16, i16x16) -> i16x16),
            "max_epi32" => p!("avx2.pmaxs.d", (i32x8, i32x8) -> i32x8),
            "max_epi8" => p!("avx2.pmaxs.b", (i8x32, i8x32) -> i8x32),
            "max_epu16" => p!("avx2.pmaxu.w", (i16x16, i16x16) -> i16x16),
            "max_epu32" => p!("avx2.pmaxu.d", (i32x8, i32x8) -> i32x8),
            "max_epu8" => p!("avx2.pmaxu.b", (i8x32, i8x32) -> i8x32),
            "min_epi16" => p!("avx2.pmins.w", (i16x16, i16x16) -> i16x16),
            "min_epi32" => p!("avx2.pmins.d", (i32x8, i32x8) -> i32x8),
            "min_epi8" => p!("avx2.pmins.b", (i8x32, i8x32) -> i8x32),
            "min_epu16" => p!("avx2.pminu.w", (i16x16, i16x16) -> i16x16),
            "min_epu32" => p!("avx2.pminu.d", (i32x8, i32x8) -> i32x8),
            "min_epu8" => p!("avx2.pminu.b", (i8x32, i8x32) -> i8x32),
            "mul_epi32" => p!("avx2.mul.dq", (i32x8, i32x8) -> i64x4),
            "mul_epu32" => p!("avx2.mulu.dq", (i32x8, i32x8) -> i64x4),
            "mulhi_epi16" => p!("avx2.pmulh.w", (i8x32, i8x32) -> i8x32),
            "mulhi_epu16" => p!("avx2.pmulhu.w", (i8x32, i8x32) -> i8x32),
            "mulhrs_epi16" => p!("avx2.pmul.hr.sw", (i16x16, i16x16) -> i16x16),
            "packs_epi16" => p!("avx2.packsswb", (i16x16, i16x16) -> i8x32),
            "packs_epi32" => p!("avx2.packssdw", (i32x8, i32x8) -> i16x16),
            "packus_epi16" => p!("avx2.packuswb", (i16x16, i16x16) -> i8x32),
            "packus_epi32" => p!("avx2.packusdw", (i32x8, i32x8) -> i16x16),
            "permutevar8x32_epi32" => p!("avx2.permd", (i32x8, i32x8) -> i32x8),
            "permutevar8x32_ps" => p!("avx2.permps", (f32x8, i32x8) -> i32x8),
            "sad_epu8" => p!("avx2.psad.bw", (i8x32, i8x32) -> i64x4),
            "shuffle_epi8" => p!("avx2.pshuf.b", (i8x32, i8x32) -> i8x32),
            "sign_epi16" => p!("avx2.psign.w", (i16x16, i16x16) -> i16x16),
            "sign_epi32" => p!("avx2.psign.d", (i32x8, i32x8) -> i32x8),
            "sign_epi8" => p!("avx2.psign.b", (i8x32, i8x32) -> i8x32),
            "subs_epi16" => p!("avx2.psubs.w", (i16x16, i16x16) -> i16x16),
            "subs_epi8" => p!("avx2.psubs.b", (i8x32, i8x32) -> i8x32),
            "subs_epu16" => p!("avx2.psubus.w", (i16x16, i16x16) -> i16x16),
            "subs_epu8" => p!("avx2.psubus.b", (i8x32, i8x32) -> i8x32),

            _ => return None,
        })
    } else {
        None
    }
}
