//! Benchmarks that use `gungraun` to be reasonably CI-stable.
#![feature(f16)]
#![feature(f128)]

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use libm::support::{HInt, Hex, hf16, hf32, hf64, hf128, i256, u256};
use libm_test::generate::spaced;
use libm_test::{CheckBasis, CheckCtx, GeneratorKind, MathOp, OpRustArgs, TupleCall, op};

const BENCH_ITER_ITEMS: u64 = 500;

macro_rules! icount_benches {
    (
        fn_name: $fn_name:ident,
        attrs: [$($_attr:meta),*],
    ) => {
        paste::paste! {
            // Construct benchmark inputs from the logspace generator.
            fn [< setup_ $fn_name >]() -> Vec<OpRustArgs<op::$fn_name::Routine>> {
                type Op = op::$fn_name::Routine;
                let mut ctx = CheckCtx::new(
                    Op::IDENTIFIER,
                    CheckBasis::None,
                    GeneratorKind::Spaced
                );
                ctx.override_iterations(BENCH_ITER_ITEMS);
                let ret = spaced::get_test_cases::<Op>(&ctx).0.collect::<Vec<_>>();
                println!("operation {}, {} steps", Op::NAME, ret.len());
                ret
            }

            // Run benchmarks with the above inputs.
            #[library_benchmark]
            #[bench::logspace([< setup_ $fn_name >]())]
            fn [< icount_bench_ $fn_name >](cases: Vec<OpRustArgs<op::$fn_name::Routine>>) {
                type Op = op::$fn_name::Routine;
                let f = black_box(Op::ROUTINE);
                for input in cases.iter().copied() {
                    input.call(f);
                }
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: icount_benches,
}

library_benchmark_group!(
    name = icount_bench_math_group,
    benchmarks = [
        // verify-apilist-start
        // verify-sorted-start
        icount_bench_acos,
        icount_bench_acosf,
        icount_bench_acosh,
        icount_bench_acoshf,
        icount_bench_addf128,
        icount_bench_addf16,
        icount_bench_addf32,
        icount_bench_addf64,
        icount_bench_ashl_u128,
        icount_bench_ashl_u32,
        icount_bench_ashl_u64,
        icount_bench_ashr_i128,
        icount_bench_ashr_i32,
        icount_bench_ashr_i64,
        icount_bench_asin,
        icount_bench_asinf,
        icount_bench_asinh,
        icount_bench_asinhf,
        icount_bench_atan,
        icount_bench_atan2,
        icount_bench_atan2f,
        icount_bench_atanf,
        icount_bench_atanh,
        icount_bench_atanhf,
        icount_bench_cbrt,
        icount_bench_cbrtf,
        icount_bench_ceil,
        icount_bench_ceilf,
        icount_bench_ceilf128,
        icount_bench_ceilf16,
        icount_bench_copysign,
        icount_bench_copysignf,
        icount_bench_copysignf128,
        icount_bench_copysignf16,
        icount_bench_cos,
        icount_bench_cosf,
        icount_bench_cosh,
        icount_bench_coshf,
        icount_bench_divf128,
        icount_bench_divf32,
        icount_bench_divf64,
        icount_bench_eqf128,
        icount_bench_eqf16,
        icount_bench_eqf32,
        icount_bench_eqf64,
        icount_bench_erf,
        icount_bench_erfc,
        icount_bench_erfcf,
        icount_bench_erff,
        icount_bench_exp,
        icount_bench_exp10,
        icount_bench_exp10f,
        icount_bench_exp2,
        icount_bench_exp2f,
        icount_bench_expf,
        icount_bench_expm1,
        icount_bench_expm1f,
        icount_bench_extend_f16_f128,
        icount_bench_extend_f16_f32,
        icount_bench_extend_f16_f64,
        icount_bench_extend_f32_f128,
        icount_bench_extend_f32_f64,
        icount_bench_extend_f64_f128,
        icount_bench_fabs,
        icount_bench_fabsf,
        icount_bench_fabsf128,
        icount_bench_fabsf16,
        icount_bench_fdim,
        icount_bench_fdimf,
        icount_bench_fdimf128,
        icount_bench_fdimf16,
        icount_bench_floor,
        icount_bench_floorf,
        icount_bench_floorf128,
        icount_bench_floorf16,
        icount_bench_fma,
        icount_bench_fmaf,
        icount_bench_fmaf128,
        icount_bench_fmax,
        icount_bench_fmaxf,
        icount_bench_fmaxf128,
        icount_bench_fmaxf16,
        icount_bench_fmaximum,
        icount_bench_fmaximum_num,
        icount_bench_fmaximum_numf,
        icount_bench_fmaximum_numf128,
        icount_bench_fmaximum_numf16,
        icount_bench_fmaximumf,
        icount_bench_fmaximumf128,
        icount_bench_fmaximumf16,
        icount_bench_fmin,
        icount_bench_fminf,
        icount_bench_fminf128,
        icount_bench_fminf16,
        icount_bench_fminimum,
        icount_bench_fminimum_num,
        icount_bench_fminimum_numf,
        icount_bench_fminimum_numf128,
        icount_bench_fminimum_numf16,
        icount_bench_fminimumf,
        icount_bench_fminimumf128,
        icount_bench_fminimumf16,
        icount_bench_fmod,
        icount_bench_fmodf,
        icount_bench_fmodf128,
        icount_bench_fmodf16,
        icount_bench_frexp,
        icount_bench_frexpf,
        icount_bench_frexpf128,
        icount_bench_frexpf16,
        icount_bench_ftoi_f128_i128,
        icount_bench_ftoi_f128_i32,
        icount_bench_ftoi_f128_i64,
        icount_bench_ftoi_f128_u128,
        icount_bench_ftoi_f128_u32,
        icount_bench_ftoi_f128_u64,
        icount_bench_ftoi_f32_i128,
        icount_bench_ftoi_f32_i32,
        icount_bench_ftoi_f32_i64,
        icount_bench_ftoi_f32_u128,
        icount_bench_ftoi_f32_u32,
        icount_bench_ftoi_f32_u64,
        icount_bench_ftoi_f64_i128,
        icount_bench_ftoi_f64_i32,
        icount_bench_ftoi_f64_i64,
        icount_bench_ftoi_f64_u128,
        icount_bench_ftoi_f64_u32,
        icount_bench_ftoi_f64_u64,
        icount_bench_gef128,
        icount_bench_gef16,
        icount_bench_gef32,
        icount_bench_gef64,
        icount_bench_gtf128,
        icount_bench_gtf16,
        icount_bench_gtf32,
        icount_bench_gtf64,
        icount_bench_hypot,
        icount_bench_hypotf,
        icount_bench_iadd_i128,
        icount_bench_iadd_u128,
        icount_bench_iaddo_i128,
        icount_bench_iaddo_u128,
        icount_bench_idiv_i128,
        icount_bench_idiv_i32,
        icount_bench_idiv_i64,
        icount_bench_idiv_u128,
        icount_bench_idiv_u32,
        icount_bench_idiv_u64,
        icount_bench_idivmod_i128,
        icount_bench_idivmod_i32,
        icount_bench_idivmod_i64,
        icount_bench_idivmod_u128,
        icount_bench_idivmod_u32,
        icount_bench_idivmod_u64,
        icount_bench_ilogb,
        icount_bench_ilogbf,
        icount_bench_ilogbf128,
        icount_bench_ilogbf16,
        icount_bench_imod_i128,
        icount_bench_imod_i32,
        icount_bench_imod_i64,
        icount_bench_imod_u128,
        icount_bench_imod_u32,
        icount_bench_imod_u64,
        icount_bench_imul_i128,
        icount_bench_imul_u64,
        icount_bench_imulo_i128,
        icount_bench_imulo_i32,
        icount_bench_imulo_i64,
        icount_bench_imulo_u128,
        icount_bench_isub_i128,
        icount_bench_isub_u128,
        icount_bench_isubo_i128,
        icount_bench_isubo_u128,
        icount_bench_itof_i128_f128,
        icount_bench_itof_i128_f32,
        icount_bench_itof_i128_f64,
        icount_bench_itof_i32_f128,
        icount_bench_itof_i32_f32,
        icount_bench_itof_i32_f64,
        icount_bench_itof_i64_f128,
        icount_bench_itof_i64_f32,
        icount_bench_itof_i64_f64,
        icount_bench_itof_u128_f128,
        icount_bench_itof_u128_f32,
        icount_bench_itof_u128_f64,
        icount_bench_itof_u32_f128,
        icount_bench_itof_u32_f32,
        icount_bench_itof_u32_f64,
        icount_bench_itof_u64_f128,
        icount_bench_itof_u64_f32,
        icount_bench_itof_u64_f64,
        icount_bench_j0,
        icount_bench_j0f,
        icount_bench_j1,
        icount_bench_j1f,
        icount_bench_jn,
        icount_bench_jnf,
        icount_bench_ldexp,
        icount_bench_ldexpf,
        icount_bench_ldexpf128,
        icount_bench_ldexpf16,
        icount_bench_leading_zeros_u128,
        icount_bench_leading_zeros_u32,
        icount_bench_leading_zeros_u64,
        icount_bench_lef128,
        icount_bench_lef16,
        icount_bench_lef32,
        icount_bench_lef64,
        icount_bench_lgamma,
        icount_bench_lgamma_r,
        icount_bench_lgammaf,
        icount_bench_lgammaf_r,
        icount_bench_log,
        icount_bench_log10,
        icount_bench_log10f,
        icount_bench_log1p,
        icount_bench_log1pf,
        icount_bench_log2,
        icount_bench_log2f,
        icount_bench_logf,
        icount_bench_lshr_u128,
        icount_bench_lshr_u32,
        icount_bench_lshr_u64,
        icount_bench_ltf128,
        icount_bench_ltf16,
        icount_bench_ltf32,
        icount_bench_ltf64,
        icount_bench_modf,
        icount_bench_modff,
        icount_bench_mulf128,
        icount_bench_mulf16,
        icount_bench_mulf32,
        icount_bench_mulf64,
        icount_bench_narrow_f128_f16,
        icount_bench_narrow_f128_f32,
        icount_bench_narrow_f128_f64,
        icount_bench_narrow_f32_f16,
        icount_bench_narrow_f64_f16,
        icount_bench_narrow_f64_f32,
        icount_bench_nef128,
        icount_bench_nef16,
        icount_bench_nef32,
        icount_bench_nef64,
        icount_bench_nextafter,
        icount_bench_nextafterf,
        icount_bench_pow,
        icount_bench_powf,
        icount_bench_powif128,
        icount_bench_powif32,
        icount_bench_powif64,
        icount_bench_remainder,
        icount_bench_remainderf,
        icount_bench_remquo,
        icount_bench_remquof,
        icount_bench_rint,
        icount_bench_rintf,
        icount_bench_rintf128,
        icount_bench_rintf16,
        icount_bench_round,
        icount_bench_roundeven,
        icount_bench_roundevenf,
        icount_bench_roundevenf128,
        icount_bench_roundevenf16,
        icount_bench_roundf,
        icount_bench_roundf128,
        icount_bench_roundf16,
        icount_bench_scalbn,
        icount_bench_scalbnf,
        icount_bench_scalbnf128,
        icount_bench_scalbnf16,
        icount_bench_sin,
        icount_bench_sincos,
        icount_bench_sincosf,
        icount_bench_sinf,
        icount_bench_sinh,
        icount_bench_sinhf,
        icount_bench_sqrt,
        icount_bench_sqrtf,
        icount_bench_sqrtf128,
        icount_bench_sqrtf16,
        icount_bench_subf128,
        icount_bench_subf16,
        icount_bench_subf32,
        icount_bench_subf64,
        icount_bench_tan,
        icount_bench_tanf,
        icount_bench_tanh,
        icount_bench_tanhf,
        icount_bench_tgamma,
        icount_bench_tgammaf,
        icount_bench_trailing_zeros_u128,
        icount_bench_trailing_zeros_u32,
        icount_bench_trailing_zeros_u64,
        icount_bench_trunc,
        icount_bench_truncf,
        icount_bench_truncf128,
        icount_bench_truncf16,
        icount_bench_unordf128,
        icount_bench_unordf16,
        icount_bench_unordf32,
        icount_bench_unordf64,
        icount_bench_y0,
        icount_bench_y0f,
        icount_bench_y1,
        icount_bench_y1f,
        icount_bench_yn,
        icount_bench_ynf,
        // verify-sorted-end
        // verify-apilist-end
    ]
);

fn setup_u128_mul() -> Vec<(u128, u128)> {
    let step = u128::MAX / 300;
    let mut x = 0u128;
    let mut y = 0u128;
    let mut v = Vec::new();

    loop {
        'inner: loop {
            match y.checked_add(step) {
                Some(new) => y = new,
                None => break 'inner,
            }

            v.push((x, y))
        }

        match x.checked_add(step) {
            Some(new) => x = new,
            None => break,
        }
    }

    v
}

fn setup_u256_add() -> Vec<(u256, u256)> {
    let mut v = Vec::new();
    for (x, y) in setup_u128_mul() {
        // square the u128 inputs to cover most of the u256 range
        v.push((x.widen_mul(x), y.widen_mul(y)));
    }
    // Doesn't get covered by `u128:MAX^2`
    v.push((u256::MAX, u256::MAX));
    v
}

fn setup_u256_shift() -> Vec<(u256, u32)> {
    let mut v = Vec::new();

    for (x, _) in setup_u128_mul() {
        let x2 = x.widen_mul(x);
        for y in 0u32..256 {
            v.push((x2, y));
        }
    }

    v
}

fn setup_i256_shift() -> Vec<(i256, u32)> {
    setup_u256_shift()
        .into_iter()
        .map(|(x, i)| (x.signed(), i))
        .collect()
}

#[library_benchmark]
#[bench::linspace(setup_u128_mul())]
fn icount_bench_u128_widen_mul(cases: Vec<(u128, u128)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x).zero_widen_mul(black_box(y)));
    }
}

#[library_benchmark]
#[bench::linspace(setup_u128_mul())]
fn icount_bench_u256_narrowing_div(cases: Vec<(u128, u128)>) {
    use libm::support::NarrowingDiv;
    for (x, y) in cases.iter().copied() {
        let x = black_box(x.widen_hi());
        let y = black_box(y);
        black_box(x.checked_narrowing_div_rem(y));
    }
}

#[library_benchmark]
#[bench::linspace(setup_u256_add())]
fn icount_bench_u256_add(cases: Vec<(u256, u256)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x) + black_box(y));
    }
}

#[library_benchmark]
#[bench::linspace(setup_u256_add())]
fn icount_bench_u256_sub(cases: Vec<(u256, u256)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x) - black_box(y));
    }
}

#[library_benchmark]
#[bench::linspace(setup_u256_shift())]
fn icount_bench_u256_shl(cases: Vec<(u256, u32)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x) << black_box(y));
    }
}

#[library_benchmark]
#[bench::linspace(setup_u256_shift())]
fn icount_bench_u256_shr(cases: Vec<(u256, u32)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x) >> black_box(y));
    }
}

#[library_benchmark]
#[bench::linspace(setup_i256_shift())]
fn icount_bench_i256_shr(cases: Vec<(i256, u32)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x) >> black_box(y));
    }
}

library_benchmark_group!(
    name = icount_bench_u128_group,
    benchmarks = [
        icount_bench_u128_widen_mul,
        icount_bench_u256_narrowing_div,
        icount_bench_u256_add,
        icount_bench_u256_sub,
        icount_bench_u256_shl,
        icount_bench_u256_shr,
        icount_bench_i256_shr,
    ]
);

#[library_benchmark]
#[bench::short("0x12.34p+8")]
#[bench::max("0x1.ffcp+15")]
fn icount_bench_hf16(s: &str) -> f16 {
    black_box(hf16(s))
}

#[library_benchmark]
#[bench::short("0x12.34p+8")]
#[bench::max("0x1.fffffep+127")]
fn icount_bench_hf32(s: &str) -> f32 {
    black_box(hf32(s))
}

#[library_benchmark]
#[bench::short("0x12.34p+8")]
#[bench::max("0x1.fffffffffffffp+1023")]
fn icount_bench_hf64(s: &str) -> f64 {
    black_box(hf64(s))
}

#[library_benchmark]
#[bench::short("0x12.34p+8")]
#[bench::max("0x1.ffffffffffffffffffffffffffffp+16383")]
fn icount_bench_hf128(s: &str) -> f128 {
    black_box(hf128(s))
}

library_benchmark_group!(
    name = icount_bench_hf_parse_group,
    benchmarks = [
        icount_bench_hf16,
        icount_bench_hf32,
        icount_bench_hf64,
        icount_bench_hf128,
    ]
);

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f16::MAX)]
fn icount_bench_print_hf16(x: f16) -> String {
    black_box(Hex(x).to_string())
}

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f32::MAX)]
fn icount_bench_print_hf32(x: f32) -> String {
    black_box(Hex(x).to_string())
}

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f64::MAX)]
fn icount_bench_print_hf64(x: f64) -> String {
    black_box(Hex(x).to_string())
}

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f128::MAX)]
fn icount_bench_print_hf128(x: f128) -> String {
    black_box(Hex(x).to_string())
}

library_benchmark_group!(
    name = icount_bench_hf_print_group,
    benchmarks = [
        icount_bench_print_hf16,
        icount_bench_print_hf32,
        icount_bench_print_hf64,
        icount_bench_print_hf128,
    ]
);

main!(
    library_benchmark_groups = [
        // Benchmarks not related to public libm math
        icount_bench_u128_group,
        icount_bench_hf_parse_group,
        icount_bench_hf_print_group,
        icount_bench_math_group,
    ]
);
