//! Benchmarks that use `iai-cachegrind` to be reasonably CI-stable.
#![feature(f16)]
#![feature(f128)]

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use libm::support::{HInt, Hexf, hf16, hf32, hf64, hf128, u256};
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

            library_benchmark_group!(
                name = [< icount_bench_ $fn_name _group  >];
                benchmarks = [< icount_bench_ $fn_name >]
            );
        }
    };
}

libm_macros::for_each_function! {
    callback: icount_benches,
}

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

#[library_benchmark]
#[bench::linspace(setup_u128_mul())]
fn icount_bench_u128_widen_mul(cases: Vec<(u128, u128)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x).zero_widen_mul(black_box(y)));
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
#[bench::linspace(setup_u256_shift())]
fn icount_bench_u256_shr(cases: Vec<(u256, u32)>) {
    for (x, y) in cases.iter().copied() {
        black_box(black_box(x) >> black_box(y));
    }
}

library_benchmark_group!(
    name = icount_bench_u128_group;
    benchmarks = icount_bench_u128_widen_mul, icount_bench_u256_add, icount_bench_u256_shr
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
    name = icount_bench_hf_parse_group;
    benchmarks =
    icount_bench_hf16,
    icount_bench_hf32,
    icount_bench_hf64,
    icount_bench_hf128
);

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f16::MAX)]
fn icount_bench_print_hf16(x: f16) -> String {
    black_box(Hexf(x).to_string())
}

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f32::MAX)]
fn icount_bench_print_hf32(x: f32) -> String {
    black_box(Hexf(x).to_string())
}

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f64::MAX)]
fn icount_bench_print_hf64(x: f64) -> String {
    black_box(Hexf(x).to_string())
}

#[library_benchmark]
#[bench::short(1.015625)]
#[bench::max(f128::MAX)]
fn icount_bench_print_hf128(x: f128) -> String {
    black_box(Hexf(x).to_string())
}

library_benchmark_group!(
    name = icount_bench_hf_print_group;
    benchmarks =
    icount_bench_print_hf16,
    icount_bench_print_hf32,
    icount_bench_print_hf64,
    icount_bench_print_hf128
);

main!(
    library_benchmark_groups =
    // Benchmarks not related to public libm math
    icount_bench_u128_group,
    icount_bench_hf_parse_group,
    icount_bench_hf_print_group,
    // verify-apilist-start
    // verify-sorted-start
    icount_bench_acos_group,
    icount_bench_acosf_group,
    icount_bench_acosh_group,
    icount_bench_acoshf_group,
    icount_bench_asin_group,
    icount_bench_asinf_group,
    icount_bench_asinh_group,
    icount_bench_asinhf_group,
    icount_bench_atan2_group,
    icount_bench_atan2f_group,
    icount_bench_atan_group,
    icount_bench_atanf_group,
    icount_bench_atanh_group,
    icount_bench_atanhf_group,
    icount_bench_cbrt_group,
    icount_bench_cbrtf_group,
    icount_bench_ceil_group,
    icount_bench_ceilf128_group,
    icount_bench_ceilf16_group,
    icount_bench_ceilf_group,
    icount_bench_copysign_group,
    icount_bench_copysignf128_group,
    icount_bench_copysignf16_group,
    icount_bench_copysignf_group,
    icount_bench_cos_group,
    icount_bench_cosf_group,
    icount_bench_cosh_group,
    icount_bench_coshf_group,
    icount_bench_erf_group,
    icount_bench_erfc_group,
    icount_bench_erfcf_group,
    icount_bench_erff_group,
    icount_bench_exp10_group,
    icount_bench_exp10f_group,
    icount_bench_exp2_group,
    icount_bench_exp2f_group,
    icount_bench_exp_group,
    icount_bench_expf_group,
    icount_bench_expm1_group,
    icount_bench_expm1f_group,
    icount_bench_fabs_group,
    icount_bench_fabsf128_group,
    icount_bench_fabsf16_group,
    icount_bench_fabsf_group,
    icount_bench_fdim_group,
    icount_bench_fdimf128_group,
    icount_bench_fdimf16_group,
    icount_bench_fdimf_group,
    icount_bench_floor_group,
    icount_bench_floorf128_group,
    icount_bench_floorf16_group,
    icount_bench_floorf_group,
    icount_bench_fma_group,
    icount_bench_fmaf128_group,
    icount_bench_fmaf_group,
    icount_bench_fmax_group,
    icount_bench_fmaxf128_group,
    icount_bench_fmaxf16_group,
    icount_bench_fmaxf_group,
    icount_bench_fmaximum_group,
    icount_bench_fmaximum_num_group,
    icount_bench_fmaximum_numf128_group,
    icount_bench_fmaximum_numf16_group,
    icount_bench_fmaximum_numf_group,
    icount_bench_fmaximumf128_group,
    icount_bench_fmaximumf16_group,
    icount_bench_fmaximumf_group,
    icount_bench_fmin_group,
    icount_bench_fminf128_group,
    icount_bench_fminf16_group,
    icount_bench_fminf_group,
    icount_bench_fminimum_group,
    icount_bench_fminimum_num_group,
    icount_bench_fminimum_numf128_group,
    icount_bench_fminimum_numf16_group,
    icount_bench_fminimum_numf_group,
    icount_bench_fminimumf128_group,
    icount_bench_fminimumf16_group,
    icount_bench_fminimumf_group,
    icount_bench_fmod_group,
    icount_bench_fmodf128_group,
    icount_bench_fmodf16_group,
    icount_bench_fmodf_group,
    icount_bench_frexp_group,
    icount_bench_frexpf_group,
    icount_bench_hypot_group,
    icount_bench_hypotf_group,
    icount_bench_ilogb_group,
    icount_bench_ilogbf_group,
    icount_bench_j0_group,
    icount_bench_j0f_group,
    icount_bench_j1_group,
    icount_bench_j1f_group,
    icount_bench_jn_group,
    icount_bench_jnf_group,
    icount_bench_ldexp_group,
    icount_bench_ldexpf128_group,
    icount_bench_ldexpf16_group,
    icount_bench_ldexpf_group,
    icount_bench_lgamma_group,
    icount_bench_lgamma_r_group,
    icount_bench_lgammaf_group,
    icount_bench_lgammaf_r_group,
    icount_bench_log10_group,
    icount_bench_log10f_group,
    icount_bench_log1p_group,
    icount_bench_log1pf_group,
    icount_bench_log2_group,
    icount_bench_log2f_group,
    icount_bench_log_group,
    icount_bench_logf_group,
    icount_bench_modf_group,
    icount_bench_modff_group,
    icount_bench_nextafter_group,
    icount_bench_nextafterf_group,
    icount_bench_pow_group,
    icount_bench_powf_group,
    icount_bench_remainder_group,
    icount_bench_remainderf_group,
    icount_bench_remquo_group,
    icount_bench_remquof_group,
    icount_bench_rint_group,
    icount_bench_rintf128_group,
    icount_bench_rintf16_group,
    icount_bench_rintf_group,
    icount_bench_round_group,
    icount_bench_roundeven_group,
    icount_bench_roundevenf128_group,
    icount_bench_roundevenf16_group,
    icount_bench_roundevenf_group,
    icount_bench_roundf128_group,
    icount_bench_roundf16_group,
    icount_bench_roundf_group,
    icount_bench_scalbn_group,
    icount_bench_scalbnf128_group,
    icount_bench_scalbnf16_group,
    icount_bench_scalbnf_group,
    icount_bench_sin_group,
    icount_bench_sincos_group,
    icount_bench_sincosf_group,
    icount_bench_sinf_group,
    icount_bench_sinh_group,
    icount_bench_sinhf_group,
    icount_bench_sqrt_group,
    icount_bench_sqrtf128_group,
    icount_bench_sqrtf16_group,
    icount_bench_sqrtf_group,
    icount_bench_tan_group,
    icount_bench_tanf_group,
    icount_bench_tanh_group,
    icount_bench_tanhf_group,
    icount_bench_tgamma_group,
    icount_bench_tgammaf_group,
    icount_bench_trunc_group,
    icount_bench_truncf128_group,
    icount_bench_truncf16_group,
    icount_bench_truncf_group,
    icount_bench_y0_group,
    icount_bench_y0f_group,
    icount_bench_y1_group,
    icount_bench_y1f_group,
    icount_bench_yn_group,
    icount_bench_ynf_group,
    // verify-sorted-end
    // verify-apilist-end
);
