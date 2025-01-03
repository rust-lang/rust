//! Benchmarks that use `iai-cachegrind` to be reasonably CI-stable.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use libm_test::gen::spaced;
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
                    GeneratorKind::QuickSpaced
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

main!(
    library_benchmark_groups = icount_bench_acos_group,
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
    icount_bench_fmaf_group,
    icount_bench_fmax_group,
    icount_bench_fmaxf128_group,
    icount_bench_fmaxf16_group,
    icount_bench_fmaxf_group,
    icount_bench_fmin_group,
    icount_bench_fminf128_group,
    icount_bench_fminf16_group,
    icount_bench_fminf_group,
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
    icount_bench_roundf128_group,
    icount_bench_roundf16_group,
    icount_bench_roundf_group,
    icount_bench_scalbn_group,
    icount_bench_scalbnf128_group,
    icount_bench_scalbnf16_group,
    icount_bench_scalbnf_group,
    icount_bench_sin_group,
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
);
