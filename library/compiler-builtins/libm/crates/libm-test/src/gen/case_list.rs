//! Test cases to verify specific values.
//!
//! Each routine can have a set of inputs and, optinoally, outputs. If an output is provided, it
//! will be used to check against. If only inputs are provided, the case will be checked against
//! a basis.
//!
//! This is useful for adding regression tests or expected failures.

#[cfg(f128_enabled)]
use libm::hf128;

use crate::{CheckBasis, CheckCtx, GeneratorKind, MathOp, op};

pub struct TestCase<Op: MathOp> {
    pub input: Op::RustArgs,
    pub output: Option<Op::RustRet>,
}

impl<Op: MathOp> TestCase<Op> {
    #[expect(dead_code)]
    fn append_inputs(v: &mut Vec<Self>, l: &[Op::RustArgs]) {
        v.extend(l.iter().copied().map(|input| Self { input, output: None }));
    }

    fn append_pairs(v: &mut Vec<Self>, l: &[(Op::RustArgs, Option<Op::RustRet>)])
    where
        Op::RustRet: Copy,
    {
        v.extend(l.iter().copied().map(|(input, output)| Self { input, output }));
    }
}

fn acos_cases() -> Vec<TestCase<op::acos::Routine>> {
    vec![]
}

fn acosf_cases() -> Vec<TestCase<op::acosf::Routine>> {
    vec![]
}

fn acosh_cases() -> Vec<TestCase<op::acosh::Routine>> {
    vec![]
}

fn acoshf_cases() -> Vec<TestCase<op::acoshf::Routine>> {
    vec![]
}

fn asin_cases() -> Vec<TestCase<op::asin::Routine>> {
    vec![]
}

fn asinf_cases() -> Vec<TestCase<op::asinf::Routine>> {
    vec![]
}

fn asinh_cases() -> Vec<TestCase<op::asinh::Routine>> {
    vec![]
}

fn asinhf_cases() -> Vec<TestCase<op::asinhf::Routine>> {
    vec![]
}

fn atan_cases() -> Vec<TestCase<op::atan::Routine>> {
    vec![]
}

fn atan2_cases() -> Vec<TestCase<op::atan2::Routine>> {
    vec![]
}

fn atan2f_cases() -> Vec<TestCase<op::atan2f::Routine>> {
    vec![]
}

fn atanf_cases() -> Vec<TestCase<op::atanf::Routine>> {
    vec![]
}

fn atanh_cases() -> Vec<TestCase<op::atanh::Routine>> {
    vec![]
}

fn atanhf_cases() -> Vec<TestCase<op::atanhf::Routine>> {
    vec![]
}

fn cbrt_cases() -> Vec<TestCase<op::cbrt::Routine>> {
    vec![]
}

fn cbrtf_cases() -> Vec<TestCase<op::cbrtf::Routine>> {
    vec![]
}

fn ceil_cases() -> Vec<TestCase<op::ceil::Routine>> {
    vec![]
}

fn ceilf_cases() -> Vec<TestCase<op::ceilf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn ceilf128_cases() -> Vec<TestCase<op::ceilf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn ceilf16_cases() -> Vec<TestCase<op::ceilf16::Routine>> {
    vec![]
}

fn copysign_cases() -> Vec<TestCase<op::copysign::Routine>> {
    vec![]
}

fn copysignf_cases() -> Vec<TestCase<op::copysignf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn copysignf128_cases() -> Vec<TestCase<op::copysignf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn copysignf16_cases() -> Vec<TestCase<op::copysignf16::Routine>> {
    vec![]
}

fn cos_cases() -> Vec<TestCase<op::cos::Routine>> {
    vec![]
}

fn cosf_cases() -> Vec<TestCase<op::cosf::Routine>> {
    vec![]
}

fn cosh_cases() -> Vec<TestCase<op::cosh::Routine>> {
    vec![]
}

fn coshf_cases() -> Vec<TestCase<op::coshf::Routine>> {
    vec![]
}

fn erf_cases() -> Vec<TestCase<op::erf::Routine>> {
    vec![]
}

fn erfc_cases() -> Vec<TestCase<op::erfc::Routine>> {
    vec![]
}

fn erfcf_cases() -> Vec<TestCase<op::erfcf::Routine>> {
    vec![]
}

fn erff_cases() -> Vec<TestCase<op::erff::Routine>> {
    vec![]
}

fn exp_cases() -> Vec<TestCase<op::exp::Routine>> {
    vec![]
}

fn exp10_cases() -> Vec<TestCase<op::exp10::Routine>> {
    vec![]
}

fn exp10f_cases() -> Vec<TestCase<op::exp10f::Routine>> {
    vec![]
}

fn exp2_cases() -> Vec<TestCase<op::exp2::Routine>> {
    vec![]
}

fn exp2f_cases() -> Vec<TestCase<op::exp2f::Routine>> {
    vec![]
}

fn expf_cases() -> Vec<TestCase<op::expf::Routine>> {
    vec![]
}

fn expm1_cases() -> Vec<TestCase<op::expm1::Routine>> {
    vec![]
}

fn expm1f_cases() -> Vec<TestCase<op::expm1f::Routine>> {
    vec![]
}

fn fabs_cases() -> Vec<TestCase<op::fabs::Routine>> {
    vec![]
}

fn fabsf_cases() -> Vec<TestCase<op::fabsf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fabsf128_cases() -> Vec<TestCase<op::fabsf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fabsf16_cases() -> Vec<TestCase<op::fabsf16::Routine>> {
    vec![]
}

fn fdim_cases() -> Vec<TestCase<op::fdim::Routine>> {
    vec![]
}

fn fdimf_cases() -> Vec<TestCase<op::fdimf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fdimf128_cases() -> Vec<TestCase<op::fdimf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fdimf16_cases() -> Vec<TestCase<op::fdimf16::Routine>> {
    vec![]
}

fn floor_cases() -> Vec<TestCase<op::floor::Routine>> {
    vec![]
}

fn floorf_cases() -> Vec<TestCase<op::floorf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn floorf128_cases() -> Vec<TestCase<op::floorf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn floorf16_cases() -> Vec<TestCase<op::floorf16::Routine>> {
    vec![]
}

fn fma_cases() -> Vec<TestCase<op::fma::Routine>> {
    let mut v = vec![];
    TestCase::append_pairs(
        &mut v,
        &[
            // Previous failure with incorrect sign
            ((5e-324, -5e-324, 0.0), Some(-0.0)),
        ],
    );
    v
}

fn fmaf_cases() -> Vec<TestCase<op::fmaf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fmaf128_cases() -> Vec<TestCase<op::fmaf128::Routine>> {
    let mut v = vec![];
    TestCase::append_pairs(
        &mut v,
        &[
            (
                // Tricky rounding case that previously failed in extensive tests
                (
                    hf128!("-0x1.1966cc01966cc01966cc01966f06p-25"),
                    hf128!("-0x1.669933fe69933fe69933fe6997c9p-16358"),
                    hf128!("-0x0.000000000000000000000000048ap-16382"),
                ),
                Some(hf128!("0x0.c5171470a3ff5e0f68d751491b18p-16382")),
            ),
            (
                // Subnormal edge case that caused a failure
                (
                    hf128!("0x0.7ffffffffffffffffffffffffff7p-16382"),
                    hf128!("0x1.ffffffffffffffffffffffffffffp-1"),
                    hf128!("0x0.8000000000000000000000000009p-16382"),
                ),
                Some(hf128!("0x1.0000000000000000000000000000p-16382")),
            ),
        ],
    );
    v
}

#[cfg(f16_enabled)]
fn fmaxf16_cases() -> Vec<TestCase<op::fmaxf16::Routine>> {
    vec![]
}

fn fmaxf_cases() -> Vec<TestCase<op::fmaxf::Routine>> {
    vec![]
}

fn fmax_cases() -> Vec<TestCase<op::fmax::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fmaxf128_cases() -> Vec<TestCase<op::fmaxf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fmaximumf16_cases() -> Vec<TestCase<op::fmaximumf16::Routine>> {
    vec![]
}

fn fmaximumf_cases() -> Vec<TestCase<op::fmaximumf::Routine>> {
    vec![]
}

fn fmaximum_cases() -> Vec<TestCase<op::fmaximum::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fmaximumf128_cases() -> Vec<TestCase<op::fmaximumf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fmaximum_numf16_cases() -> Vec<TestCase<op::fmaximum_numf16::Routine>> {
    vec![]
}

fn fmaximum_numf_cases() -> Vec<TestCase<op::fmaximum_numf::Routine>> {
    vec![]
}

fn fmaximum_num_cases() -> Vec<TestCase<op::fmaximum_num::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fmaximum_numf128_cases() -> Vec<TestCase<op::fmaximum_numf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fminf16_cases() -> Vec<TestCase<op::fminf16::Routine>> {
    vec![]
}

fn fminf_cases() -> Vec<TestCase<op::fminf::Routine>> {
    vec![]
}

fn fmin_cases() -> Vec<TestCase<op::fmin::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fminf128_cases() -> Vec<TestCase<op::fminf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fminimumf16_cases() -> Vec<TestCase<op::fminimumf16::Routine>> {
    vec![]
}

fn fminimumf_cases() -> Vec<TestCase<op::fminimumf::Routine>> {
    vec![]
}

fn fminimum_cases() -> Vec<TestCase<op::fminimum::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fminimumf128_cases() -> Vec<TestCase<op::fminimumf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fminimum_numf16_cases() -> Vec<TestCase<op::fminimum_numf16::Routine>> {
    vec![]
}

fn fminimum_numf_cases() -> Vec<TestCase<op::fminimum_numf::Routine>> {
    vec![]
}

fn fminimum_num_cases() -> Vec<TestCase<op::fminimum_num::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fminimum_numf128_cases() -> Vec<TestCase<op::fminimum_numf128::Routine>> {
    vec![]
}

fn fmod_cases() -> Vec<TestCase<op::fmod::Routine>> {
    vec![]
}

fn fmodf_cases() -> Vec<TestCase<op::fmodf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn fmodf128_cases() -> Vec<TestCase<op::fmodf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn fmodf16_cases() -> Vec<TestCase<op::fmodf16::Routine>> {
    vec![]
}

fn frexp_cases() -> Vec<TestCase<op::frexp::Routine>> {
    vec![]
}

fn frexpf_cases() -> Vec<TestCase<op::frexpf::Routine>> {
    vec![]
}

fn hypot_cases() -> Vec<TestCase<op::hypot::Routine>> {
    vec![]
}

fn hypotf_cases() -> Vec<TestCase<op::hypotf::Routine>> {
    vec![]
}

fn ilogb_cases() -> Vec<TestCase<op::ilogb::Routine>> {
    vec![]
}

fn ilogbf_cases() -> Vec<TestCase<op::ilogbf::Routine>> {
    vec![]
}

fn j0_cases() -> Vec<TestCase<op::j0::Routine>> {
    vec![]
}

fn j0f_cases() -> Vec<TestCase<op::j0f::Routine>> {
    vec![]
}

fn j1_cases() -> Vec<TestCase<op::j1::Routine>> {
    vec![]
}

fn j1f_cases() -> Vec<TestCase<op::j1f::Routine>> {
    vec![]
}

fn jn_cases() -> Vec<TestCase<op::jn::Routine>> {
    vec![]
}

fn jnf_cases() -> Vec<TestCase<op::jnf::Routine>> {
    vec![]
}

fn ldexp_cases() -> Vec<TestCase<op::ldexp::Routine>> {
    vec![]
}

fn ldexpf_cases() -> Vec<TestCase<op::ldexpf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn ldexpf128_cases() -> Vec<TestCase<op::ldexpf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn ldexpf16_cases() -> Vec<TestCase<op::ldexpf16::Routine>> {
    vec![]
}

fn lgamma_cases() -> Vec<TestCase<op::lgamma::Routine>> {
    vec![]
}

fn lgamma_r_cases() -> Vec<TestCase<op::lgamma_r::Routine>> {
    vec![]
}

fn lgammaf_cases() -> Vec<TestCase<op::lgammaf::Routine>> {
    vec![]
}

fn lgammaf_r_cases() -> Vec<TestCase<op::lgammaf_r::Routine>> {
    vec![]
}

fn log_cases() -> Vec<TestCase<op::log::Routine>> {
    vec![]
}

fn log10_cases() -> Vec<TestCase<op::log10::Routine>> {
    vec![]
}

fn log10f_cases() -> Vec<TestCase<op::log10f::Routine>> {
    vec![]
}

fn log1p_cases() -> Vec<TestCase<op::log1p::Routine>> {
    vec![]
}

fn log1pf_cases() -> Vec<TestCase<op::log1pf::Routine>> {
    vec![]
}

fn log2_cases() -> Vec<TestCase<op::log2::Routine>> {
    vec![]
}

fn log2f_cases() -> Vec<TestCase<op::log2f::Routine>> {
    vec![]
}

fn logf_cases() -> Vec<TestCase<op::logf::Routine>> {
    vec![]
}

fn modf_cases() -> Vec<TestCase<op::modf::Routine>> {
    vec![]
}

fn modff_cases() -> Vec<TestCase<op::modff::Routine>> {
    vec![]
}

fn nextafter_cases() -> Vec<TestCase<op::nextafter::Routine>> {
    vec![]
}

fn nextafterf_cases() -> Vec<TestCase<op::nextafterf::Routine>> {
    vec![]
}

fn pow_cases() -> Vec<TestCase<op::pow::Routine>> {
    vec![]
}

fn powf_cases() -> Vec<TestCase<op::powf::Routine>> {
    vec![]
}

fn remainder_cases() -> Vec<TestCase<op::remainder::Routine>> {
    vec![]
}

fn remainderf_cases() -> Vec<TestCase<op::remainderf::Routine>> {
    vec![]
}

fn remquo_cases() -> Vec<TestCase<op::remquo::Routine>> {
    vec![]
}

fn remquof_cases() -> Vec<TestCase<op::remquof::Routine>> {
    vec![]
}

fn rint_cases() -> Vec<TestCase<op::rint::Routine>> {
    vec![]
}

fn rintf_cases() -> Vec<TestCase<op::rintf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn rintf128_cases() -> Vec<TestCase<op::rintf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn rintf16_cases() -> Vec<TestCase<op::rintf16::Routine>> {
    vec![]
}

fn round_cases() -> Vec<TestCase<op::round::Routine>> {
    vec![]
}

fn roundf_cases() -> Vec<TestCase<op::roundf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn roundf128_cases() -> Vec<TestCase<op::roundf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn roundf16_cases() -> Vec<TestCase<op::roundf16::Routine>> {
    vec![]
}

fn scalbn_cases() -> Vec<TestCase<op::scalbn::Routine>> {
    vec![]
}

fn scalbnf_cases() -> Vec<TestCase<op::scalbnf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn scalbnf128_cases() -> Vec<TestCase<op::scalbnf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn scalbnf16_cases() -> Vec<TestCase<op::scalbnf16::Routine>> {
    vec![]
}

fn sin_cases() -> Vec<TestCase<op::sin::Routine>> {
    vec![]
}

fn sincos_cases() -> Vec<TestCase<op::sincos::Routine>> {
    vec![]
}

fn sincosf_cases() -> Vec<TestCase<op::sincosf::Routine>> {
    vec![]
}

fn sinf_cases() -> Vec<TestCase<op::sinf::Routine>> {
    vec![]
}

fn sinh_cases() -> Vec<TestCase<op::sinh::Routine>> {
    vec![]
}

fn sinhf_cases() -> Vec<TestCase<op::sinhf::Routine>> {
    vec![]
}

fn sqrt_cases() -> Vec<TestCase<op::sqrt::Routine>> {
    vec![]
}

fn sqrtf_cases() -> Vec<TestCase<op::sqrtf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn sqrtf128_cases() -> Vec<TestCase<op::sqrtf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn sqrtf16_cases() -> Vec<TestCase<op::sqrtf16::Routine>> {
    vec![]
}

fn tan_cases() -> Vec<TestCase<op::tan::Routine>> {
    vec![]
}

fn tanf_cases() -> Vec<TestCase<op::tanf::Routine>> {
    vec![]
}

fn tanh_cases() -> Vec<TestCase<op::tanh::Routine>> {
    vec![]
}

fn tanhf_cases() -> Vec<TestCase<op::tanhf::Routine>> {
    vec![]
}

fn tgamma_cases() -> Vec<TestCase<op::tgamma::Routine>> {
    vec![]
}

fn tgammaf_cases() -> Vec<TestCase<op::tgammaf::Routine>> {
    vec![]
}

fn trunc_cases() -> Vec<TestCase<op::trunc::Routine>> {
    vec![]
}

fn truncf_cases() -> Vec<TestCase<op::truncf::Routine>> {
    vec![]
}

#[cfg(f128_enabled)]
fn truncf128_cases() -> Vec<TestCase<op::truncf128::Routine>> {
    vec![]
}

#[cfg(f16_enabled)]
fn truncf16_cases() -> Vec<TestCase<op::truncf16::Routine>> {
    vec![]
}

fn y0_cases() -> Vec<TestCase<op::y0::Routine>> {
    vec![]
}

fn y0f_cases() -> Vec<TestCase<op::y0f::Routine>> {
    vec![]
}

fn y1_cases() -> Vec<TestCase<op::y1::Routine>> {
    vec![]
}

fn y1f_cases() -> Vec<TestCase<op::y1f::Routine>> {
    vec![]
}

fn yn_cases() -> Vec<TestCase<op::yn::Routine>> {
    vec![]
}

fn ynf_cases() -> Vec<TestCase<op::ynf::Routine>> {
    vec![]
}

pub trait CaseListInput: MathOp + Sized {
    fn get_cases() -> Vec<TestCase<Self>>;
}

macro_rules! impl_case_list {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
    ) => {
        paste::paste! {
            $(#[$attr])*
            impl CaseListInput for crate::op::$fn_name::Routine {
                fn get_cases() -> Vec<TestCase<Self>> {
                    [< $fn_name _cases >]()
                }
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: impl_case_list,
}

/// This is the test generator for standalone tests, i.e. those with no basis. For this, it
/// only extracts tests with a known output.
pub fn get_test_cases_standalone<Op>(
    ctx: &CheckCtx,
) -> impl Iterator<Item = (Op::RustArgs, Op::RustRet)> + use<'_, Op>
where
    Op: MathOp + CaseListInput,
{
    assert_eq!(ctx.basis, CheckBasis::None);
    assert_eq!(ctx.gen_kind, GeneratorKind::List);
    Op::get_cases().into_iter().filter_map(|x| x.output.map(|o| (x.input, o)))
}

/// Opposite of the above; extract only test cases that don't have a known output, to be run
/// against a basis.
pub fn get_test_cases_basis<Op>(
    ctx: &CheckCtx,
) -> (impl Iterator<Item = Op::RustArgs> + use<'_, Op>, u64)
where
    Op: MathOp + CaseListInput,
{
    assert_ne!(ctx.basis, CheckBasis::None);
    assert_eq!(ctx.gen_kind, GeneratorKind::List);

    let cases = Op::get_cases();
    let count: u64 = cases.iter().filter(|case| case.output.is_none()).count().try_into().unwrap();

    (cases.into_iter().filter(|x| x.output.is_none()).map(|x| x.input), count)
}
