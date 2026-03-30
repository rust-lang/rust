//! Test cases to verify specific values.
//!
//! Each routine can have a set of inputs and, optinoally, outputs. If an output is provided, it
//! will be used to check against. If only inputs are provided, the case will be checked against
//! a basis.
//!
//! This is useful for adding regression tests or expected failures.

#[cfg(f128_enabled)]
use libm::hf128;
use libm::{hf32, hf64};

use crate::{CheckBasis, CheckCtx, GeneratorKind, MathOp, op};

pub struct TestCase<Op: MathOp> {
    pub input: Op::RustArgs,
    pub output: Option<Op::RustRet>,
}

impl<Op: MathOp> TestCase<Op> {
    /// Turn into a different operation with the same types.
    fn cast<Op2: MathOp>(self) -> TestCase<Op2>
    where
        Op2::RustArgs: From<Op::RustArgs>,
        Op2::RustRet: From<Op::RustRet>,
    {
        TestCase {
            input: self.input.into(),
            output: self.output.map(Into::into),
        }
    }
}

macro_rules! cases {
    (
        $(
            $(#[$($meta:tt)*])*
            ($($tt:tt)*)
        ),* $(,)?
    ) => {{
        Vec::from_iter([
            $(
               $(#[$($meta)*])*
                cases!(@single $($tt)*),
            )*
        ])
    }};

    // Variant without a result, which will check against MPFR.
    (@single ($($arg:expr),* $(,)?), None $(,)?) => {
        TestCase{
            input: ($($arg,)*),
            output: None,
        }
    };

    // Variant for when the result is specified.
    (@single ($($arg:expr),* $(,)?), $res:expr $(,)?) => {
        TestCase{
            input: ($($arg,)*),
            output: Some($res),
        }
    };
}

/********************************
 * compiler-builtins test cases *
 ********************************/

#[cfg(f16_enabled)]
fn addf16_cases() -> Vec<TestCase<op::addf16::Routine>> {
    cases![]
}

fn addf32_cases() -> Vec<TestCase<op::addf32::Routine>> {
    cases![]
}

fn addf64_cases() -> Vec<TestCase<op::addf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn addf128_cases() -> Vec<TestCase<op::addf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn subf16_cases() -> Vec<TestCase<op::subf16::Routine>> {
    cases![]
}

fn subf32_cases() -> Vec<TestCase<op::subf32::Routine>> {
    cases![]
}

fn subf64_cases() -> Vec<TestCase<op::subf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn subf128_cases() -> Vec<TestCase<op::subf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn mulf16_cases() -> Vec<TestCase<op::mulf16::Routine>> {
    cases![]
}

fn mulf32_cases() -> Vec<TestCase<op::mulf32::Routine>> {
    cases![]
}

fn mulf64_cases() -> Vec<TestCase<op::mulf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn mulf128_cases() -> Vec<TestCase<op::mulf128::Routine>> {
    cases![]
}

fn divf32_cases() -> Vec<TestCase<op::divf32::Routine>> {
    cases![]
}

fn divf64_cases() -> Vec<TestCase<op::divf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn divf128_cases() -> Vec<TestCase<op::divf128::Routine>> {
    cases![]
}

fn powif32_cases() -> Vec<TestCase<op::powif32::Routine>> {
    cases![]
}

fn powif64_cases() -> Vec<TestCase<op::powif64::Routine>> {
    cases![
        // High error
        ((0.9999497584118668, -5858518), 6.823250355352412e127),
    ]
}

#[cfg(f128_enabled)]
fn powif128_cases() -> Vec<TestCase<op::powif128::Routine>> {
    cases![]
}

/* comparison */

#[cfg(f16_enabled)]
fn eqf16_cases() -> Vec<TestCase<op::eqf16::Routine>> {
    cases![]
}

fn eqf32_cases() -> Vec<TestCase<op::eqf32::Routine>> {
    cases![]
}

fn eqf64_cases() -> Vec<TestCase<op::eqf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn eqf128_cases() -> Vec<TestCase<op::eqf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn gtf16_cases() -> Vec<TestCase<op::gtf16::Routine>> {
    cases![]
}

fn gtf32_cases() -> Vec<TestCase<op::gtf32::Routine>> {
    cases![]
}

fn gtf64_cases() -> Vec<TestCase<op::gtf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn gtf128_cases() -> Vec<TestCase<op::gtf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn gef16_cases() -> Vec<TestCase<op::gef16::Routine>> {
    cases![]
}

fn gef32_cases() -> Vec<TestCase<op::gef32::Routine>> {
    cases![]
}

fn gef64_cases() -> Vec<TestCase<op::gef64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn gef128_cases() -> Vec<TestCase<op::gef128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn ltf16_cases() -> Vec<TestCase<op::ltf16::Routine>> {
    cases![]
}

fn ltf32_cases() -> Vec<TestCase<op::ltf32::Routine>> {
    cases![]
}

fn ltf64_cases() -> Vec<TestCase<op::ltf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ltf128_cases() -> Vec<TestCase<op::ltf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn lef16_cases() -> Vec<TestCase<op::lef16::Routine>> {
    cases![]
}

fn lef32_cases() -> Vec<TestCase<op::lef32::Routine>> {
    cases![]
}

fn lef64_cases() -> Vec<TestCase<op::lef64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn lef128_cases() -> Vec<TestCase<op::lef128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn nef16_cases() -> Vec<TestCase<op::nef16::Routine>> {
    cases![]
}

fn nef32_cases() -> Vec<TestCase<op::nef32::Routine>> {
    cases![]
}

fn nef64_cases() -> Vec<TestCase<op::nef64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn nef128_cases() -> Vec<TestCase<op::nef128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn unordf16_cases() -> Vec<TestCase<op::unordf16::Routine>> {
    cases![]
}

fn unordf32_cases() -> Vec<TestCase<op::unordf32::Routine>> {
    cases![]
}

fn unordf64_cases() -> Vec<TestCase<op::unordf64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn unordf128_cases() -> Vec<TestCase<op::unordf128::Routine>> {
    cases![]
}

/* conversion */

#[cfg(f16_enabled)]
fn extend_f16_f32_cases() -> Vec<TestCase<op::extend_f16_f32::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn extend_f16_f64_cases() -> Vec<TestCase<op::extend_f16_f64::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
#[cfg(f128_enabled)]
fn extend_f16_f128_cases() -> Vec<TestCase<op::extend_f16_f128::Routine>> {
    cases![]
}

fn extend_f32_f64_cases() -> Vec<TestCase<op::extend_f32_f64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn extend_f32_f128_cases() -> Vec<TestCase<op::extend_f32_f128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn extend_f64_f128_cases() -> Vec<TestCase<op::extend_f64_f128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn narrow_f32_f16_cases() -> Vec<TestCase<op::narrow_f32_f16::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn narrow_f64_f16_cases() -> Vec<TestCase<op::narrow_f64_f16::Routine>> {
    cases![]
}

fn narrow_f64_f32_cases() -> Vec<TestCase<op::narrow_f64_f32::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
#[cfg(f128_enabled)]
fn narrow_f128_f16_cases() -> Vec<TestCase<op::narrow_f128_f16::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn narrow_f128_f32_cases() -> Vec<TestCase<op::narrow_f128_f32::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn narrow_f128_f64_cases() -> Vec<TestCase<op::narrow_f128_f64::Routine>> {
    cases![]
}

fn ftoi_f32_i32_cases() -> Vec<TestCase<op::ftoi_f32_i32::Routine>> {
    cases![]
}

fn ftoi_f32_i64_cases() -> Vec<TestCase<op::ftoi_f32_i64::Routine>> {
    cases![]
}

fn ftoi_f32_i128_cases() -> Vec<TestCase<op::ftoi_f32_i128::Routine>> {
    cases![]
}

fn ftoi_f64_i32_cases() -> Vec<TestCase<op::ftoi_f64_i32::Routine>> {
    cases![]
}

fn ftoi_f64_i64_cases() -> Vec<TestCase<op::ftoi_f64_i64::Routine>> {
    cases![]
}

fn ftoi_f64_i128_cases() -> Vec<TestCase<op::ftoi_f64_i128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ftoi_f128_i32_cases() -> Vec<TestCase<op::ftoi_f128_i32::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ftoi_f128_i64_cases() -> Vec<TestCase<op::ftoi_f128_i64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ftoi_f128_i128_cases() -> Vec<TestCase<op::ftoi_f128_i128::Routine>> {
    cases![]
}

fn ftoi_f32_u32_cases() -> Vec<TestCase<op::ftoi_f32_u32::Routine>> {
    cases![]
}

fn ftoi_f32_u64_cases() -> Vec<TestCase<op::ftoi_f32_u64::Routine>> {
    cases![]
}

fn ftoi_f32_u128_cases() -> Vec<TestCase<op::ftoi_f32_u128::Routine>> {
    cases![]
}

fn ftoi_f64_u32_cases() -> Vec<TestCase<op::ftoi_f64_u32::Routine>> {
    cases![]
}

fn ftoi_f64_u64_cases() -> Vec<TestCase<op::ftoi_f64_u64::Routine>> {
    cases![]
}

fn ftoi_f64_u128_cases() -> Vec<TestCase<op::ftoi_f64_u128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ftoi_f128_u32_cases() -> Vec<TestCase<op::ftoi_f128_u32::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ftoi_f128_u64_cases() -> Vec<TestCase<op::ftoi_f128_u64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ftoi_f128_u128_cases() -> Vec<TestCase<op::ftoi_f128_u128::Routine>> {
    cases![]
}

fn itof_i32_f32_cases() -> Vec<TestCase<op::itof_i32_f32::Routine>> {
    cases![]
}

fn itof_i64_f32_cases() -> Vec<TestCase<op::itof_i64_f32::Routine>> {
    cases![]
}

fn itof_i128_f32_cases() -> Vec<TestCase<op::itof_i128_f32::Routine>> {
    cases![]
}

fn itof_i32_f64_cases() -> Vec<TestCase<op::itof_i32_f64::Routine>> {
    cases![]
}

fn itof_i64_f64_cases() -> Vec<TestCase<op::itof_i64_f64::Routine>> {
    cases![]
}

fn itof_i128_f64_cases() -> Vec<TestCase<op::itof_i128_f64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn itof_i32_f128_cases() -> Vec<TestCase<op::itof_i32_f128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn itof_i64_f128_cases() -> Vec<TestCase<op::itof_i64_f128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn itof_i128_f128_cases() -> Vec<TestCase<op::itof_i128_f128::Routine>> {
    cases![]
}

fn itof_u32_f32_cases() -> Vec<TestCase<op::itof_u32_f32::Routine>> {
    cases![]
}

fn itof_u64_f32_cases() -> Vec<TestCase<op::itof_u64_f32::Routine>> {
    cases![]
}

fn itof_u128_f32_cases() -> Vec<TestCase<op::itof_u128_f32::Routine>> {
    cases![]
}

fn itof_u32_f64_cases() -> Vec<TestCase<op::itof_u32_f64::Routine>> {
    cases![]
}

fn itof_u64_f64_cases() -> Vec<TestCase<op::itof_u64_f64::Routine>> {
    cases![]
}

fn itof_u128_f64_cases() -> Vec<TestCase<op::itof_u128_f64::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn itof_u32_f128_cases() -> Vec<TestCase<op::itof_u32_f128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn itof_u64_f128_cases() -> Vec<TestCase<op::itof_u64_f128::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn itof_u128_f128_cases() -> Vec<TestCase<op::itof_u128_f128::Routine>> {
    cases![]
}

/* int arithmetic */

fn iadd_i128_cases() -> Vec<TestCase<op::iadd_i128::Routine>> {
    cases![]
}

fn iadd_u128_cases() -> Vec<TestCase<op::iadd_u128::Routine>> {
    cases![]
}

fn iaddo_i128_cases() -> Vec<TestCase<op::iaddo_i128::Routine>> {
    cases![]
}

fn iaddo_u128_cases() -> Vec<TestCase<op::iaddo_u128::Routine>> {
    cases![]
}

fn isub_i128_cases() -> Vec<TestCase<op::isub_i128::Routine>> {
    cases![]
}

fn isub_u128_cases() -> Vec<TestCase<op::isub_u128::Routine>> {
    cases![]
}

fn isubo_i128_cases() -> Vec<TestCase<op::isubo_i128::Routine>> {
    cases![]
}

fn isubo_u128_cases() -> Vec<TestCase<op::isubo_u128::Routine>> {
    cases![]
}

fn idiv_i128_cases() -> Vec<TestCase<op::idiv_i128::Routine>> {
    cases![]
}

fn idiv_i32_cases() -> Vec<TestCase<op::idiv_i32::Routine>> {
    cases![]
}

fn idiv_i64_cases() -> Vec<TestCase<op::idiv_i64::Routine>> {
    cases![]
}

fn idiv_u128_cases() -> Vec<TestCase<op::idiv_u128::Routine>> {
    cases![]
}

fn idiv_u32_cases() -> Vec<TestCase<op::idiv_u32::Routine>> {
    cases![]
}

fn idiv_u64_cases() -> Vec<TestCase<op::idiv_u64::Routine>> {
    cases![]
}

fn idivmod_i128_cases() -> Vec<TestCase<op::idivmod_i128::Routine>> {
    cases![]
}

fn idivmod_i32_cases() -> Vec<TestCase<op::idivmod_i32::Routine>> {
    cases![]
}

fn idivmod_i64_cases() -> Vec<TestCase<op::idivmod_i64::Routine>> {
    cases![]
}

fn idivmod_u128_cases() -> Vec<TestCase<op::idivmod_u128::Routine>> {
    cases![]
}

fn idivmod_u32_cases() -> Vec<TestCase<op::idivmod_u32::Routine>> {
    cases![]
}

fn idivmod_u64_cases() -> Vec<TestCase<op::idivmod_u64::Routine>> {
    cases![]
}

fn imod_i128_cases() -> Vec<TestCase<op::imod_i128::Routine>> {
    cases![]
}

fn imod_i32_cases() -> Vec<TestCase<op::imod_i32::Routine>> {
    cases![]
}

fn imod_i64_cases() -> Vec<TestCase<op::imod_i64::Routine>> {
    cases![]
}

fn imod_u128_cases() -> Vec<TestCase<op::imod_u128::Routine>> {
    cases![]
}

fn imod_u32_cases() -> Vec<TestCase<op::imod_u32::Routine>> {
    cases![]
}

fn imod_u64_cases() -> Vec<TestCase<op::imod_u64::Routine>> {
    cases![]
}

fn imul_i128_cases() -> Vec<TestCase<op::imul_i128::Routine>> {
    cases![]
}

fn imul_u64_cases() -> Vec<TestCase<op::imul_u64::Routine>> {
    cases![]
}

fn imulo_i128_cases() -> Vec<TestCase<op::imulo_i128::Routine>> {
    cases![]
}

fn imulo_i32_cases() -> Vec<TestCase<op::imulo_i32::Routine>> {
    cases![]
}

fn imulo_i64_cases() -> Vec<TestCase<op::imulo_i64::Routine>> {
    cases![]
}

fn imulo_u128_cases() -> Vec<TestCase<op::imulo_u128::Routine>> {
    cases![]
}

/* int shifts */

fn ashl_u32_cases() -> Vec<TestCase<op::ashl_u32::Routine>> {
    cases![]
}

fn ashl_u64_cases() -> Vec<TestCase<op::ashl_u64::Routine>> {
    cases![]
}

fn ashl_u128_cases() -> Vec<TestCase<op::ashl_u128::Routine>> {
    cases![]
}

fn ashr_i32_cases() -> Vec<TestCase<op::ashr_i32::Routine>> {
    cases![]
}

fn ashr_i64_cases() -> Vec<TestCase<op::ashr_i64::Routine>> {
    cases![]
}

fn ashr_i128_cases() -> Vec<TestCase<op::ashr_i128::Routine>> {
    cases![]
}

fn lshr_u32_cases() -> Vec<TestCase<op::lshr_u32::Routine>> {
    cases![]
}

fn lshr_u64_cases() -> Vec<TestCase<op::lshr_u64::Routine>> {
    cases![]
}

fn lshr_u128_cases() -> Vec<TestCase<op::lshr_u128::Routine>> {
    cases![]
}

/* int bitwise ops */

fn leading_zeros_u32_cases() -> Vec<TestCase<op::leading_zeros_u32::Routine>> {
    cases![]
}

fn leading_zeros_u64_cases() -> Vec<TestCase<op::leading_zeros_u64::Routine>> {
    cases![]
}

fn leading_zeros_u128_cases() -> Vec<TestCase<op::leading_zeros_u128::Routine>> {
    cases![]
}

fn trailing_zeros_u32_cases() -> Vec<TestCase<op::trailing_zeros_u32::Routine>> {
    cases![]
}

fn trailing_zeros_u64_cases() -> Vec<TestCase<op::trailing_zeros_u64::Routine>> {
    cases![]
}

fn trailing_zeros_u128_cases() -> Vec<TestCase<op::trailing_zeros_u128::Routine>> {
    cases![]
}

/*******************
 * libm test cases *
 *******************/

fn acos_cases() -> Vec<TestCase<op::acos::Routine>> {
    cases![]
}

fn acosf_cases() -> Vec<TestCase<op::acosf::Routine>> {
    cases![]
}

fn acosh_cases() -> Vec<TestCase<op::acosh::Routine>> {
    cases![]
}

fn acoshf_cases() -> Vec<TestCase<op::acoshf::Routine>> {
    cases![]
}

fn asin_cases() -> Vec<TestCase<op::asin::Routine>> {
    cases![]
}

fn asinf_cases() -> Vec<TestCase<op::asinf::Routine>> {
    cases![]
}

fn asinhf_cases() -> Vec<TestCase<op::asinhf::Routine>> {
    cases![
        // Failure on i586
        ((-0.37330312), -0.3651353),
        ((-0.421092), -0.40954682),
    ]
}

fn asinh_cases() -> Vec<TestCase<op::asinh::Routine>> {
    cases![]
}

fn atan_cases() -> Vec<TestCase<op::atan::Routine>> {
    cases![]
}

fn atan2_cases() -> Vec<TestCase<op::atan2::Routine>> {
    cases![]
}

fn atan2f_cases() -> Vec<TestCase<op::atan2f::Routine>> {
    cases![]
}

fn atanf_cases() -> Vec<TestCase<op::atanf::Routine>> {
    cases![]
}

fn atanh_cases() -> Vec<TestCase<op::atanh::Routine>> {
    cases![]
}

fn atanhf_cases() -> Vec<TestCase<op::atanhf::Routine>> {
    cases![]
}

fn cbrt_cases() -> Vec<TestCase<op::cbrt::Routine>> {
    cases![]
}

fn cbrtf_cases() -> Vec<TestCase<op::cbrtf::Routine>> {
    cases![]
}

fn ceil_cases() -> Vec<TestCase<op::ceil::Routine>> {
    cases![]
}

fn ceilf_cases() -> Vec<TestCase<op::ceilf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ceilf128_cases() -> Vec<TestCase<op::ceilf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn ceilf16_cases() -> Vec<TestCase<op::ceilf16::Routine>> {
    cases![]
}

fn copysign_cases() -> Vec<TestCase<op::copysign::Routine>> {
    cases![]
}

fn copysignf_cases() -> Vec<TestCase<op::copysignf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn copysignf128_cases() -> Vec<TestCase<op::copysignf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn copysignf16_cases() -> Vec<TestCase<op::copysignf16::Routine>> {
    cases![]
}

fn cos_cases() -> Vec<TestCase<op::cos::Routine>> {
    cases![]
}

fn cosf_cases() -> Vec<TestCase<op::cosf::Routine>> {
    cases![]
}

fn cosh_cases() -> Vec<TestCase<op::cosh::Routine>> {
    cases![]
}

fn coshf_cases() -> Vec<TestCase<op::coshf::Routine>> {
    cases![]
}

fn erf_cases() -> Vec<TestCase<op::erf::Routine>> {
    cases![]
}

fn erfc_cases() -> Vec<TestCase<op::erfc::Routine>> {
    cases![]
}

fn erfcf_cases() -> Vec<TestCase<op::erfcf::Routine>> {
    cases![]
}

fn erff_cases() -> Vec<TestCase<op::erff::Routine>> {
    cases![]
}

fn exp_cases() -> Vec<TestCase<op::exp::Routine>> {
    cases![]
}

fn exp10_cases() -> Vec<TestCase<op::exp10::Routine>> {
    cases![]
}

fn exp10f_cases() -> Vec<TestCase<op::exp10f::Routine>> {
    cases![]
}

fn exp2_cases() -> Vec<TestCase<op::exp2::Routine>> {
    cases![]
}

fn exp2f_cases() -> Vec<TestCase<op::exp2f::Routine>> {
    cases![]
}

fn expf_cases() -> Vec<TestCase<op::expf::Routine>> {
    cases![
        ((hf32!("-0x1.2d245ap-8")), hf32!("0x1.fda718p-1")),
        ((hf32!("0x1.db1b7ap-7")), hf32!("0x1.03bd22p+0")),
        ((hf32!("-0x1.dc15fcp+5")), hf32!("0x1.1ae6e6p-86")),
    ]
}

fn expm1_cases() -> Vec<TestCase<op::expm1::Routine>> {
    cases![]
}

fn expm1f_cases() -> Vec<TestCase<op::expm1f::Routine>> {
    cases![]
}

fn fabs_cases() -> Vec<TestCase<op::fabs::Routine>> {
    cases![]
}

fn fabsf_cases() -> Vec<TestCase<op::fabsf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fabsf128_cases() -> Vec<TestCase<op::fabsf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fabsf16_cases() -> Vec<TestCase<op::fabsf16::Routine>> {
    cases![]
}

fn fdim_cases() -> Vec<TestCase<op::fdim::Routine>> {
    cases![
        // Failures on i586
        (
            (
                hf64!("0x1.10d2f8a8dffd1p+355"),
                hf64!("-0x1.5203b17e54a8cp+373")
            ),
            hf64!("0x1.5203f5b312d2fp+373")
        ),
        (
            (
                hf64!("0x1.9ffdf64f0d2f8p+294"),
                hf64!("-0x1.71addd21280b5p+344")
            ),
            hf64!("0x1.71addd21280bbp+344")
        ),
        (
            (
                hf64!("0x1.f3600eb4ad0e0p-953"),
                hf64!("-0x1.0c29b2b40023dp-976")
            ),
            hf64!("0x1.f36010cd00737p-953")
        ),
    ]
}

fn fdimf_cases() -> Vec<TestCase<op::fdimf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fdimf128_cases() -> Vec<TestCase<op::fdimf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fdimf16_cases() -> Vec<TestCase<op::fdimf16::Routine>> {
    cases![]
}

fn floor_cases() -> Vec<TestCase<op::floor::Routine>> {
    cases![]
}

fn floorf_cases() -> Vec<TestCase<op::floorf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn floorf128_cases() -> Vec<TestCase<op::floorf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn floorf16_cases() -> Vec<TestCase<op::floorf16::Routine>> {
    cases![]
}

fn fmaf_cases() -> Vec<TestCase<op::fmaf::Routine>> {
    cases![
        // Known rounding error for some implementations (notably MinGW)
        ((-1.9369631e13f32, 2.1513551e-7, -1.7354427e-24), -4167095.8),
        // Failure on i586
        (
            (
                hf32!("-0x1.c92494p+109"),
                hf32!("-0x0.000018p-126"),
                hf32!("-0x1.6db6f0p-91"),
            ),
            hf32!("0x1.56db6ep-36")
        ),
    ]
}

fn fma_cases() -> Vec<TestCase<op::fma::Routine>> {
    cases![
        // Previous failure with incorrect sign
        ((5e-324, -5e-324, 0.0), -0.0),
        // Failure on i586
        (
            (0.999999999999999, 1.0000000000000013, 0.0),
            1.0000000000000002
        ),
        // Failure on musl i686/i586
        (
            (
                hf64!("0x0.0000000100001p-1022"),
                hf64!("0x1.ffffffffffffbp+1023"),
                hf64!("0x0p+0")
            ),
            hf64!("0x1.00000fffffffdp-30")
        )
    ]
}

#[cfg(f128_enabled)]
fn fmaf128_cases() -> Vec<TestCase<op::fmaf128::Routine>> {
    cases![
        (
            // Tricky rounding case that previously failed in extensive tests
            (
                hf128!("-0x1.1966cc01966cc01966cc01966f06p-25"),
                hf128!("-0x1.669933fe69933fe69933fe6997c9p-16358"),
                hf128!("-0x0.000000000000000000000000048ap-16382"),
            ),
            hf128!("0x0.c5171470a3ff5e0f68d751491b18p-16382")
        ),
        (
            // Subnormal edge case that caused a failure
            (
                hf128!("0x0.7ffffffffffffffffffffffffff7p-16382"),
                hf128!("0x1.ffffffffffffffffffffffffffffp-1"),
                hf128!("0x0.8000000000000000000000000009p-16382"),
            ),
            hf128!("0x1.0000000000000000000000000000p-16382")
        ),
    ]
}

#[cfg(f16_enabled)]
fn fmaxf16_cases() -> Vec<TestCase<op::fmaxf16::Routine>> {
    cases![]
}

fn fmaxf_cases() -> Vec<TestCase<op::fmaxf::Routine>> {
    cases![]
}

fn fmax_cases() -> Vec<TestCase<op::fmax::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fmaxf128_cases() -> Vec<TestCase<op::fmaxf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fmaximumf16_cases() -> Vec<TestCase<op::fmaximumf16::Routine>> {
    cases![]
}

fn fmaximumf_cases() -> Vec<TestCase<op::fmaximumf::Routine>> {
    cases![]
}

fn fmaximum_cases() -> Vec<TestCase<op::fmaximum::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fmaximumf128_cases() -> Vec<TestCase<op::fmaximumf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fmaximum_numf16_cases() -> Vec<TestCase<op::fmaximum_numf16::Routine>> {
    cases![]
}

fn fmaximum_numf_cases() -> Vec<TestCase<op::fmaximum_numf::Routine>> {
    cases![]
}

fn fmaximum_num_cases() -> Vec<TestCase<op::fmaximum_num::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fmaximum_numf128_cases() -> Vec<TestCase<op::fmaximum_numf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fminf16_cases() -> Vec<TestCase<op::fminf16::Routine>> {
    cases![]
}

fn fminf_cases() -> Vec<TestCase<op::fminf::Routine>> {
    cases![]
}

fn fmin_cases() -> Vec<TestCase<op::fmin::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fminf128_cases() -> Vec<TestCase<op::fminf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fminimumf16_cases() -> Vec<TestCase<op::fminimumf16::Routine>> {
    cases![]
}

fn fminimumf_cases() -> Vec<TestCase<op::fminimumf::Routine>> {
    cases![]
}

fn fminimum_cases() -> Vec<TestCase<op::fminimum::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fminimumf128_cases() -> Vec<TestCase<op::fminimumf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fminimum_numf16_cases() -> Vec<TestCase<op::fminimum_numf16::Routine>> {
    cases![]
}

fn fminimum_numf_cases() -> Vec<TestCase<op::fminimum_numf::Routine>> {
    cases![]
}

fn fminimum_num_cases() -> Vec<TestCase<op::fminimum_num::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn fminimum_numf128_cases() -> Vec<TestCase<op::fminimum_numf128::Routine>> {
    cases![]
}

fn fmod_cases() -> Vec<TestCase<op::fmod::Routine>> {
    cases![
        // Previous failure with incorrect loop iteration
        // <https://github.com/rust-lang/libm/pull/469#discussion_r2022337272>
        ((2.1, 3.123e-320), 2.0696e-320),
        ((2.1, 2.253547e-318), 1.772535e-318),
    ]
}

fn fmodf_cases() -> Vec<TestCase<op::fmodf::Routine>> {
    cases![
        // Previous failure with incorrect loop iteration
        // <https://github.com/rust-lang/libm/pull/469#discussion_r2022337272>
        ((2.1, 8.858e-42), 8.085e-42),
        ((2.1, 6.39164e-40), 6.1636e-40),
        ((5.5, 6.39164e-40), 4.77036e-40),
        ((-151.189, 6.39164e-40), -5.64734e-40),
    ]
}

#[cfg(f128_enabled)]
fn fmodf128_cases() -> Vec<TestCase<op::fmodf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn fmodf16_cases() -> Vec<TestCase<op::fmodf16::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn frexpf16_cases() -> Vec<TestCase<op::frexpf16::Routine>> {
    cases![]
}

fn frexpf_cases() -> Vec<TestCase<op::frexpf::Routine>> {
    cases![]
}

fn frexp_cases() -> Vec<TestCase<op::frexp::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn frexpf128_cases() -> Vec<TestCase<op::frexpf128::Routine>> {
    cases![]
}

fn hypot_cases() -> Vec<TestCase<op::hypot::Routine>> {
    cases![]
}

fn hypotf_cases() -> Vec<TestCase<op::hypotf::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn ilogbf16_cases() -> Vec<TestCase<op::ilogbf16::Routine>> {
    cases![]
}

fn ilogbf_cases() -> Vec<TestCase<op::ilogbf::Routine>> {
    cases![]
}

fn ilogb_cases() -> Vec<TestCase<op::ilogb::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ilogbf128_cases() -> Vec<TestCase<op::ilogbf128::Routine>> {
    cases![]
}

fn j0_cases() -> Vec<TestCase<op::j0::Routine>> {
    cases![]
}

fn j0f_cases() -> Vec<TestCase<op::j0f::Routine>> {
    cases![]
}

fn j1_cases() -> Vec<TestCase<op::j1::Routine>> {
    cases![]
}

fn j1f_cases() -> Vec<TestCase<op::j1f::Routine>> {
    cases![]
}

fn jnf_cases() -> Vec<TestCase<op::jnf::Routine>> {
    cases![]
}

fn jn_cases() -> Vec<TestCase<op::jn::Routine>> {
    cases![
        // Inputs that produce high errors
        ((190, 1005.366268038242), 7.328620335959289e-10),
        ((238, -311.0349), 7.270196433535006e-8),
    ]
}

fn ldexp_cases() -> Vec<TestCase<op::ldexp::Routine>> {
    cases![]
}

fn ldexpf_cases() -> Vec<TestCase<op::ldexpf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn ldexpf128_cases() -> Vec<TestCase<op::ldexpf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn ldexpf16_cases() -> Vec<TestCase<op::ldexpf16::Routine>> {
    cases![]
}

fn lgamma_cases() -> Vec<TestCase<op::lgamma::Routine>> {
    cases![]
}

fn lgammaf_cases() -> Vec<TestCase<op::lgammaf::Routine>> {
    cases![
        // High error
        ((-4.933393,), -1.9580022),
    ]
}

fn lgamma_r_cases() -> Vec<TestCase<op::lgamma_r::Routine>> {
    cases![]
}

fn lgammaf_r_cases() -> Vec<TestCase<op::lgammaf_r::Routine>> {
    cases![]
}

fn log_cases() -> Vec<TestCase<op::log::Routine>> {
    cases![]
}

fn log10_cases() -> Vec<TestCase<op::log10::Routine>> {
    cases![]
}

fn log10f_cases() -> Vec<TestCase<op::log10f::Routine>> {
    cases![]
}

fn log1pf_cases() -> Vec<TestCase<op::log1pf::Routine>> {
    cases![
        // Musl failures on i586
        ((hf32!("-0x1.8292f6p-2")), hf32!("-0x1.e56918p-2")),
        ((hf32!("0x1.12d15ep-1")), hf32!("0x1.b7fbf8p-2")),
        ((hf32!("-0x1.904ebep-2")), hf32!("-0x1.fbb6cap-2")),
    ]
}

fn log1p_cases() -> Vec<TestCase<op::log1p::Routine>> {
    cases![
        // Musl failure on i586
        (
            (hf64!("-0x1.9094dbf7f2e85p-2"),),
            hf64!("-0x1.fc29f046c88a1p-2")
        ),
    ]
}

fn log2_cases() -> Vec<TestCase<op::log2::Routine>> {
    cases![]
}

fn log2f_cases() -> Vec<TestCase<op::log2f::Routine>> {
    cases![]
}

fn logf_cases() -> Vec<TestCase<op::logf::Routine>> {
    cases![]
}

fn modf_cases() -> Vec<TestCase<op::modf::Routine>> {
    cases![]
}

fn modff_cases() -> Vec<TestCase<op::modff::Routine>> {
    cases![]
}

fn nextafter_cases() -> Vec<TestCase<op::nextafter::Routine>> {
    cases![]
}

fn nextafterf_cases() -> Vec<TestCase<op::nextafterf::Routine>> {
    cases![]
}

fn pow_cases() -> Vec<TestCase<op::pow::Routine>> {
    cases![]
}

fn powf_cases() -> Vec<TestCase<op::powf::Routine>> {
    cases![]
}

fn remainder_cases() -> Vec<TestCase<op::remainder::Routine>> {
    cases![]
}

fn remainderf_cases() -> Vec<TestCase<op::remainderf::Routine>> {
    cases![]
}

fn remquo_cases() -> Vec<TestCase<op::remquo::Routine>> {
    cases![]
}

fn remquof_cases() -> Vec<TestCase<op::remquof::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn rintf16_cases() -> Vec<TestCase<op::rintf16::Routine>> {
    // Out rint doesn't respect rounding modes so it is the same as roundeven
    roundevenf16_cases()
        .into_iter()
        .map(TestCase::cast)
        .collect()
}

fn rintf_cases() -> Vec<TestCase<op::rintf::Routine>> {
    // Out rint doesn't respect rounding modes so it is the same as roundeven
    roundevenf_cases().into_iter().map(TestCase::cast).collect()
}

fn rint_cases() -> Vec<TestCase<op::rint::Routine>> {
    // Out rint doesn't respect rounding modes so it is the same as roundeven
    roundeven_cases().into_iter().map(TestCase::cast).collect()
}

#[cfg(f128_enabled)]
fn rintf128_cases() -> Vec<TestCase<op::rintf128::Routine>> {
    // Out rint doesn't respect rounding modes so it is the same as roundeven
    roundevenf128_cases()
        .into_iter()
        .map(TestCase::cast)
        .collect()
}

#[cfg(f16_enabled)]
fn roundf16_cases() -> Vec<TestCase<op::roundf16::Routine>> {
    cases![]
}

fn round_cases() -> Vec<TestCase<op::round::Routine>> {
    cases![
        // Failure on i586
        (
            (hf64!("0x1.9efc6a203d4a9p+52"),),
            hf64!("0x1.9efc6a203d4a9p+52")
        )
    ]
}

fn roundf_cases() -> Vec<TestCase<op::roundf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn roundf128_cases() -> Vec<TestCase<op::roundf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn roundevenf16_cases() -> Vec<TestCase<op::roundevenf16::Routine>> {
    cases![]
}

fn roundevenf_cases() -> Vec<TestCase<op::roundevenf::Routine>> {
    cases![]
}

fn roundeven_cases() -> Vec<TestCase<op::roundeven::Routine>> {
    cases![
        // Failure on i586
        ((-519629176421.49976,), -519629176421.0),
        // Failures with a previous algorithm
        ((-849751480.5001163,), -849751481.0),
        ((-12493089.499809155,), -12493089.0),
        ((-1308.5000830345912,), -1309.0),
    ]
}

#[cfg(f128_enabled)]
fn roundevenf128_cases() -> Vec<TestCase<op::roundevenf128::Routine>> {
    cases![]
}

fn scalbn_cases() -> Vec<TestCase<op::scalbn::Routine>> {
    cases![]
}

fn scalbnf_cases() -> Vec<TestCase<op::scalbnf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn scalbnf128_cases() -> Vec<TestCase<op::scalbnf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn scalbnf16_cases() -> Vec<TestCase<op::scalbnf16::Routine>> {
    cases![]
}

fn sin_cases() -> Vec<TestCase<op::sin::Routine>> {
    cases![]
}

fn sincos_cases() -> Vec<TestCase<op::sincos::Routine>> {
    cases![]
}

fn sincosf_cases() -> Vec<TestCase<op::sincosf::Routine>> {
    cases![]
}

fn sinf_cases() -> Vec<TestCase<op::sinf::Routine>> {
    cases![]
}

fn sinh_cases() -> Vec<TestCase<op::sinh::Routine>> {
    cases![]
}

fn sinhf_cases() -> Vec<TestCase<op::sinhf::Routine>> {
    cases![]
}

fn sqrt_cases() -> Vec<TestCase<op::sqrt::Routine>> {
    cases![]
}

fn sqrtf_cases() -> Vec<TestCase<op::sqrtf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn sqrtf128_cases() -> Vec<TestCase<op::sqrtf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn sqrtf16_cases() -> Vec<TestCase<op::sqrtf16::Routine>> {
    cases![]
}

fn tanf_cases() -> Vec<TestCase<op::tanf::Routine>> {
    cases![]
}

fn tan_cases() -> Vec<TestCase<op::tan::Routine>> {
    cases![
        // Musl failures on i586
        (
            (hf64!("0x1.fffffffffffafp+1023"),),
            hf64!("0x1.c573c6dd8c00ap+0")
        ),
        (
            (hf64!("0x1.fffffffffffafp+1023"),),
            hf64!("0x1.c573c6dd8c00ap+0")
        ),
        (
            (hf64!("-0x1.0b10f6eaf2ca0p+883"),),
            hf64!("0x1.cefbd167e2402p+0")
        ),
    ]
}

fn tanh_cases() -> Vec<TestCase<op::tanh::Routine>> {
    cases![(
        (hf64!("0x1.fbfdb8b31b9b4p-3"),),
        hf64!("0x1.f1d2bcb4e1b45p-3")
    )]
}

fn tanhf_cases() -> Vec<TestCase<op::tanhf::Routine>> {
    cases![
        // Inaccuracy in musl
        ((0.24503659,), 0.24024734),
        ((0.19125812,), 0.18895969),
    ]
}

fn tgamma_cases() -> Vec<TestCase<op::tgamma::Routine>> {
    cases![]
}

fn tgammaf_cases() -> Vec<TestCase<op::tgammaf::Routine>> {
    cases![]
}

fn trunc_cases() -> Vec<TestCase<op::trunc::Routine>> {
    cases![]
}

fn truncf_cases() -> Vec<TestCase<op::truncf::Routine>> {
    cases![]
}

#[cfg(f128_enabled)]
fn truncf128_cases() -> Vec<TestCase<op::truncf128::Routine>> {
    cases![]
}

#[cfg(f16_enabled)]
fn truncf16_cases() -> Vec<TestCase<op::truncf16::Routine>> {
    cases![]
}

fn y0_cases() -> Vec<TestCase<op::y0::Routine>> {
    cases![]
}

fn y0f_cases() -> Vec<TestCase<op::y0f::Routine>> {
    cases![]
}

fn y1_cases() -> Vec<TestCase<op::y1::Routine>> {
    cases![]
}

fn y1f_cases() -> Vec<TestCase<op::y1f::Routine>> {
    cases![]
}

fn ynf_cases() -> Vec<TestCase<op::ynf::Routine>> {
    cases![]
}

fn yn_cases() -> Vec<TestCase<op::yn::Routine>> {
    cases![
        // Inputs that should be finite but tend to round to infinity
        ((228, 120.75621), -3.3293829e38),
        ((148, 61.379253), -3.2585946e38),
        ((184, 87.26689), -3.2943882e38),
    ]
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
    Op::get_cases()
        .into_iter()
        .filter_map(|x| x.output.map(|o| (x.input, o)))
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
    let count: u64 = cases
        .iter()
        .filter(|case| case.output.is_none())
        .count()
        .try_into()
        .unwrap();

    (
        cases
            .into_iter()
            .filter(|x| x.output.is_none())
            .map(|x| x.input),
        count,
    )
}
