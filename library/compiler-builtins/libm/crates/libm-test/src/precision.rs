//! Configuration for skipping or changing the result for individual test cases (inputs) rather
//! than ignoring entire tests.

use core::f32;

use CheckBasis::{Mpfr, Musl};
use libm::support::CastFrom;
use {BaseName as Bn, Identifier as Id};

use crate::{BaseName, CheckBasis, CheckCtx, Float, Identifier, Int, TestResult};

/// Type implementing [`IgnoreCase`].
pub struct SpecialCase;

/// ULP allowed to differ from the results returned by a test basis.
pub fn default_ulp(ctx: &CheckCtx) -> u32 {
    // ULP compared to the infinite (MPFR) result.
    let mut ulp = match ctx.base_name {
        // Operations that require exact results. This list should correlate with what we
        // have documented at <https://doc.rust-lang.org/std/primitive.f32.html>.
        Bn::Ceil
        | Bn::Copysign
        | Bn::Fabs
        | Bn::Fdim
        | Bn::Floor
        | Bn::Fma
        | Bn::Fmax
        | Bn::Fmin
        | Bn::Fmod
        | Bn::Frexp
        | Bn::Ilogb
        | Bn::Ldexp
        | Bn::Modf
        | Bn::Nextafter
        | Bn::Remainder
        | Bn::Remquo
        | Bn::Rint
        | Bn::Round
        | Bn::Scalbn
        | Bn::Sqrt
        | Bn::Trunc => 0,

        // Operations that aren't required to be exact, but our implementations are.
        Bn::Cbrt if ctx.fn_ident != Id::Cbrt => 0,

        // Bessel functions have large inaccuracies.
        Bn::J0 | Bn::J1 | Bn::Y0 | Bn::Y1 | Bn::Jn | Bn::Yn => 8_000_000,

        // For all other operations, specify our implementation's worst case precision.
        Bn::Acos => 1,
        Bn::Acosh => 4,
        Bn::Asin => 1,
        Bn::Asinh => 2,
        Bn::Atan => 1,
        Bn::Atan2 => 2,
        Bn::Atanh => 2,
        Bn::Cbrt => 1,
        Bn::Cos => 1,
        Bn::Cosh => 1,
        Bn::Erf => 1,
        Bn::Erfc => 4,
        Bn::Exp => 1,
        Bn::Exp10 => 6,
        Bn::Exp2 => 1,
        Bn::Expm1 => 1,
        Bn::Hypot => 1,
        Bn::Lgamma | Bn::LgammaR => 16,
        Bn::Log => 1,
        Bn::Log10 => 1,
        Bn::Log1p => 1,
        Bn::Log2 => 1,
        Bn::Pow => 1,
        Bn::Sin => 1,
        Bn::Sincos => 1,
        Bn::Sinh => 2,
        Bn::Tan => 1,
        Bn::Tanh => 2,
        // tgammaf has higher accuracy than tgamma.
        Bn::Tgamma if ctx.fn_ident != Id::Tgamma => 1,
        Bn::Tgamma => 20,
    };

    // There are some cases where musl's approximation is less accurate than ours. For these
    // cases, increase the ULP.
    if ctx.basis == Musl {
        match ctx.base_name {
            Bn::Cosh => ulp = 2,
            Bn::Exp10 if usize::BITS < 64 => ulp = 4,
            Bn::Lgamma | Bn::LgammaR => ulp = 400,
            Bn::Tanh => ulp = 4,
            _ => (),
        }

        match ctx.fn_ident {
            // FIXME(#401): musl has an incorrect result here.
            Id::Fdim => ulp = 2,
            Id::Sincosf => ulp = 500,
            Id::Tgamma => ulp = 20,
            _ => (),
        }
    }

    if cfg!(target_arch = "x86") {
        match ctx.fn_ident {
            // Input `fma(0.999999999999999, 1.0000000000000013, 0.0) = 1.0000000000000002` is
            // incorrect on i586 and i686.
            Id::Fma => ulp = 1,
            _ => (),
        }
    }

    // In some cases, our implementation is less accurate than musl on i586.
    if cfg!(x86_no_sse) {
        match ctx.fn_ident {
            // FIXME(#401): these need to be correctly rounded but are not.
            Id::Fmaf => ulp = 1,
            Id::Fdim => ulp = 1,
            Id::Round => ulp = 1,

            Id::Asinh => ulp = 3,
            Id::Asinhf => ulp = 3,
            Id::Exp10 | Id::Exp10f => ulp = 1_000_000,
            Id::Exp2 | Id::Exp2f => ulp = 10_000_000,
            Id::Log1p | Id::Log1pf => ulp = 2,
            Id::Tan => ulp = 2,
            _ => (),
        }
    }

    ulp
}

/// Result of checking for possible overrides.
#[derive(Debug, Default)]
pub enum CheckAction {
    /// The check should pass. Default case.
    #[default]
    AssertSuccess,

    /// Override the ULP for this check.
    AssertWithUlp(u32),

    /// Failure is expected, ensure this is the case (xfail). Takes a contxt string to help trace
    /// back exactly why we expect this to fail.
    AssertFailure(&'static str),

    /// The override somehow validated the result, here it is.
    Custom(TestResult),

    /// Disregard the output.
    Skip,
}

/// Don't run further validation on this test case.
const SKIP: CheckAction = CheckAction::Skip;

/// Return this to skip checks on a test that currently fails but shouldn't. Takes a description
/// of context.
const XFAIL: fn(&'static str) -> CheckAction = CheckAction::AssertFailure;

/// Indicates that we expect a test to fail but we aren't asserting that it does (e.g. some results
/// within a range do actually pass).
///
/// Same as `SKIP`, just indicates we have something to eventually fix.
const XFAIL_NOCHECK: CheckAction = CheckAction::Skip;

/// By default, all tests should pass.
const DEFAULT: CheckAction = CheckAction::AssertSuccess;

/// Allow overriding the outputs of specific test cases.
///
/// There are some cases where we want to xfail specific cases or handle certain inputs
/// differently than the rest of calls to `validate`. This provides a hook to do that.
///
/// If `None` is returned, checks will proceed as usual. If `Some(result)` is returned, checks
/// are skipped and the provided result is returned instead.
///
/// This gets implemented once per input type, then the functions provide further filtering
/// based on function name and values.
///
/// `ulp` can also be set to adjust the ULP for that specific test, even if `None` is still
/// returned.
pub trait MaybeOverride<Input> {
    fn check_float<F: Float>(
        _input: Input,
        _actual: F,
        _expected: F,
        _ctx: &CheckCtx,
    ) -> CheckAction {
        DEFAULT
    }

    fn check_int<I: Int>(_input: Input, _actual: I, _expected: I, _ctx: &CheckCtx) -> CheckAction {
        DEFAULT
    }
}

#[cfg(f16_enabled)]
impl MaybeOverride<(f16,)> for SpecialCase {}

impl MaybeOverride<(f32,)> for SpecialCase {
    fn check_float<F: Float>(input: (f32,), actual: F, expected: F, ctx: &CheckCtx) -> CheckAction {
        if ctx.base_name == BaseName::Expm1
            && !input.0.is_infinite()
            && input.0 > 80.0
            && actual.is_infinite()
            && !expected.is_infinite()
        {
            // we return infinity but the number is representable
            if ctx.basis == CheckBasis::Musl {
                return XFAIL_NOCHECK;
            }
            return XFAIL("expm1 representable numbers");
        }

        if cfg!(x86_no_sse)
            && ctx.base_name == BaseName::Exp2
            && !expected.is_infinite()
            && actual.is_infinite()
        {
            // We return infinity when there is a representable value. Test input: 127.97238
            return XFAIL("586 exp2 representable numbers");
        }

        if ctx.base_name == BaseName::Sinh && input.0.abs() > 80.0 && actual.is_nan() {
            // we return some NaN that should be real values or infinite
            if ctx.basis == CheckBasis::Musl {
                return XFAIL_NOCHECK;
            }
            return XFAIL("sinh unexpected NaN");
        }

        if (ctx.base_name == BaseName::Lgamma || ctx.base_name == BaseName::LgammaR)
            && input.0 > 4e36
            && expected.is_infinite()
            && !actual.is_infinite()
        {
            // This result should saturate but we return a finite value.
            return XFAIL_NOCHECK;
        }

        if ctx.base_name == BaseName::J0 && input.0 < -1e34 {
            // Errors get huge close to -inf
            return XFAIL_NOCHECK;
        }

        unop_common(input, actual, expected, ctx)
    }

    fn check_int<I: Int>(input: (f32,), actual: I, expected: I, ctx: &CheckCtx) -> CheckAction {
        // On MPFR for lgammaf_r, we set -1 as the integer result for negative infinity but MPFR
        // sets +1
        if ctx.basis == CheckBasis::Mpfr
            && ctx.base_name == BaseName::LgammaR
            && input.0 == f32::NEG_INFINITY
            && actual.abs() == expected.abs()
        {
            return XFAIL("lgammar integer result");
        }

        DEFAULT
    }
}

impl MaybeOverride<(f64,)> for SpecialCase {
    fn check_float<F: Float>(input: (f64,), actual: F, expected: F, ctx: &CheckCtx) -> CheckAction {
        if cfg!(x86_no_sse)
            && ctx.base_name == BaseName::Ceil
            && ctx.basis == CheckBasis::Musl
            && input.0 < 0.0
            && input.0 > -1.0
            && expected == F::ZERO
            && actual == F::ZERO
        {
            // musl returns -0.0, we return +0.0
            return XFAIL("i586 ceil signed zero");
        }

        if cfg!(x86_no_sse)
            && ctx.base_name == BaseName::Rint
            && (expected - actual).abs() <= F::ONE
            && (expected - actual).abs() > F::ZERO
        {
            // Our rounding mode is incorrect.
            return XFAIL("i586 rint rounding mode");
        }

        if cfg!(x86_no_sse)
            && (ctx.fn_ident == Identifier::Ceil || ctx.fn_ident == Identifier::Floor)
            && expected.eq_repr(F::NEG_ZERO)
            && actual.eq_repr(F::ZERO)
        {
            // FIXME: the x87 implementations do not keep the distinction between -0.0 and 0.0.
            // See https://github.com/rust-lang/libm/pull/404#issuecomment-2572399955
            return XFAIL("i586 ceil/floor signed zero");
        }

        if cfg!(x86_no_sse)
            && (ctx.fn_ident == Identifier::Exp10 || ctx.fn_ident == Identifier::Exp2)
        {
            // FIXME: i586 has very imprecise results with ULP > u32::MAX for these
            // operations so we can't reasonably provide a limit.
            return XFAIL_NOCHECK;
        }

        if ctx.base_name == BaseName::J0 && input.0 < -1e300 {
            // Errors get huge close to -inf
            return XFAIL_NOCHECK;
        }

        // maybe_check_nan_bits(actual, expected, ctx)
        unop_common(input, actual, expected, ctx)
    }

    fn check_int<I: Int>(input: (f64,), actual: I, expected: I, ctx: &CheckCtx) -> CheckAction {
        // On MPFR for lgamma_r, we set -1 as the integer result for negative infinity but MPFR
        // sets +1
        if ctx.basis == CheckBasis::Mpfr
            && ctx.base_name == BaseName::LgammaR
            && input.0 == f64::NEG_INFINITY
            && actual.abs() == expected.abs()
        {
            return XFAIL("lgammar integer result");
        }

        DEFAULT
    }
}

#[cfg(f128_enabled)]
impl MaybeOverride<(f128,)> for SpecialCase {}

// F1 and F2 are always the same type, this is just to please generics
fn unop_common<F1: Float, F2: Float>(
    input: (F1,),
    actual: F2,
    expected: F2,
    ctx: &CheckCtx,
) -> CheckAction {
    if ctx.base_name == BaseName::Acosh
        && input.0 < F1::NEG_ONE
        && !(expected.is_nan() && actual.is_nan())
    {
        // acoshf is undefined for x <= 1.0, but we return a random result at lower values.

        if ctx.basis == CheckBasis::Musl {
            return XFAIL_NOCHECK;
        }

        return XFAIL("acoshf undefined");
    }

    if (ctx.base_name == BaseName::Lgamma || ctx.base_name == BaseName::LgammaR)
        && input.0 < F1::ZERO
        && !input.0.is_infinite()
    {
        // loggamma should not be defined for x < 0, yet we both return results
        return XFAIL_NOCHECK;
    }

    // fabs and copysign must leave NaNs untouched.
    if ctx.base_name == BaseName::Fabs && input.0.is_nan() {
        // LLVM currently uses x87 instructions which quieten signalling NaNs to handle the i686
        // `extern "C"` `f32`/`f64` return ABI.
        // LLVM issue <https://github.com/llvm/llvm-project/issues/66803>
        // Rust issue <https://github.com/rust-lang/rust/issues/115567>
        if cfg!(target_arch = "x86") && ctx.basis == CheckBasis::Musl && actual.is_nan() {
            return XFAIL_NOCHECK;
        }

        // MPFR only has one NaN bitpattern; allow the default `.is_nan()` checks to validate.
        if ctx.basis == CheckBasis::Mpfr {
            return DEFAULT;
        }

        // abs and copysign require signaling NaNs to be propagated, so verify bit equality.
        if actual.to_bits() == expected.to_bits() {
            return CheckAction::Custom(Ok(()));
        } else {
            return CheckAction::Custom(Err(anyhow::anyhow!("NaNs have different bitpatterns")));
        }
    }

    DEFAULT
}

#[cfg(f16_enabled)]
impl MaybeOverride<(f16, f16)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f16, f16),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        binop_common(input, actual, expected, ctx)
    }
}

impl MaybeOverride<(f32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f32, f32),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        binop_common(input, actual, expected, ctx)
    }

    fn check_int<I: Int>(
        _input: (f32, f32),
        actual: I,
        expected: I,
        ctx: &CheckCtx,
    ) -> CheckAction {
        remquo_common(actual, expected, ctx)
    }
}

impl MaybeOverride<(f64, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f64, f64),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        binop_common(input, actual, expected, ctx)
    }

    fn check_int<I: Int>(
        _input: (f64, f64),
        actual: I,
        expected: I,
        ctx: &CheckCtx,
    ) -> CheckAction {
        remquo_common(actual, expected, ctx)
    }
}

#[cfg(f128_enabled)]
impl MaybeOverride<(f128, f128)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f128, f128),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        binop_common(input, actual, expected, ctx)
    }
}

// F1 and F2 are always the same type, this is just to please generics
fn binop_common<F1: Float, F2: Float>(
    input: (F1, F1),
    actual: F2,
    expected: F2,
    ctx: &CheckCtx,
) -> CheckAction {
    // MPFR only has one NaN bitpattern; allow the default `.is_nan()` checks to validate. Skip if
    // the first input (magnitude source) is NaN and the output is also a NaN, or if the second
    // input (sign source) is NaN.
    if ctx.basis == CheckBasis::Mpfr
        && ((input.0.is_nan() && actual.is_nan() && expected.is_nan()) || input.1.is_nan())
    {
        return SKIP;
    }

    /* FIXME(#439): our fmin and fmax do not compare signed zeros */

    if ctx.base_name == BaseName::Fmin
        && input.0.biteq(F1::NEG_ZERO)
        && input.1.biteq(F1::ZERO)
        && expected.biteq(F2::NEG_ZERO)
        && actual.biteq(F2::ZERO)
    {
        return XFAIL("fmin signed zeroes");
    }

    if ctx.base_name == BaseName::Fmax
        && input.0.biteq(F1::NEG_ZERO)
        && input.1.biteq(F1::ZERO)
        && expected.biteq(F2::ZERO)
        && actual.biteq(F2::NEG_ZERO)
    {
        return XFAIL("fmax signed zeroes");
    }

    // Musl propagates NaNs if one is provided as the input, but we return the other input.
    if (ctx.base_name == BaseName::Fmax || ctx.base_name == BaseName::Fmin)
        && ctx.basis == Musl
        && (input.0.is_nan() ^ input.1.is_nan())
        && expected.is_nan()
    {
        return XFAIL("fmax/fmin musl NaN");
    }

    DEFAULT
}

fn remquo_common<I: Int>(actual: I, expected: I, ctx: &CheckCtx) -> CheckAction {
    // FIXME: Our MPFR implementation disagrees with musl and may need to be updated.
    if ctx.basis == CheckBasis::Mpfr
        && ctx.base_name == BaseName::Remquo
        && expected == I::MIN
        && actual == I::ZERO
    {
        return XFAIL("remquo integer mismatch");
    }

    DEFAULT
}

impl MaybeOverride<(i32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (i32, f32),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        // `ynf(213, 109.15641) = -inf` with our library, should be finite.
        if ctx.basis == Mpfr
            && ctx.base_name == BaseName::Yn
            && input.0 > 200
            && !expected.is_infinite()
            && actual.is_infinite()
        {
            return XFAIL("ynf infinity mismatch");
        }

        int_float_common(input, actual, expected, ctx)
    }
}

impl MaybeOverride<(i32, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (i32, f64),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        int_float_common(input, actual, expected, ctx)
    }
}

fn int_float_common<F1: Float, F2: Float>(
    input: (i32, F1),
    actual: F2,
    expected: F2,
    ctx: &CheckCtx,
) -> CheckAction {
    if ctx.basis == Mpfr
        && (ctx.base_name == BaseName::Jn || ctx.base_name == BaseName::Yn)
        && input.1 == F1::NEG_INFINITY
        && actual == F2::ZERO
        && expected == F2::ZERO
    {
        return XFAIL("mpfr b");
    }

    // Our bessel functions blow up with large N values
    if ctx.basis == Musl && (ctx.base_name == BaseName::Jn || ctx.base_name == BaseName::Yn) {
        if input.0 > 4000 {
            return XFAIL_NOCHECK;
        } else if input.0 > 2000 {
            return CheckAction::AssertWithUlp(20_000);
        } else if input.0 > 1000 {
            return CheckAction::AssertWithUlp(4_000);
        }
    }

    // Values near infinity sometimes get cut off for us. `ynf(681, 509.90924) = -inf` but should
    // be -3.2161271e38.
    if ctx.basis == Musl
        && ctx.fn_ident == Identifier::Ynf
        && !expected.is_infinite()
        && actual.is_infinite()
        && (expected.abs().to_bits().abs_diff(actual.abs().to_bits())
            < F2::Int::cast_from(1_000_000u32))
    {
        return XFAIL_NOCHECK;
    }

    DEFAULT
}

impl MaybeOverride<(f32, i32)> for SpecialCase {}
impl MaybeOverride<(f64, i32)> for SpecialCase {}

impl MaybeOverride<(f32, f32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f32, f32, f32),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        ternop_common(input, actual, expected, ctx)
    }
}
impl MaybeOverride<(f64, f64, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f64, f64, f64),
        actual: F,
        expected: F,
        ctx: &CheckCtx,
    ) -> CheckAction {
        ternop_common(input, actual, expected, ctx)
    }
}

// F1 and F2 are always the same type, this is just to please generics
fn ternop_common<F1: Float, F2: Float>(
    input: (F1, F1, F1),
    actual: F2,
    expected: F2,
    ctx: &CheckCtx,
) -> CheckAction {
    // FIXME(fma): 754-2020 says "When the exact result of (a × b) + c is non-zero yet the result
    // of fusedMultiplyAdd is zero because of rounding, the zero result takes the sign of the
    // exact result". Our implementation returns the wrong sign:
    //     fma(5e-324, -5e-324, 0.0) = 0.0 (should be -0.0)
    if ctx.base_name == BaseName::Fma
        && (input.0.is_sign_negative() ^ input.1.is_sign_negative())
        && input.0 != F1::ZERO
        && input.1 != F1::ZERO
        && input.2.biteq(F1::ZERO)
        && expected.biteq(F2::NEG_ZERO)
        && actual.biteq(F2::ZERO)
    {
        return XFAIL("fma sign");
    }

    DEFAULT
}
