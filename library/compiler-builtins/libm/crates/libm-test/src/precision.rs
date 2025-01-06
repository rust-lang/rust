//! Configuration for skipping or changing the result for individual test cases (inputs) rather
//! than ignoring entire tests.

use core::f32;

use CheckBasis::{Mpfr, Musl};
use {BaseName as Bn, Identifier as Id};

use crate::{BaseName, CheckBasis, CheckCtx, Float, Identifier, Int, TestResult};

/// Type implementing [`IgnoreCase`].
pub struct SpecialCase;

/// ULP allowed to differ from the results returned by a test basis.
///
/// Note that these results were obtained using 400M rounds of random inputs, which
/// is not a value used by default.
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
        Bn::Tgamma if ctx.fn_ident != Id::Tgamma => 0,

        // Bessel functions have large inaccuracies.
        Bn::J0 | Bn::J1 | Bn::Y0 | Bn::Y1 | Bn::Jn | Bn::Yn => 8_000_000,

        // For all other operations, specify our implementation's worst case precision.
        Bn::Acos => 1,
        Bn::Acosh => 4,
        Bn::Asin => 1,
        Bn::Asinh => 2,
        Bn::Atan => 1,
        Bn::Atan2 => 1,
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

    // In some cases, our implementation is less accurate than musl on i586.
    if cfg!(x86_no_sse) {
        match ctx.fn_ident {
            Id::Asinh => ulp = 3,
            Id::Asinhf => ulp = 3,
            Id::Log1p | Id::Log1pf => ulp = 2,
            Id::Round => ulp = 1,
            Id::Tan => ulp = 2,
            _ => (),
        }
    }

    ulp
}

/// Don't run further validation on this test case.
const SKIP: Option<TestResult> = Some(Ok(()));

/// Return this to skip checks on a test that currently fails but shouldn't. Looks
/// the same as skip, but we keep them separate to better indicate purpose.
const XFAIL: Option<TestResult> = Some(Ok(()));

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
        _ulp: &mut u32,
        _ctx: &CheckCtx,
    ) -> Option<TestResult> {
        None
    }

    fn check_int<I: Int>(
        _input: Input,
        _actual: I,
        _expected: I,
        _ctx: &CheckCtx,
    ) -> Option<TestResult> {
        None
    }
}

impl MaybeOverride<(f32,)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f32,),
        actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        if ctx.base_name == BaseName::Expm1 && input.0 > 80.0 && actual.is_infinite() {
            // we return infinity but the number is representable
            return XFAIL;
        }

        if ctx.base_name == BaseName::Sinh && input.0.abs() > 80.0 && actual.is_nan() {
            // we return some NaN that should be real values or infinite
            return XFAIL;
        }

        if ctx.base_name == BaseName::Acosh && input.0 < -1.0 {
            // acoshf is undefined for x <= 1.0, but we return a random result at lower
            // values.
            return XFAIL;
        }

        if ctx.base_name == BaseName::Lgamma || ctx.base_name == BaseName::LgammaR && input.0 < 0.0
        {
            // loggamma should not be defined for x < 0, yet we both return results
            return XFAIL;
        }

        maybe_check_nan_bits(actual, expected, ctx)
    }

    fn check_int<I: Int>(
        input: (f32,),
        actual: I,
        expected: I,
        ctx: &CheckCtx,
    ) -> Option<anyhow::Result<()>> {
        // On MPFR for lgammaf_r, we set -1 as the integer result for negative infinity but MPFR
        // sets +1
        if ctx.basis == CheckBasis::Mpfr
            && ctx.base_name == BaseName::LgammaR
            && input.0 == f32::NEG_INFINITY
            && actual.abs() == expected.abs()
        {
            XFAIL
        } else {
            None
        }
    }
}

impl MaybeOverride<(f64,)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f64,),
        actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        if ctx.basis == CheckBasis::Musl {
            if cfg!(target_arch = "x86") && ctx.base_name == BaseName::Acosh && input.0 < 1.0 {
                // The function is undefined, both implementations return random results
                return SKIP;
            }

            if cfg!(x86_no_sse)
                && ctx.base_name == BaseName::Ceil
                && input.0 < 0.0
                && input.0 > -1.0
                && expected == F::ZERO
                && actual == F::ZERO
            {
                // musl returns -0.0, we return +0.0
                return XFAIL;
            }
        }

        if ctx.base_name == BaseName::Acosh && input.0 < 1.0 {
            // The function is undefined for the inputs, musl and our libm both return
            // random results.
            return XFAIL;
        }

        if ctx.base_name == BaseName::Lgamma || ctx.base_name == BaseName::LgammaR && input.0 < 0.0
        {
            // loggamma should not be defined for x < 0, yet we both return results
            return XFAIL;
        }

        maybe_check_nan_bits(actual, expected, ctx)
    }

    fn check_int<I: Int>(
        input: (f64,),
        actual: I,
        expected: I,
        ctx: &CheckCtx,
    ) -> Option<anyhow::Result<()>> {
        // On MPFR for lgamma_r, we set -1 as the integer result for negative infinity but MPFR
        // sets +1
        if ctx.basis == CheckBasis::Mpfr
            && ctx.base_name == BaseName::LgammaR
            && input.0 == f64::NEG_INFINITY
            && actual.abs() == expected.abs()
        {
            XFAIL
        } else {
            None
        }
    }
}

/// Check NaN bits if the function requires it
fn maybe_check_nan_bits<F: Float>(actual: F, expected: F, ctx: &CheckCtx) -> Option<TestResult> {
    if !(ctx.base_name == BaseName::Fabs || ctx.base_name == BaseName::Copysign) {
        return None;
    }

    // LLVM currently uses x87 instructions which quieten signalling NaNs to handle the i686
    // `extern "C"` `f32`/`f64` return ABI.
    // LLVM issue <https://github.com/llvm/llvm-project/issues/66803>
    // Rust issue <https://github.com/rust-lang/rust/issues/115567>
    if cfg!(target_arch = "x86") && ctx.basis == CheckBasis::Musl {
        return SKIP;
    }

    // MPFR only has one NaN bitpattern; allow the default `.is_nan()` checks to validate.
    if ctx.basis == CheckBasis::Mpfr {
        return SKIP;
    }

    // abs and copysign require signaling NaNs to be propagated, so verify bit equality.
    if actual.to_bits() == expected.to_bits() {
        SKIP
    } else {
        Some(Err(anyhow::anyhow!("NaNs have different bitpatterns")))
    }
}

impl MaybeOverride<(f32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f32, f32),
        _actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        maybe_skip_binop_nan(input, expected, ctx)
    }
}

impl MaybeOverride<(f64, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f64, f64),
        _actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        maybe_skip_binop_nan(input, expected, ctx)
    }
}

/// Musl propagates NaNs if one is provided as the input, but we return the other input.
// F1 and F2 are always the same type, this is just to please generics
fn maybe_skip_binop_nan<F1: Float, F2: Float>(
    input: (F1, F1),
    expected: F2,
    ctx: &CheckCtx,
) -> Option<TestResult> {
    match (&ctx.basis, ctx.base_name) {
        (Musl, BaseName::Fmin | BaseName::Fmax)
            if (input.0.is_nan() || input.1.is_nan()) && expected.is_nan() =>
        {
            XFAIL
        }

        (Mpfr, BaseName::Copysign) if input.1.is_nan() => SKIP,

        _ => None,
    }
}

impl MaybeOverride<(i32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (i32, f32),
        actual: F,
        expected: F,
        ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        match (&ctx.basis, ctx.base_name) {
            (Musl, _) => bessel_prec_dropoff(input, ulp, ctx),

            // We return +0.0, MPFR returns -0.0
            (Mpfr, BaseName::Jn | BaseName::Yn)
                if input.1 == f32::NEG_INFINITY && actual == F::ZERO && expected == F::ZERO =>
            {
                XFAIL
            }

            _ => None,
        }
    }
}
impl MaybeOverride<(i32, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (i32, f64),
        actual: F,
        expected: F,
        ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        match (&ctx.basis, ctx.base_name) {
            (Musl, _) => bessel_prec_dropoff(input, ulp, ctx),

            // We return +0.0, MPFR returns -0.0
            (Mpfr, BaseName::Jn | BaseName::Yn)
                if input.1 == f64::NEG_INFINITY && actual == F::ZERO && expected == F::ZERO =>
            {
                XFAIL
            }

            _ => None,
        }
    }
}

/// Our bessel functions blow up with large N values
fn bessel_prec_dropoff<F: Float>(
    input: (i32, F),
    ulp: &mut u32,
    ctx: &CheckCtx,
) -> Option<TestResult> {
    if ctx.base_name == BaseName::Jn || ctx.base_name == BaseName::Yn {
        if input.0 > 4000 {
            return XFAIL;
        } else if input.0 > 2000 {
            // *ulp = 20_000;
            *ulp = 20000;
        } else if input.0 > 1000 {
            *ulp = 4000;
        }
    }

    None
}

impl MaybeOverride<(f32, f32, f32)> for SpecialCase {}
impl MaybeOverride<(f64, f64, f64)> for SpecialCase {}
impl MaybeOverride<(f32, i32)> for SpecialCase {}
impl MaybeOverride<(f64, i32)> for SpecialCase {}
