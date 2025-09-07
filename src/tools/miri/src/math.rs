use std::ops::Neg;
use std::{f32, f64};

use rand::Rng as _;
use rustc_apfloat::Float as _;
use rustc_apfloat::ieee::{DoubleS, IeeeFloat, Semantics, SingleS};
use rustc_middle::ty::{self, FloatTy, ScalarInt};

use crate::*;

/// Disturbes a floating-point result by a relative error in the range (-2^scale, 2^scale).
pub(crate) fn apply_random_float_error<F: rustc_apfloat::Float>(
    ecx: &mut crate::MiriInterpCx<'_>,
    val: F,
    err_scale: i32,
) -> F {
    if !ecx.machine.float_nondet
        || matches!(ecx.machine.float_rounding_error, FloatRoundingErrorMode::None)
        // relative errors don't do anything to zeros... avoid messing up the sign
        || val.is_zero()
        // The logic below makes no sense if the input is already non-finite.
        || !val.is_finite()
    {
        return val;
    }
    let rng = ecx.machine.rng.get_mut();

    // Generate a random integer in the range [0, 2^PREC).
    // (When read as binary, the position of the first `1` determines the exponent,
    // and the remaining bits fill the mantissa. `PREC` is one plus the size of the mantissa,
    // so this all works out.)
    let r = F::from_u128(match ecx.machine.float_rounding_error {
        FloatRoundingErrorMode::Random => rng.random_range(0..(1 << F::PRECISION)),
        FloatRoundingErrorMode::Max => (1 << F::PRECISION) - 1, // force max error
        FloatRoundingErrorMode::None => unreachable!(),
    })
    .value;
    // Multiply this with 2^(scale - PREC). The result is between 0 and
    // 2^PREC * 2^(scale - PREC) = 2^scale.
    let err = r.scalbn(err_scale.strict_sub(F::PRECISION.try_into().unwrap()));
    // give it a random sign
    let err = if rng.random() { -err } else { err };
    // Compute `val*(1+err)`, distributed out as `val + val*err` to avoid the imprecise addition
    // error being amplified by multiplication.
    (val + (val * err).value).value
}

/// Applies an error of `[-N, +N]` ULP to the given value.
pub(crate) fn apply_random_float_error_ulp<F: rustc_apfloat::Float>(
    ecx: &mut crate::MiriInterpCx<'_>,
    val: F,
    max_error: u32,
) -> F {
    // We could try to be clever and reuse `apply_random_float_error`, but that is hard to get right
    // (see <https://github.com/rust-lang/miri/pull/4558#discussion_r2316838085> for why) so we
    // implement the logic directly instead.
    if !ecx.machine.float_nondet
        || matches!(ecx.machine.float_rounding_error, FloatRoundingErrorMode::None)
        // FIXME: also disturb zeros? That requires a lot more cases in `fixed_float_value`
        // and might make the std test suite quite unhappy.
        || val.is_zero()
        // The logic below makes no sense if the input is already non-finite.
        || !val.is_finite()
    {
        return val;
    }
    let rng = ecx.machine.rng.get_mut();

    let max_error = i64::from(max_error);
    let error = match ecx.machine.float_rounding_error {
        FloatRoundingErrorMode::Random => rng.random_range(-max_error..=max_error),
        FloatRoundingErrorMode::Max =>
            if rng.random() {
                max_error
            } else {
                -max_error
            },
        FloatRoundingErrorMode::None => unreachable!(),
    };
    // If upwards ULP and downwards ULP differ, we take the average.
    let ulp = (((val.next_up().value - val).value + (val - val.next_down().value).value).value
        / F::from_u128(2).value)
        .value;
    // Shift the value by N times the ULP
    (val + (ulp * F::from_i128(error.into()).value).value).value
}

/// Applies an error of `[-N, +N]` ULP to the given value.
/// Will fail if `val` is not a floating point number.
pub(crate) fn apply_random_float_error_to_imm<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    val: ImmTy<'tcx>,
    max_error: u32,
) -> InterpResult<'tcx, ImmTy<'tcx>> {
    let scalar = val.to_scalar_int()?;
    let res: ScalarInt = match val.layout.ty.kind() {
        ty::Float(FloatTy::F16) =>
            apply_random_float_error_ulp(ecx, scalar.to_f16(), max_error).into(),
        ty::Float(FloatTy::F32) =>
            apply_random_float_error_ulp(ecx, scalar.to_f32(), max_error).into(),
        ty::Float(FloatTy::F64) =>
            apply_random_float_error_ulp(ecx, scalar.to_f64(), max_error).into(),
        ty::Float(FloatTy::F128) =>
            apply_random_float_error_ulp(ecx, scalar.to_f128(), max_error).into(),
        _ => bug!("intrinsic called with non-float input type"),
    };

    interp_ok(ImmTy::from_scalar_int(res, val.layout))
}

/// Given a floating-point operation and a floating-point value, clamps the result to the output
/// range of the given operation according to the C standard, if any.
pub(crate) fn clamp_float_value<S: Semantics>(
    intrinsic_name: &str,
    val: IeeeFloat<S>,
) -> IeeeFloat<S>
where
    IeeeFloat<S>: IeeeExt,
{
    let zero = IeeeFloat::<S>::ZERO;
    let one = IeeeFloat::<S>::one();
    let two = IeeeFloat::<S>::two();
    let pi = IeeeFloat::<S>::pi();
    let pi_over_2 = (pi / two).value;

    match intrinsic_name {
        // sin, cos, tanh: [-1, 1]
        #[rustfmt::skip]
        | "sinf32"
        | "sinf64"
        | "cosf32"
        | "cosf64"
        | "tanhf"
        | "tanh"
         => val.clamp(one.neg(), one),

        // exp: [0, +INF)
        "expf32" | "exp2f32" | "expf64" | "exp2f64" => val.maximum(zero),

        // cosh: [1, +INF)
        "coshf" | "cosh" => val.maximum(one),

        // acos: [0, π]
        "acosf" | "acos" => val.clamp(zero, pi),

        // asin: [-π, +π]
        "asinf" | "asin" => val.clamp(pi.neg(), pi),

        // atan: (-π/2, +π/2)
        "atanf" | "atan" => val.clamp(pi_over_2.neg(), pi_over_2),

        // erfc: (-1, 1)
        "erff" | "erf" => val.clamp(one.neg(), one),

        // erfc: (0, 2)
        "erfcf" | "erfc" => val.clamp(zero, two),

        // atan2(y, x): arctan(y/x) in [−π, +π]
        "atan2f" | "atan2" => val.clamp(pi.neg(), pi),

        _ => val,
    }
}

/// For the intrinsics:
/// - sinf32, sinf64, sinhf, sinh
/// - cosf32, cosf64, coshf, cosh
/// - tanhf, tanh, atanf, atan, atan2f, atan2
/// - expf32, expf64, exp2f32, exp2f64
/// - logf32, logf64, log2f32, log2f64, log10f32, log10f64
/// - powf32, powf64
/// - erff, erf, erfcf, erfc
/// - hypotf, hypot
///
/// # Return
///
/// Returns `Some(output)` if the `intrinsic` results in a defined fixed `output` specified in the C standard
/// (specifically, C23 annex F.10)  when given `args` as arguments. Outputs that are unaffected by a relative error
/// (such as INF and zero) are not handled here, they are assumed to be handled by the underlying
/// implementation. Returns `None` if no specific value is guaranteed.
///
/// # Note
///
/// For `powf*` operations of the form:
///
/// - `(SNaN)^(±0)`
/// - `1^(SNaN)`
///
/// The result is implementation-defined:
/// - musl returns for both `1.0`
/// - glibc returns for both `NaN`
///
/// This discrepancy exists because SNaN handling is not consistently defined across platforms,
/// and the C standard leaves behavior for SNaNs unspecified.
///
/// Miri chooses to adhere to both implementations and returns either one of them non-deterministically.
pub(crate) fn fixed_float_value<S: Semantics>(
    ecx: &mut MiriInterpCx<'_>,
    intrinsic_name: &str,
    args: &[IeeeFloat<S>],
) -> Option<IeeeFloat<S>>
where
    IeeeFloat<S>: IeeeExt,
{
    let this = ecx.eval_context_mut();
    let one = IeeeFloat::<S>::one();
    let two = IeeeFloat::<S>::two();
    let three = IeeeFloat::<S>::three();
    let pi = IeeeFloat::<S>::pi();
    let pi_over_2 = (pi / two).value;
    let pi_over_4 = (pi_over_2 / two).value;

    Some(match (intrinsic_name, args) {
        // cos(±0) and cosh(±0)= 1
        ("cosf32" | "cosf64" | "coshf" | "cosh", [input]) if input.is_zero() => one,

        // e^0 = 1
        ("expf32" | "expf64" | "exp2f32" | "exp2f64", [input]) if input.is_zero() => one,

        // tanh(±INF) = ±1
        ("tanhf" | "tanh", [input]) if input.is_infinite() => one.copy_sign(*input),

        // atan(±INF) = ±π/2
        ("atanf" | "atan", [input]) if input.is_infinite() => pi_over_2.copy_sign(*input),

        // erf(±INF) = ±1
        ("erff" | "erf", [input]) if input.is_infinite() => one.copy_sign(*input),

        // erfc(-INF) = 2
        ("erfcf" | "erfc", [input]) if input.is_neg_infinity() => (one + one).value,

        // hypot(x, ±0) = abs(x), if x is not a NaN.
        ("_hypotf" | "hypotf" | "_hypot" | "hypot", [x, y]) if !x.is_nan() && y.is_zero() =>
            x.abs(),

        // atan2(±0,−0) = ±π.
        // atan2(±0, y) = ±π for y < 0.
        // Must check for non NaN because `y.is_negative()` also applies to NaN.
        ("atan2f" | "atan2", [x, y]) if (x.is_zero() && (y.is_negative() && !y.is_nan())) =>
            pi.copy_sign(*x),

        // atan2(±x,−∞) = ±π for finite x > 0.
        ("atan2f" | "atan2", [x, y])
            if (!x.is_zero() && !x.is_infinite()) && y.is_neg_infinity() =>
            pi.copy_sign(*x),

        // atan2(x, ±0) = −π/2 for x < 0.
        // atan2(x, ±0) =  π/2 for x > 0.
        ("atan2f" | "atan2", [x, y]) if !x.is_zero() && y.is_zero() => pi_over_2.copy_sign(*x),

        //atan2(±∞, −∞) = ±3π/4
        ("atan2f" | "atan2", [x, y]) if x.is_infinite() && y.is_neg_infinity() =>
            (pi_over_4 * three).value.copy_sign(*x),

        //atan2(±∞, +∞) = ±π/4
        ("atan2f" | "atan2", [x, y]) if x.is_infinite() && y.is_pos_infinity() =>
            pi_over_4.copy_sign(*x),

        // atan2(±∞, y) returns ±π/2 for finite y.
        ("atan2f" | "atan2", [x, y]) if x.is_infinite() && (!y.is_infinite() && !y.is_nan()) =>
            pi_over_2.copy_sign(*x),

        // (-1)^(±INF) = 1
        ("powf32" | "powf64", [base, exp]) if *base == -one && exp.is_infinite() => one,

        // 1^y = 1 for any y, even a NaN
        ("powf32" | "powf64", [base, exp]) if *base == one => {
            let rng = this.machine.rng.get_mut();
            // SNaN exponents get special treatment: they might return 1, or a NaN.
            let return_nan = exp.is_signaling() && this.machine.float_nondet && rng.random();
            // Handle both the musl and glibc cases non-deterministically.
            if return_nan { this.generate_nan(args) } else { one }
        }

        // x^(±0) = 1 for any x, even a NaN
        ("powf32" | "powf64", [base, exp]) if exp.is_zero() => {
            let rng = this.machine.rng.get_mut();
            // SNaN bases get special treatment: they might return 1, or a NaN.
            let return_nan = base.is_signaling() && this.machine.float_nondet && rng.random();
            // Handle both the musl and glibc cases non-deterministically.
            if return_nan { this.generate_nan(args) } else { one }
        }

        // There are a lot of cases for fixed outputs according to the C Standard, but these are
        // mainly INF or zero which are not affected by the applied error.
        _ => return None,
    })
}

/// Returns `Some(output)` if `powi` (called `pown` in C) results in a fixed value specified in the
/// C standard (specifically, C23 annex F.10.4.6) when doing `base^exp`. Otherwise, returns `None`.
pub(crate) fn fixed_powi_value<S: Semantics>(
    ecx: &mut MiriInterpCx<'_>,
    base: IeeeFloat<S>,
    exp: i32,
) -> Option<IeeeFloat<S>>
where
    IeeeFloat<S>: IeeeExt,
{
    match exp {
        0 => {
            let one = IeeeFloat::<S>::one();
            let rng = ecx.machine.rng.get_mut();
            let return_nan = ecx.machine.float_nondet && rng.random() && base.is_signaling();
            // For SNaN treatment, we are consistent with `powf`above.
            // (We wouldn't have two, unlike powf all implementations seem to agree for powi,
            // but for now we are maximally conservative.)
            Some(if return_nan { ecx.generate_nan(&[base]) } else { one })
        }

        _ => return None,
    }
}

pub(crate) fn sqrt<S: rustc_apfloat::ieee::Semantics>(x: IeeeFloat<S>) -> IeeeFloat<S> {
    match x.category() {
        // preserve zero sign
        rustc_apfloat::Category::Zero => x,
        // propagate NaN
        rustc_apfloat::Category::NaN => x,
        // sqrt of negative number is NaN
        _ if x.is_negative() => IeeeFloat::NAN,
        // sqrt(∞) = ∞
        rustc_apfloat::Category::Infinity => IeeeFloat::INFINITY,
        rustc_apfloat::Category::Normal => {
            // Floating point precision, excluding the integer bit
            let prec = i32::try_from(S::PRECISION).unwrap() - 1;

            // x = 2^(exp - prec) * mant
            // where mant is an integer with prec+1 bits
            // mant is a u128, which should be large enough for the largest prec (112 for f128)
            let mut exp = x.ilogb();
            let mut mant = x.scalbn(prec - exp).to_u128(128).value;

            if exp % 2 != 0 {
                // Make exponent even, so it can be divided by 2
                exp -= 1;
                mant <<= 1;
            }

            // Bit-by-bit (base-2 digit-by-digit) sqrt of mant.
            // mant is treated here as a fixed point number with prec fractional bits.
            // mant will be shifted left by one bit to have an extra fractional bit, which
            // will be used to determine the rounding direction.

            // res is the truncated sqrt of mant, where one bit is added at each iteration.
            let mut res = 0u128;
            // rem is the remainder with the current res
            // rem_i = 2^i * ((mant<<1) - res_i^2)
            // starting with res = 0, rem = mant<<1
            let mut rem = mant << 1;
            // s_i = 2*res_i
            let mut s = 0u128;
            // d is used to iterate over bits, from high to low (d_i = 2^(-i))
            let mut d = 1u128 << (prec + 1);

            // For iteration j=i+1, we need to find largest b_j = 0 or 1 such that
            //  (res_i + b_j * 2^(-j))^2 <= mant<<1
            // Expanding (a + b)^2 = a^2 + b^2 + 2*a*b:
            //  res_i^2 + (b_j * 2^(-j))^2 + 2 * res_i * b_j * 2^(-j) <= mant<<1
            // And rearranging the terms:
            //  b_j^2 * 2^(-j) + 2 * res_i * b_j <= 2^j * (mant<<1 - res_i^2)
            //  b_j^2 * 2^(-j) + 2 * res_i * b_j <= rem_i

            while d != 0 {
                // Probe b_j^2 * 2^(-j) + 2 * res_i * b_j <= rem_i with b_j = 1:
                // t = 2*res_i + 2^(-j)
                let t = s + d;
                if rem >= t {
                    // b_j should be 1, so make res_j = res_i + 2^(-j) and adjust rem
                    res += d;
                    s += d + d;
                    rem -= t;
                }
                // Adjust rem for next iteration
                rem <<= 1;
                // Shift iterator
                d >>= 1;
            }

            // Remove extra fractional bit from result, rounding to nearest.
            // If the last bit is 0, then the nearest neighbor is definitely the lower one.
            // If the last bit is 1, it sounds like this may either be a tie (if there's
            // infinitely many 0s after this 1), or the nearest neighbor is the upper one.
            // However, since square roots are either exact or irrational, and an exact root
            // would lead to the last "extra" bit being 0, we can exclude a tie in this case.
            // We therefore always round up if the last bit is 1. When the last bit is 0,
            // adding 1 will not do anything since the shift will discard it.
            res = (res + 1) >> 1;

            // Build resulting value with res as mantissa and exp/2 as exponent
            IeeeFloat::from_u128(res).value.scalbn(exp / 2 - prec)
        }
    }
}

/// Extend functionality of `rustc_apfloat` softfloats for IEEE float types.
pub trait IeeeExt: rustc_apfloat::Float {
    // Some values we use:

    #[inline]
    fn one() -> Self {
        Self::from_u128(1).value
    }

    #[inline]
    fn two() -> Self {
        Self::from_u128(2).value
    }

    #[inline]
    fn three() -> Self {
        Self::from_u128(3).value
    }

    fn pi() -> Self;

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.maximum(min).minimum(max)
    }
}

macro_rules! impl_ieee_pi {
    ($float_ty:ident, $semantic:ty) => {
        impl IeeeExt for IeeeFloat<$semantic> {
            #[inline]
            fn pi() -> Self {
                // We take the value from the standard library as the most reasonable source for an exact π here.
                Self::from_bits($float_ty::consts::PI.to_bits().into())
            }
        }
    };
}

impl_ieee_pi!(f32, SingleS);
impl_ieee_pi!(f64, DoubleS);

#[cfg(test)]
mod tests {
    use rustc_apfloat::ieee::{DoubleS, HalfS, IeeeFloat, QuadS, SingleS};

    use super::sqrt;

    #[test]
    fn test_sqrt() {
        #[track_caller]
        fn test<S: rustc_apfloat::ieee::Semantics>(x: &str, expected: &str) {
            let x: IeeeFloat<S> = x.parse().unwrap();
            let expected: IeeeFloat<S> = expected.parse().unwrap();
            let result = sqrt(x);
            assert_eq!(result, expected);
        }

        fn exact_tests<S: rustc_apfloat::ieee::Semantics>() {
            test::<S>("0", "0");
            test::<S>("1", "1");
            test::<S>("1.5625", "1.25");
            test::<S>("2.25", "1.5");
            test::<S>("4", "2");
            test::<S>("5.0625", "2.25");
            test::<S>("9", "3");
            test::<S>("16", "4");
            test::<S>("25", "5");
            test::<S>("36", "6");
            test::<S>("49", "7");
            test::<S>("64", "8");
            test::<S>("81", "9");
            test::<S>("100", "10");

            test::<S>("0.5625", "0.75");
            test::<S>("0.25", "0.5");
            test::<S>("0.0625", "0.25");
            test::<S>("0.00390625", "0.0625");
        }

        exact_tests::<HalfS>();
        exact_tests::<SingleS>();
        exact_tests::<DoubleS>();
        exact_tests::<QuadS>();

        test::<SingleS>("2", "1.4142135");
        test::<DoubleS>("2", "1.4142135623730951");

        test::<SingleS>("1.1", "1.0488088");
        test::<DoubleS>("1.1", "1.0488088481701516");

        test::<SingleS>("2.2", "1.4832398");
        test::<DoubleS>("2.2", "1.4832396974191326");

        test::<SingleS>("1.22101e-40", "1.10499205e-20");
        test::<DoubleS>("1.22101e-310", "1.1049932126488395e-155");

        test::<SingleS>("3.4028235e38", "1.8446743e19");
        test::<DoubleS>("1.7976931348623157e308", "1.3407807929942596e154");
    }
}
