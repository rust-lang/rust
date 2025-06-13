use rand::Rng as _;
use rustc_apfloat::Float as _;
use rustc_apfloat::ieee::IeeeFloat;
use rustc_middle::ty::{self, FloatTy, ScalarInt};

use crate::*;

/// Disturbes a floating-point result by a relative error in the range (-2^scale, 2^scale).
///
/// For a 2^N ULP error, you can use an `err_scale` of `-(F::PRECISION - 1 - N)`.
/// In other words, a 1 ULP (absolute) error is the same as a `2^-(F::PRECISION-1)` relative error.
/// (Subtracting 1 compensates for the integer bit.)
pub(crate) fn apply_random_float_error<F: rustc_apfloat::Float>(
    ecx: &mut crate::MiriInterpCx<'_>,
    val: F,
    err_scale: i32,
) -> F {
    if !ecx.machine.float_nondet {
        return val;
    }

    let rng = ecx.machine.rng.get_mut();
    // Generate a random integer in the range [0, 2^PREC).
    // (When read as binary, the position of the first `1` determines the exponent,
    // and the remaining bits fill the mantissa. `PREC` is one plus the size of the mantissa,
    // so this all works out.)
    let r = F::from_u128(rng.random_range(0..(1 << F::PRECISION))).value;
    // Multiply this with 2^(scale - PREC). The result is between 0 and
    // 2^PREC * 2^(scale - PREC) = 2^scale.
    let err = r.scalbn(err_scale.strict_sub(F::PRECISION.try_into().unwrap()));
    // give it a random sign
    let err = if rng.random() { -err } else { err };
    // multiple the value with (1+err)
    (val * (F::from_u128(1).value + err).value).value
}

/// [`apply_random_float_error`] gives instructions to apply a 2^N ULP error.
/// This function implements these instructions such that applying a 2^N ULP error is less error prone.
/// So for a 2^N ULP error, you would pass N as the `ulp_exponent` argument.
pub(crate) fn apply_random_float_error_ulp<F: rustc_apfloat::Float>(
    ecx: &mut crate::MiriInterpCx<'_>,
    val: F,
    ulp_exponent: u32,
) -> F {
    let n = i32::try_from(ulp_exponent)
        .expect("`err_scale_for_ulp`: exponent is too large to create an error scale");
    // we know this fits
    let prec = i32::try_from(F::PRECISION).unwrap();
    let err_scale = -(prec - n - 1);
    apply_random_float_error(ecx, val, err_scale)
}

/// Applies a random 16ULP floating point error to `val` and returns the new value.
/// Will fail if `val` is not a floating point number.
pub(crate) fn apply_random_float_error_to_imm<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    val: ImmTy<'tcx>,
    ulp_exponent: u32,
) -> InterpResult<'tcx, ImmTy<'tcx>> {
    let scalar = val.to_scalar_int()?;
    let res: ScalarInt = match val.layout.ty.kind() {
        ty::Float(FloatTy::F16) =>
            apply_random_float_error_ulp(ecx, scalar.to_f16(), ulp_exponent).into(),
        ty::Float(FloatTy::F32) =>
            apply_random_float_error_ulp(ecx, scalar.to_f32(), ulp_exponent).into(),
        ty::Float(FloatTy::F64) =>
            apply_random_float_error_ulp(ecx, scalar.to_f64(), ulp_exponent).into(),
        ty::Float(FloatTy::F128) =>
            apply_random_float_error_ulp(ecx, scalar.to_f128(), ulp_exponent).into(),
        _ => bug!("intrinsic called with non-float input type"),
    };

    interp_ok(ImmTy::from_scalar_int(res, val.layout))
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

/// Extend functionality of rustc_apfloat softfloats
pub trait IeeeExt: rustc_apfloat::Float {
    #[inline]
    fn one() -> Self {
        Self::from_u128(1).value
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.maximum(min).minimum(max)
    }
}
impl<S: rustc_apfloat::ieee::Semantics> IeeeExt for IeeeFloat<S> {}

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
