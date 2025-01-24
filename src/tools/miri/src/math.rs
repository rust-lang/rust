use rand::Rng as _;
use rand::distributions::Distribution as _;
use rustc_apfloat::Float as _;
use rustc_apfloat::ieee::IeeeFloat;

/// Disturbes a floating-point result by a relative error on the order of (-2^scale, 2^scale).
pub(crate) fn apply_random_float_error<F: rustc_apfloat::Float>(
    ecx: &mut crate::MiriInterpCx<'_>,
    val: F,
    err_scale: i32,
) -> F {
    let rng = ecx.machine.rng.get_mut();
    // Generate a random integer in the range [0, 2^PREC).
    let dist = rand::distributions::Uniform::new(0, 1 << F::PRECISION);
    let err = F::from_u128(dist.sample(rng))
        .value
        .scalbn(err_scale.strict_sub(F::PRECISION.try_into().unwrap()));
    // give it a random sign
    let err = if rng.gen::<bool>() { -err } else { err };
    // multiple the value with (1+err)
    (val * (F::from_u128(1).value + err).value).value
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
