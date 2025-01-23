use super::super::{CastFrom, CastInto, Float, IntTy, MinInt};

/// Scale the exponent.
///
/// From N3220:
///
/// > The scalbn and scalbln functions compute `x * b^n`, where `b = FLT_RADIX` if the return type
/// > of the function is a standard floating type, or `b = 10` if the return type of the function
/// > is a decimal floating type. A range error occurs for some finite x, depending on n.
/// >
/// > [...]
/// >
/// > * `scalbn(±0, n)` returns `±0`.
/// > * `scalbn(x, 0)` returns `x`.
/// > * `scalbn(±∞, n)` returns `±∞`.
/// >
/// > If the calculation does not overflow or underflow, the returned value is exact and
/// > independent of the current rounding direction mode.
pub fn scalbn<F: Float>(mut x: F, mut n: i32) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastFrom<i32>,
    F::Int: CastFrom<u32>,
{
    let zero = IntTy::<F>::ZERO;

    // Bits including the implicit bit
    let sig_total_bits = F::SIG_BITS + 1;

    // Maximum and minimum values when biased
    let exp_max: i32 = F::EXP_BIAS as i32;
    let exp_min = -(exp_max - 1);

    // 2 ^ Emax, where Emax is the maximum biased exponent value (1023 for f64)
    let f_exp_max = F::from_parts(false, F::EXP_BIAS << 1, zero);

    // 2 ^ Emin, where Emin is the minimum biased exponent value (-1022 for f64)
    let f_exp_min = F::from_parts(false, 1, zero);

    // 2 ^ sig_total_bits, representation of what can be accounted for with subnormals
    let f_exp_subnorm = F::from_parts(false, sig_total_bits + F::EXP_BIAS, zero);

    if n > exp_max {
        x *= f_exp_max;
        n -= exp_max;
        if n > exp_max {
            x *= f_exp_max;
            n -= exp_max;
            if n > exp_max {
                n = exp_max;
            }
        }
    } else if n < exp_min {
        let mul = f_exp_min * f_exp_subnorm;
        let add = (exp_max - 1) - sig_total_bits as i32;

        x *= mul;
        n += add;
        if n < exp_min {
            x *= mul;
            n += add;
            if n < exp_min {
                n = exp_min;
            }
        }
    }

    x * F::from_parts(false, (F::EXP_BIAS as i32 + n) as u32, zero)
}

#[cfg(test)]
mod tests {
    use super::super::super::Int;
    use super::*;

    // Tests against N3220
    fn spec_test<F: Float>()
    where
        u32: CastInto<F::Int>,
        F::Int: CastFrom<i32>,
        F::Int: CastFrom<u32>,
    {
        // `scalbn(±0, n)` returns `±0`.
        assert_biteq!(scalbn(F::NEG_ZERO, 10), F::NEG_ZERO);
        assert_biteq!(scalbn(F::NEG_ZERO, 0), F::NEG_ZERO);
        assert_biteq!(scalbn(F::NEG_ZERO, -10), F::NEG_ZERO);
        assert_biteq!(scalbn(F::ZERO, 10), F::ZERO);
        assert_biteq!(scalbn(F::ZERO, 0), F::ZERO);
        assert_biteq!(scalbn(F::ZERO, -10), F::ZERO);

        // `scalbn(x, 0)` returns `x`.
        assert_biteq!(scalbn(F::MIN, 0), F::MIN);
        assert_biteq!(scalbn(F::MAX, 0), F::MAX);
        assert_biteq!(scalbn(F::INFINITY, 0), F::INFINITY);
        assert_biteq!(scalbn(F::NEG_INFINITY, 0), F::NEG_INFINITY);
        assert_biteq!(scalbn(F::ZERO, 0), F::ZERO);
        assert_biteq!(scalbn(F::NEG_ZERO, 0), F::NEG_ZERO);

        // `scalbn(±∞, n)` returns `±∞`.
        assert_biteq!(scalbn(F::INFINITY, 10), F::INFINITY);
        assert_biteq!(scalbn(F::INFINITY, -10), F::INFINITY);
        assert_biteq!(scalbn(F::NEG_INFINITY, 10), F::NEG_INFINITY);
        assert_biteq!(scalbn(F::NEG_INFINITY, -10), F::NEG_INFINITY);

        // NaN should remain NaNs.
        assert!(scalbn(F::NAN, 10).is_nan());
        assert!(scalbn(F::NAN, 0).is_nan());
        assert!(scalbn(F::NAN, -10).is_nan());
        assert!(scalbn(-F::NAN, 10).is_nan());
        assert!(scalbn(-F::NAN, 0).is_nan());
        assert!(scalbn(-F::NAN, -10).is_nan());
    }

    #[test]
    fn spec_test_f32() {
        spec_test::<f32>();
    }

    #[test]
    fn spec_test_f64() {
        spec_test::<f64>();
    }
}
