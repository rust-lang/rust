use super::super::{Float, Int, IntTy, MinInt};

pub fn trunc<F: Float>(x: F) -> F {
    let mut xi: F::Int = x.to_bits();
    let e: i32 = x.exp_unbiased();

    // C1: The represented value has no fractional part, so no truncation is needed
    if e >= F::SIG_BITS as i32 {
        return x;
    }

    let mask = if e < 0 {
        // C2: If the exponent is negative, the result will be zero so we mask out everything
        // except the sign.
        F::SIGN_MASK
    } else {
        // C3: Otherwise, we mask out the last `e` bits of the significand.
        !(F::SIG_MASK >> e.unsigned())
    };

    // C4: If the to-be-masked-out portion is already zero, we have an exact result
    if (xi & !mask) == IntTy::<F>::ZERO {
        return x;
    }

    // C5: Otherwise the result is inexact and we will truncate. Raise `FE_INEXACT`, mask the
    // result, and return.
    force_eval!(x + F::MAX);
    xi &= mask;
    F::from_bits(xi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_check() {
        assert_biteq!(trunc(1.1f32), 1.0);
        assert_biteq!(trunc(1.1f64), 1.0);

        // C1
        assert_biteq!(trunc(hf32!("0x1p23")), hf32!("0x1p23"));
        assert_biteq!(trunc(hf64!("0x1p52")), hf64!("0x1p52"));
        assert_biteq!(trunc(hf32!("-0x1p23")), hf32!("-0x1p23"));
        assert_biteq!(trunc(hf64!("-0x1p52")), hf64!("-0x1p52"));

        // C2
        assert_biteq!(trunc(hf32!("0x1p-1")), 0.0);
        assert_biteq!(trunc(hf64!("0x1p-1")), 0.0);
        assert_biteq!(trunc(hf32!("-0x1p-1")), -0.0);
        assert_biteq!(trunc(hf64!("-0x1p-1")), -0.0);
    }
}
