//! Helpful numeric operations.

use std::cmp::min;
use std::ops::RangeInclusive;

use libm::support::Float;

use crate::{Int, MinInt};

/// Extension to `libm`'s `Float` trait with methods that are useful for tests but not
/// needed in `libm` itself.
pub trait FloatExt: Float {
    /// The minimum subnormal number.
    const TINY_BITS: Self::Int = Self::Int::ONE;

    /// Retrieve additional constants for this float type.
    fn consts() -> Consts<Self> {
        Consts::new()
    }

    /// Increment by one ULP, saturating at infinity.
    fn next_up(self) -> Self {
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::INFINITY.to_bits() {
            return self;
        }

        let abs = self.abs().to_bits();
        let next_bits = if abs == Self::Int::ZERO {
            // Next up from 0 is the smallest subnormal
            Self::TINY_BITS
        } else if bits == abs {
            // Positive: counting up is more positive
            bits + Self::Int::ONE
        } else {
            // Negative: counting down is more positive
            bits - Self::Int::ONE
        };
        Self::from_bits(next_bits)
    }

    /// A faster way to effectively call `next_up` `n` times.
    fn n_up(self, n: Self::Int) -> Self {
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::INFINITY.to_bits() || n == Self::Int::ZERO {
            return self;
        }

        let abs = self.abs().to_bits();
        let is_positive = bits == abs;
        let crosses_zero = !is_positive && n > abs;
        let inf_bits = Self::INFINITY.to_bits();

        let next_bits = if abs == Self::Int::ZERO {
            min(n, inf_bits)
        } else if crosses_zero {
            min(n - abs, inf_bits)
        } else if is_positive {
            // Positive, counting up is more positive but this may overflow
            match bits.checked_add(n) {
                Some(v) if v >= inf_bits => inf_bits,
                Some(v) => v,
                None => inf_bits,
            }
        } else {
            // Negative, counting down is more positive
            bits - n
        };
        Self::from_bits(next_bits)
    }

    /// Decrement by one ULP, saturating at negative infinity.
    fn next_down(self) -> Self {
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::NEG_INFINITY.to_bits() {
            return self;
        }

        let abs = self.abs().to_bits();
        let next_bits = if abs == Self::Int::ZERO {
            // Next up from 0 is the smallest negative subnormal
            Self::TINY_BITS | Self::SIGN_MASK
        } else if bits == abs {
            // Positive: counting down is more negative
            bits - Self::Int::ONE
        } else {
            // Negative: counting up is more negative
            bits + Self::Int::ONE
        };
        Self::from_bits(next_bits)
    }

    /// A faster way to effectively call `next_down` `n` times.
    fn n_down(self, n: Self::Int) -> Self {
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::NEG_INFINITY.to_bits() || n == Self::Int::ZERO {
            return self;
        }

        let abs = self.abs().to_bits();
        let is_positive = bits == abs;
        let crosses_zero = is_positive && n > abs;
        let inf_bits = Self::INFINITY.to_bits();
        let ninf_bits = Self::NEG_INFINITY.to_bits();

        let next_bits = if abs == Self::Int::ZERO {
            min(n, inf_bits) | Self::SIGN_MASK
        } else if crosses_zero {
            min(n - abs, inf_bits) | Self::SIGN_MASK
        } else if is_positive {
            // Positive, counting down is more negative
            bits - n
        } else {
            // Negative, counting up is more negative but this may overflow
            match bits.checked_add(n) {
                Some(v) if v > ninf_bits => ninf_bits,
                Some(v) => v,
                None => ninf_bits,
            }
        };
        Self::from_bits(next_bits)
    }
}

impl<F> FloatExt for F where F: Float {}

/// Extra constants that are useful for tests.
#[derive(Debug, Clone, Copy)]
pub struct Consts<F> {
    /// The default quiet NaN, which is also the minimum quiet NaN.
    pub pos_nan: F,
    /// The default quiet NaN with negative sign.
    pub neg_nan: F,
    /// NaN with maximum (unsigned) significand to be a quiet NaN. The significand is saturated.
    pub max_qnan: F,
    /// NaN with minimum (unsigned) significand to be a signaling NaN.
    pub min_snan: F,
    /// NaN with maximum (unsigned) significand to be a signaling NaN.
    pub max_snan: F,
    pub neg_max_qnan: F,
    pub neg_min_snan: F,
    pub neg_max_snan: F,
}

impl<F: FloatExt> Consts<F> {
    fn new() -> Self {
        let top_sigbit_mask = F::Int::ONE << (F::SIG_BITS - 1);
        let pos_nan = F::EXP_MASK | top_sigbit_mask;
        let max_qnan = F::EXP_MASK | F::SIG_MASK;
        let min_snan = F::EXP_MASK | F::Int::ONE;
        let max_snan = (F::EXP_MASK | F::SIG_MASK) ^ top_sigbit_mask;

        let neg_nan = pos_nan | F::SIGN_MASK;
        let neg_max_qnan = max_qnan | F::SIGN_MASK;
        let neg_min_snan = min_snan | F::SIGN_MASK;
        let neg_max_snan = max_snan | F::SIGN_MASK;

        Self {
            pos_nan: F::from_bits(pos_nan),
            neg_nan: F::from_bits(neg_nan),
            max_qnan: F::from_bits(max_qnan),
            min_snan: F::from_bits(min_snan),
            max_snan: F::from_bits(max_snan),
            neg_max_qnan: F::from_bits(neg_max_qnan),
            neg_min_snan: F::from_bits(neg_min_snan),
            neg_max_snan: F::from_bits(neg_max_snan),
        }
    }

    pub fn iter(self) -> impl Iterator<Item = F> {
        // Destructure so we get unused warnings if we forget a list entry.
        let Self {
            pos_nan,
            neg_nan,
            max_qnan,
            min_snan,
            max_snan,
            neg_max_qnan,
            neg_min_snan,
            neg_max_snan,
        } = self;

        [pos_nan, neg_nan, max_qnan, min_snan, max_snan, neg_max_qnan, neg_min_snan, neg_max_snan]
            .into_iter()
    }
}

/// Return the number of steps between two floats, returning `None` if either input is NaN.
///
/// This is the number of steps needed for `n_up` or `n_down` to go between values. Infinities
/// are treated the same as those functions (will return the nearest finite value), and only one
/// of `-0` or `+0` is counted. It does not matter which value is greater.
pub fn ulp_between<F: Float>(x: F, y: F) -> Option<F::Int> {
    let a = as_ulp_steps(x)?;
    let b = as_ulp_steps(y)?;
    Some(a.abs_diff(b))
}

/// Return the (signed) number of steps from zero to `x`.
fn as_ulp_steps<F: Float>(x: F) -> Option<F::SignedInt> {
    let s = x.to_bits_signed();
    let val = if s >= F::SignedInt::ZERO {
        // each increment from `s = 0` is one step up from `x = 0.0`
        s
    } else {
        // each increment from `s = F::SignedInt::MIN` is one step down from `x = -0.0`
        F::SignedInt::MIN - s
    };

    // If `x` is NaN, return `None`
    (!x.is_nan()).then_some(val)
}

/// An iterator that returns floats with linearly spaced integer representations, which translates
/// to logarithmic spacing of their values.
///
/// Note that this tends to skip negative zero, so that needs to be checked explicitly.
pub fn logspace<F: FloatExt>(start: F, end: F, steps: F::Int) -> impl Iterator<Item = F> + Clone
where
    RangeInclusive<F::Int>: Iterator,
{
    assert!(!start.is_nan());
    assert!(!end.is_nan());
    assert!(end >= start);

    let mut steps = steps.checked_sub(F::Int::ONE).expect("`steps` must be at least 2");
    let between = ulp_between(start, end).expect("`start` or `end` is NaN");
    let spacing = (between / steps).max(F::Int::ONE);
    steps = steps.min(between); // At maximum, one step per ULP

    let mut x = start;
    (F::Int::ZERO..=steps).map(move |_| {
        let ret = x;
        x = x.n_up(spacing);
        ret
    })
}

#[cfg(test)]
mod tests {
    use std::cmp::max;

    use super::*;
    use crate::f8;

    #[test]
    fn test_next_up_down() {
        for (i, v) in f8::ALL.into_iter().enumerate() {
            let down = v.next_down().to_bits();
            let up = v.next_up().to_bits();

            if i == 0 {
                assert_eq!(down, f8::NEG_INFINITY.to_bits(), "{i} next_down({v:#010b})");
            } else {
                let expected =
                    if v == f8::ZERO { 1 | f8::SIGN_MASK } else { f8::ALL[i - 1].to_bits() };
                assert_eq!(down, expected, "{i} next_down({v:#010b})");
            }

            if i == f8::ALL_LEN - 1 {
                assert_eq!(up, f8::INFINITY.to_bits(), "{i} next_up({v:#010b})");
            } else {
                let expected = if v == f8::NEG_ZERO { 1 } else { f8::ALL[i + 1].to_bits() };
                assert_eq!(up, expected, "{i} next_up({v:#010b})");
            }
        }
    }

    #[test]
    fn test_next_up_down_inf_nan() {
        assert_eq!(f8::NEG_INFINITY.next_up().to_bits(), f8::ALL[0].to_bits(),);
        assert_eq!(f8::NEG_INFINITY.next_down().to_bits(), f8::NEG_INFINITY.to_bits(),);
        assert_eq!(f8::INFINITY.next_down().to_bits(), f8::ALL[f8::ALL_LEN - 1].to_bits(),);
        assert_eq!(f8::INFINITY.next_up().to_bits(), f8::INFINITY.to_bits(),);
        assert_eq!(f8::NAN.next_up().to_bits(), f8::NAN.to_bits(),);
        assert_eq!(f8::NAN.next_down().to_bits(), f8::NAN.to_bits(),);
    }

    #[test]
    fn test_n_up_down_quick() {
        assert_eq!(f8::ALL[0].n_up(4).to_bits(), f8::ALL[4].to_bits(),);
        assert_eq!(
            f8::ALL[f8::ALL_LEN - 1].n_down(4).to_bits(),
            f8::ALL[f8::ALL_LEN - 5].to_bits(),
        );

        // Check around zero
        assert_eq!(f8::from_bits(0b0).n_up(7).to_bits(), 0b0_0000_111);
        assert_eq!(f8::from_bits(0b0).n_down(7).to_bits(), 0b1_0000_111);

        // Check across zero
        assert_eq!(f8::from_bits(0b1_0000_111).n_up(8).to_bits(), 0b0_0000_001);
        assert_eq!(f8::from_bits(0b0_0000_111).n_down(8).to_bits(), 0b1_0000_001);
    }

    #[test]
    fn test_n_up_down_one() {
        // Verify that `n_up(1)` and `n_down(1)` are the same as `next_up()` and next_down()`.`
        for i in 0..u8::MAX {
            let v = f8::from_bits(i);
            assert_eq!(v.next_up().to_bits(), v.n_up(1).to_bits());
            assert_eq!(v.next_down().to_bits(), v.n_down(1).to_bits());
        }
    }

    #[test]
    fn test_n_up_down_inf_nan_zero() {
        assert_eq!(f8::NEG_INFINITY.n_up(1).to_bits(), f8::ALL[0].to_bits());
        assert_eq!(f8::NEG_INFINITY.n_up(239).to_bits(), f8::ALL[f8::ALL_LEN - 1].to_bits());
        assert_eq!(f8::NEG_INFINITY.n_up(240).to_bits(), f8::INFINITY.to_bits());
        assert_eq!(f8::NEG_INFINITY.n_down(u8::MAX).to_bits(), f8::NEG_INFINITY.to_bits());

        assert_eq!(f8::INFINITY.n_down(1).to_bits(), f8::ALL[f8::ALL_LEN - 1].to_bits());
        assert_eq!(f8::INFINITY.n_down(239).to_bits(), f8::ALL[0].to_bits());
        assert_eq!(f8::INFINITY.n_down(240).to_bits(), f8::NEG_INFINITY.to_bits());
        assert_eq!(f8::INFINITY.n_up(u8::MAX).to_bits(), f8::INFINITY.to_bits());

        assert_eq!(f8::NAN.n_up(u8::MAX).to_bits(), f8::NAN.to_bits());
        assert_eq!(f8::NAN.n_down(u8::MAX).to_bits(), f8::NAN.to_bits());

        assert_eq!(f8::ZERO.n_down(1).to_bits(), f8::TINY_BITS | f8::SIGN_MASK);
        assert_eq!(f8::NEG_ZERO.n_up(1).to_bits(), f8::TINY_BITS);
    }

    /// True if the specified range of `f8::ALL` includes both +0 and -0
    fn crossed_zero(start: usize, end: usize) -> bool {
        let crossed = &f8::ALL[start..=end];
        crossed.iter().any(|f| f8::eq_repr(*f, f8::ZERO))
            && crossed.iter().any(|f| f8::eq_repr(*f, f8::NEG_ZERO))
    }

    #[test]
    fn test_n_up_down() {
        for (i, v) in f8::ALL.into_iter().enumerate() {
            for n in 0..f8::ALL_LEN {
                let down = v.n_down(n as u8).to_bits();
                let up = v.n_up(n as u8).to_bits();

                if let Some(down_exp_idx) = i.checked_sub(n) {
                    // No overflow
                    let mut expected = f8::ALL[down_exp_idx].to_bits();
                    if n >= 1 && crossed_zero(down_exp_idx, i) {
                        // If both -0 and +0 are included, we need to adjust our expected value
                        match down_exp_idx.checked_sub(1) {
                            Some(v) => expected = f8::ALL[v].to_bits(),
                            // Saturate to -inf if we are out of values
                            None => expected = f8::NEG_INFINITY.to_bits(),
                        }
                    }
                    assert_eq!(down, expected, "{i} {n} n_down({v:#010b})");
                } else {
                    // Overflow to -inf
                    assert_eq!(down, f8::NEG_INFINITY.to_bits(), "{i} {n} n_down({v:#010b})");
                }

                let mut up_exp_idx = i + n;
                if up_exp_idx < f8::ALL_LEN {
                    // No overflow
                    if n >= 1 && up_exp_idx < f8::ALL_LEN && crossed_zero(i, up_exp_idx) {
                        // If both -0 and +0 are included, we need to adjust our expected value
                        up_exp_idx += 1;
                    }

                    let expected = if up_exp_idx >= f8::ALL_LEN {
                        f8::INFINITY.to_bits()
                    } else {
                        f8::ALL[up_exp_idx].to_bits()
                    };

                    assert_eq!(up, expected, "{i} {n} n_up({v:#010b})");
                } else {
                    // Overflow to +inf
                    assert_eq!(up, f8::INFINITY.to_bits(), "{i} {n} n_up({v:#010b})");
                }
            }
        }
    }

    #[test]
    fn test_ulp_between() {
        for (i, x) in f8::ALL.into_iter().enumerate() {
            for (j, y) in f8::ALL.into_iter().enumerate() {
                let ulp = ulp_between(x, y).unwrap();
                let make_msg = || format!("i: {i} j: {j} x: {x:b} y: {y:b} ulp {ulp}");

                let i_low = min(i, j);
                let i_hi = max(i, j);
                let mut expected = u8::try_from(i_hi - i_low).unwrap();
                if crossed_zero(i_low, i_hi) {
                    expected -= 1;
                }

                assert_eq!(ulp, expected, "{}", make_msg());

                // Skip if either are zero since `next_{up,down}` will count over it
                let either_zero = x == f8::ZERO || y == f8::ZERO;
                if x < y && !either_zero {
                    assert_eq!(x.n_up(ulp).to_bits(), y.to_bits(), "{}", make_msg());
                    assert_eq!(y.n_down(ulp).to_bits(), x.to_bits(), "{}", make_msg());
                } else if !either_zero {
                    assert_eq!(y.n_up(ulp).to_bits(), x.to_bits(), "{}", make_msg());
                    assert_eq!(x.n_down(ulp).to_bits(), y.to_bits(), "{}", make_msg());
                }
            }
        }
    }

    #[test]
    fn test_ulp_between_inf_nan_zero() {
        assert_eq!(ulp_between(f8::NEG_INFINITY, f8::INFINITY).unwrap(), f8::ALL_LEN as u8);
        assert_eq!(ulp_between(f8::INFINITY, f8::NEG_INFINITY).unwrap(), f8::ALL_LEN as u8);
        assert_eq!(
            ulp_between(f8::NEG_INFINITY, f8::ALL[f8::ALL_LEN - 1]).unwrap(),
            f8::ALL_LEN as u8 - 1
        );
        assert_eq!(ulp_between(f8::INFINITY, f8::ALL[0]).unwrap(), f8::ALL_LEN as u8 - 1);

        assert_eq!(ulp_between(f8::ZERO, f8::NEG_ZERO).unwrap(), 0);
        assert_eq!(ulp_between(f8::NAN, f8::ZERO), None);
        assert_eq!(ulp_between(f8::ZERO, f8::NAN), None);
    }

    #[test]
    fn test_logspace() {
        let ls: Vec<_> = logspace(f8::from_bits(0x0), f8::from_bits(0x4), 2).collect();
        let exp = [f8::from_bits(0x0), f8::from_bits(0x4)];
        assert_eq!(ls, exp);

        let ls: Vec<_> = logspace(f8::from_bits(0x0), f8::from_bits(0x4), 3).collect();
        let exp = [f8::from_bits(0x0), f8::from_bits(0x2), f8::from_bits(0x4)];
        assert_eq!(ls, exp);

        // Check that we include all values with no repeats if `steps` exceeds the maximum number
        // of steps.
        let ls: Vec<_> = logspace(f8::from_bits(0x0), f8::from_bits(0x3), 10).collect();
        let exp = [f8::from_bits(0x0), f8::from_bits(0x1), f8::from_bits(0x2), f8::from_bits(0x3)];
        assert_eq!(ls, exp);
    }

    #[test]
    fn test_consts() {
        let Consts {
            pos_nan,
            neg_nan,
            max_qnan,
            min_snan,
            max_snan,
            neg_max_qnan,
            neg_min_snan,
            neg_max_snan,
        } = f8::consts();

        assert_eq!(pos_nan.to_bits(), 0b0_1111_100);
        assert_eq!(neg_nan.to_bits(), 0b1_1111_100);
        assert_eq!(max_qnan.to_bits(), 0b0_1111_111);
        assert_eq!(min_snan.to_bits(), 0b0_1111_001);
        assert_eq!(max_snan.to_bits(), 0b0_1111_011);
        assert_eq!(neg_max_qnan.to_bits(), 0b1_1111_111);
        assert_eq!(neg_min_snan.to_bits(), 0b1_1111_001);
        assert_eq!(neg_max_snan.to_bits(), 0b1_1111_011);
    }
}
