use core::ops;

use int::LargeInt;
use int::Int;

trait Mul: LargeInt {
    fn mul(self, other: Self) -> Self {
        let half_bits = Self::bits() / 4;
        let lower_mask = !<<Self as LargeInt>::LowHalf>::zero() >> half_bits;
        let mut low = (self.low() & lower_mask).wrapping_mul(other.low() & lower_mask);
        let mut t = low >> half_bits;
        low &= lower_mask;
        t += (self.low() >> half_bits).wrapping_mul(other.low() & lower_mask);
        low += (t & lower_mask) << half_bits;
        let mut high = Self::low_as_high(t >> half_bits);
        t = low >> half_bits;
        low &= lower_mask;
        t += (other.low() >> half_bits).wrapping_mul(self.low() & lower_mask);
        low += (t & lower_mask) << half_bits;
        high += Self::low_as_high(t >> half_bits);
        high += Self::low_as_high((self.low() >> half_bits).wrapping_mul(other.low() >> half_bits));
        high = high.wrapping_add(self.high().wrapping_mul(Self::low_as_high(other.low())))
                   .wrapping_add(Self::low_as_high(self.low()).wrapping_mul(other.high()));
        Self::from_parts(low, high)
    }
}

impl Mul for u64 {}
impl Mul for i128 {}

trait Mulo: Int + ops::Neg<Output = Self> {
    fn mulo(self, other: Self, overflow: &mut i32) -> Self {
        *overflow = 0;
        let result = self.wrapping_mul(other);
        if self == Self::min_value() {
            if other != Self::zero() && other != Self::one() {
                *overflow = 1;
            }
            return result;
        }
        if other == Self::min_value() {
            if self != Self::zero() && self != Self::one() {
                *overflow = 1;
            }
            return result;
        }

        let sa = self >> (Self::bits() - 1);
        let abs_a = (self ^ sa) - sa;
        let sb = other >> (Self::bits() - 1);
        let abs_b = (other ^ sb) - sb;
        let two = Self::one() + Self::one();
        if abs_a < two || abs_b < two {
            return result;
        }
        if sa == sb {
            if abs_a > Self::max_value() / abs_b {
                *overflow = 1;
            }
        } else {
            if abs_a > Self::min_value() / -abs_b {
                *overflow = 1;
            }
        }
        result
    }
}

impl Mulo for i32 {}
impl Mulo for i64 {}
impl Mulo for i128 {}

intrinsics! {
    #[cfg(not(all(feature = "c", target_arch = "x86")))]
    pub extern "C" fn __muldi3(a: u64, b: u64) -> u64 {
        a.mul(b)
    }

    #[aapcs_on_arm]
    pub extern "C" fn __multi3(a: i128, b: i128) -> i128 {
        a.mul(b)
    }

    pub extern "C" fn __mulosi4(a: i32, b: i32, oflow: &mut i32) -> i32 {
        a.mulo(b, oflow)
    }

    pub extern "C" fn __mulodi4(a: i64, b: i64, oflow: &mut i32) -> i64 {
        a.mulo(b, oflow)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __muloti4(a: i128, b: i128, oflow: &mut i32) -> i128 {
        a.mulo(b, oflow)
    }
}
