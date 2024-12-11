//! Limited unsigned 256-bit arithmetic for internal use.

#[cfg(test)]
mod tests;

use core::ops::{BitOr, BitOrAssign, Div, DivAssign, Shl, ShlAssign, Sub, SubAssign};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct U256 {
    high: u128,
    low: u128,
}

impl U256 {
    pub(crate) const ZERO: Self = Self { high: 0, low: 0 };

    pub(crate) const ONE: Self = Self { high: 0, low: 1 };

    /// The number of leading 0 bits.
    pub(crate) fn leading_zeros(self) -> u32 {
        if self.high != 0 { self.high.leading_zeros() } else { 128 + self.low.leading_zeros() }
    }

    /// Returns (quotient, remainder).
    pub(crate) fn div_rem(mut self, rhs: Self) -> (Self, Self) {
        assert!(rhs != Self::ZERO);
        let shift_limit = (rhs.leading_zeros() + 1).saturating_sub(self.leading_zeros());
        let mut q = Self::ZERO;
        for shift in (0..shift_limit).rev() {
            let rhs_shifted = rhs << shift;
            if self >= rhs_shifted {
                q |= Self::ONE << shift;
                self -= rhs_shifted;
            }
        }
        (q, self)
    }

    /// Wrap to `u128`.
    pub(crate) fn wrap_u128(self) -> u128 {
        self.low
    }
}

impl From<u128> for U256 {
    fn from(value: u128) -> Self {
        Self { high: 0, low: value }
    }
}

impl Shl<u32> for U256 {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self {
        if rhs == 0 {
            self
        } else if rhs < 128 {
            Self { high: self.high << rhs | self.low >> (128 - rhs), low: self.low << rhs }
        } else if rhs == 128 {
            Self { high: self.low, low: 0 }
        } else if rhs < 256 {
            Self { high: self.low << (rhs - 128), low: 0 }
        } else {
            Self::ZERO
        }
    }
}

impl ShlAssign<u32> for U256 {
    fn shl_assign(&mut self, rhs: u32) {
        *self = *self << rhs;
    }
}

impl Sub for U256 {
    type Output = Self;

    fn sub(self, rhs: U256) -> Self {
        let (low, borrow) = self.low.overflowing_sub(rhs.low);
        let (high, _) = self.high.borrowing_sub(rhs.high, borrow);
        Self { low, high }
    }
}

impl SubAssign for U256 {
    fn sub_assign(&mut self, rhs: U256) {
        *self = *self - rhs;
    }
}

impl Div<U256> for U256 {
    type Output = Self;

    fn div(self, rhs: U256) -> Self {
        self.div_rem(rhs).0
    }
}

impl DivAssign for U256 {
    fn div_assign(&mut self, rhs: U256) {
        *self = *self / rhs;
    }
}

impl BitOr<U256> for U256 {
    type Output = Self;

    fn bitor(self, rhs: U256) -> Self {
        Self { high: self.high | rhs.high, low: self.low | rhs.low }
    }
}

impl BitOrAssign<U256> for U256 {
    fn bitor_assign(&mut self, rhs: U256) {
        *self = *self | rhs;
    }
}
