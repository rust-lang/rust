//! Very limited 256-bit arithmetic.

#[cfg(test)]
mod tests;

use core::ops::{Sub, SubAssign};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct U256 {
    high: u128,
    low: u128,
}

impl U256 {
    pub(crate) const ONE: Self = U256 { high: 0, low: 1 };

    /// `x << shift`.
    ///
    /// Requires `0 <= shift < 128`.
    pub(crate) fn shl_u128(x: u128, shift: u32) -> Self {
        Self { low: x << shift, high: if shift != 0 { x >> (128 - shift) } else { 0 } }
    }

    /// The number of leading 0 bits.
    pub(crate) fn leading_zeros(self) -> u32 {
        if self.high != 0 { self.high.leading_zeros() } else { 128 + self.low.leading_zeros() }
    }

    /// Returns (quotient, remainder).
    ///
    /// Requires `self < (rhs << 128)`
    pub(crate) fn div_rem(mut self, rhs: u128) -> (u128, u128) {
        let rhs_len = 128 - rhs.leading_zeros();
        let self_len = 256 - self.leading_zeros();
        // self < (rhs << shift_limit)
        let shift_limit = (self_len + 1).saturating_sub(rhs_len).min(128);
        let mut q = 0;
        for shift in (0..shift_limit).rev() {
            let rhs_shifted = Self::shl_u128(rhs, shift);
            if self >= rhs_shifted {
                q |= 1 << shift;
                self -= rhs_shifted;
            }
        }
        (q, self.low)
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
