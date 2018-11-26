use super::imp::{BitMaskWord, BITMASK_MASK, BITMASK_SHIFT};
use core::intrinsics;

/// A bit mask which contains the result of a `Match` operation on a `Group` and
/// allows iterating through them.
///
/// The bit mask is arranged so that low-order bits represent lower memory
/// addresses for group match results.
#[derive(Copy, Clone)]
pub struct BitMask(pub BitMaskWord);

impl BitMask {
    /// Returns a new `BitMask` with all bits inverted.
    #[inline]
    #[must_use]
    pub fn invert(self) -> BitMask {
        BitMask(self.0 ^ BITMASK_MASK)
    }

    /// Returns a new `BitMask` with the lowest bit removed.
    #[inline]
    #[must_use]
    pub fn remove_lowest_bit(self) -> BitMask {
        BitMask(self.0 & (self.0 - 1))
    }
    /// Returns whether the `BitMask` has at least one set bits.
    #[inline]
    pub fn any_bit_set(self) -> bool {
        self.0 != 0
    }

    /// Returns the first set bit in the `BitMask`, if there is one.
    #[inline]
    pub fn lowest_set_bit(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.trailing_zeros())
        }
    }

    /// Returns the first set bit in the `BitMask`, if there is one. The
    /// bitmask must not be empty.
    #[inline]
    pub unsafe fn lowest_set_bit_nonzero(self) -> usize {
        intrinsics::cttz_nonzero(self.0) as usize >> BITMASK_SHIFT
    }

    /// Returns the number of trailing zeroes in the `BitMask`.
    #[inline]
    pub fn trailing_zeros(self) -> usize {
        // ARM doesn't have a CTZ instruction, and instead uses RBIT + CLZ.
        // However older ARM versions (pre-ARMv7) don't have RBIT and need to
        // emulate it instead. Since we only have 1 bit set in each byte we can
        // use REV + CLZ instead.
        if cfg!(target_arch = "arm") && BITMASK_SHIFT >= 3 {
            self.0.swap_bytes().leading_zeros() as usize >> BITMASK_SHIFT
        } else {
            self.0.trailing_zeros() as usize >> BITMASK_SHIFT
        }
    }

    /// Returns the number of leading zeroes in the `BitMask`.
    #[inline]
    pub fn leading_zeros(self) -> usize {
        self.0.leading_zeros() as usize >> BITMASK_SHIFT
    }
}

impl IntoIterator for BitMask {
    type Item = usize;
    type IntoIter = BitMaskIter;

    #[inline]
    fn into_iter(self) -> BitMaskIter {
        BitMaskIter(self)
    }
}

/// Iterator over the contents of a `BitMask`, returning the indicies of set
/// bits.
pub struct BitMaskIter(BitMask);

impl Iterator for BitMaskIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let bit = self.0.lowest_set_bit()?;
        self.0 = self.0.remove_lowest_bit();
        Some(bit)
    }
}
