use crate::LanesAtMost32;

/// A mask where each lane is represented by a single bit.
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
#[repr(transparent)]
pub struct BitMask<const LANES: usize>(u64)

impl<const LANES: usize> BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    pub fn splat(value: bool) -> Self {
        if value {
            Self(u64::MAX >> (64 - LANES))
        } else {
            Self(u64::MIN)
        }
    }

    #[inline]
    pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
        (self.0 >> lane) & 0x1 > 0
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0 ^= ((value ^ self.test(lane)) as u64) << lane
    }
}

impl<const LANES: usize> core::ops::BitAnd for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl<const LANES: usize> core::ops::BitAnd<bool> for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: bool) -> Self {
        self & Self::splat(rhs)
    }
}

impl<const LANES: usize> core::ops::BitAnd<BitMask<LANES>> for bool
where
    BitMask<LANES>: LanesAtMost32,
{
    type Output = BitMask<LANES>;
    #[inline]
    fn bitand(self, rhs: BitMask<LANES>) -> BitMask<LANES> {
        BitMask::<LANES>::splat(self) & rhs
    }
}

impl<const LANES: usize> core::ops::BitOr for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl<const LANES: usize> core::ops::BitOr<bool> for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: bool) -> Self {
        self | Self::splat(rhs)
    }
}

impl<const LANES: usize> core::ops::BitOr<BitMask<LANES>> for bool
where
    BitMask<LANES>: LanesAtMost32,
{
    type Output = BitMask<LANES>;
    #[inline]
    fn bitor(self, rhs: BitMask<LANES>) -> BitMask<LANES> {
        BitMask::<LANES>::splat(self) | rhs
    }
}

impl<const LANES: usize> core::ops::BitXor for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<const LANES: usize> core::ops::BitXor<bool> for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: bool) -> Self::Output {
        self ^ Self::splat(rhs)
    }
}

impl<const LANES: usize> core::ops::BitXor<BitMask<LANES>> for bool
where
    BitMask<LANES>: LanesAtMost32,
{
    type Output = BitMask<LANES>;
    #[inline]
    fn bitxor(self, rhs: BitMask<LANES>) -> Self::Output {
        BitMask::<LANES>::splat(self) ^ rhs
    }
}

impl<const LANES: usize> core::ops::Not for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    type Output = BitMask<LANES>;
    #[inline]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl<const LANES: usize> core::ops::BitAndAssign for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl<const LANES: usize> core::ops::BitAndAssign<bool> for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: bool) {
        *self &= Self::splat(rhs);
    }
}

impl<const LANES: usize> core::ops::BitOrAssign for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl<const LANES: usize> core::ops::BitOrAssign<bool> for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: bool) {
        *self |= Self::splat(rhs);
    }
}

impl<const LANES: usize> core::ops::BitXorAssign for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl<const LANES: usize> core::ops::BitXorAssign<bool> for BitMask<LANES>
where
    Self: LanesAtMost32,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: bool) {
        *self ^= Self::splat(rhs);
    }
}
