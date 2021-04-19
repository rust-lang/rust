/// A mask where each lane is represented by a single bit.
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
#[repr(transparent)]
pub struct BitMask<const LANES: usize>(u64);

impl<const LANES: usize> BitMask<LANES>
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
        self.0 ^= ((value ^ self.test_unchecked(lane)) as u64) << lane
    }

    #[inline]
    pub fn to_int<V, T>(self) -> V
    where
        V: Default + AsMut<[T; LANES]>,
        T: From<i8>,
    {
        // TODO this should be an intrinsic sign-extension
        let mut v = V::default();
        for i in 0..LANES {
            let lane = unsafe { self.test_unchecked(i) };
            v.as_mut()[i] = (-(lane as i8)).into();
        }
        v
    }

    #[inline]
    pub unsafe fn from_int_unchecked<V>(value: V) -> Self
    where
        V: crate::LanesAtMost32,
    {
        let mask: V::BitMask = crate::intrinsics::simd_bitmask(value);
        Self(mask.into())
    }

    #[inline]
    pub fn to_bitmask(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn any(self) -> bool {
        self != Self::splat(false)
    }

    #[inline]
    pub fn all(self) -> bool {
        self == Self::splat(true)
    }
}

impl<const LANES: usize> core::ops::BitAnd for BitMask<LANES>
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl<const LANES: usize> core::ops::BitAnd<bool> for BitMask<LANES>
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: bool) -> Self {
        self & Self::splat(rhs)
    }
}

impl<const LANES: usize> core::ops::BitAnd<BitMask<LANES>> for bool
{
    type Output = BitMask<LANES>;
    #[inline]
    fn bitand(self, rhs: BitMask<LANES>) -> BitMask<LANES> {
        BitMask::<LANES>::splat(self) & rhs
    }
}

impl<const LANES: usize> core::ops::BitOr for BitMask<LANES>
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl<const LANES: usize> core::ops::BitXor for BitMask<LANES>
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<const LANES: usize> core::ops::Not for BitMask<LANES>
{
    type Output = BitMask<LANES>;
    #[inline]
    fn not(self) -> Self::Output {
        Self(!self.0) & Self::splat(true)
    }
}

impl<const LANES: usize> core::ops::BitAndAssign for BitMask<LANES>
{
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl<const LANES: usize> core::ops::BitOrAssign for BitMask<LANES>
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl<const LANES: usize> core::ops::BitXorAssign for BitMask<LANES>
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

pub type Mask8<const LANES: usize> = BitMask<LANES>;
pub type Mask16<const LANES: usize> = BitMask<LANES>;
pub type Mask32<const LANES: usize> = BitMask<LANES>;
pub type Mask64<const LANES: usize> = BitMask<LANES>;
pub type Mask128<const LANES: usize> = BitMask<LANES>;
pub type MaskSize<const LANES: usize> = BitMask<LANES>;
