//! Masks that take up full SIMD vector registers.

macro_rules! define_mask {
    {
        $(#[$attr:meta])*
        struct $name:ident<const $lanes:ident: usize>(
            crate::$type:ident<$lanes2:ident>
        );
    } => {
        $(#[$attr])*
        #[repr(transparent)]
        pub struct $name<const $lanes: usize>(crate::$type<$lanes>)
        where
            crate::LaneCount<$lanes>: crate::SupportedLaneCount;

        impl_full_mask_reductions! { $name, $type }

        impl<const LANES: usize> Copy for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {}

        impl<const LANES: usize> Clone for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const LANES: usize> PartialEq for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<const LANES: usize> PartialOrd for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<const LANES: usize> Eq for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {}

        impl<const LANES: usize> Ord for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }

        impl<const LANES: usize> $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            pub fn splat(value: bool) -> Self {
                Self(
                    <crate::$type<LANES>>::splat(
                        if value {
                            -1
                        } else {
                            0
                        }
                    ),
                )
            }

            #[inline]
            pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
                self.0[lane] == -1
            }

            #[inline]
            pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
                self.0[lane] = if value {
                    -1
                } else {
                    0
                }
            }

            #[inline]
            pub fn to_int(self) -> crate::$type<LANES> {
                self.0
            }

            #[inline]
            pub unsafe fn from_int_unchecked(value: crate::$type<LANES>) -> Self {
                Self(value)
            }

            #[inline]
            pub fn to_bitmask(self) -> [u8; crate::LaneCount::<LANES>::BITMASK_LEN] {
                unsafe {
                    // TODO remove the transmute when rustc can use arrays of u8 as bitmasks
                    assert_eq!(
                        core::mem::size_of::<<crate::LaneCount::<LANES> as crate::SupportedLaneCount>::IntBitMask>(),
                        crate::LaneCount::<LANES>::BITMASK_LEN,
                    );
                    let bitmask: <crate::LaneCount::<LANES> as crate::SupportedLaneCount>::IntBitMask = crate::intrinsics::simd_bitmask(self.0);
                    let mut bitmask: [u8; crate::LaneCount::<LANES>::BITMASK_LEN] = core::mem::transmute_copy(&bitmask);

                    // There is a bug where LLVM appears to implement this operation with the wrong
                    // bit order.
                    // TODO fix this in a better way
                    if cfg!(any(target_arch = "mips", target_arch = "mips64")) {
                        for x in bitmask.as_mut() {
                            *x = x.reverse_bits();
                        }
                    }

                    bitmask
                }
            }

            #[inline]
            pub fn from_bitmask(mut bitmask: [u8; crate::LaneCount::<LANES>::BITMASK_LEN]) -> Self {
                unsafe {
                    // There is a bug where LLVM appears to implement this operation with the wrong
                    // bit order.
                    // TODO fix this in a better way
                    if cfg!(any(target_arch = "mips", target_arch = "mips64")) {
                        for x in bitmask.as_mut() {
                            *x = x.reverse_bits();
                        }
                    }

                    // TODO remove the transmute when rustc can use arrays of u8 as bitmasks
                    assert_eq!(
                        core::mem::size_of::<<crate::LaneCount::<LANES> as crate::SupportedLaneCount>::IntBitMask>(),
                        crate::LaneCount::<LANES>::BITMASK_LEN,
                    );
                    let bitmask: <crate::LaneCount::<LANES> as crate::SupportedLaneCount>::IntBitMask = core::mem::transmute_copy(&bitmask);

                    Self::from_int_unchecked(crate::intrinsics::simd_select_bitmask(
                        bitmask,
                        Self::splat(true).to_int(),
                        Self::splat(false).to_int(),
                    ))
                }
            }
        }

        impl<const LANES: usize> core::convert::From<$name<LANES>> for crate::$type<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            fn from(value: $name<LANES>) -> Self {
                value.0
            }
        }

        impl<const LANES: usize> core::ops::BitAnd for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitOr for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitXor for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::Not for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }
    }
}

define_mask! {
    /// A mask equivalent to [SimdI8](crate::SimdI8), where all bits in the lane must be either set
    /// or unset.
    struct Mask8<const LANES: usize>(crate::SimdI8<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI16](crate::SimdI16), where all bits in the lane must be either set
    /// or unset.
    struct Mask16<const LANES: usize>(crate::SimdI16<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI32](crate::SimdI32), where all bits in the lane must be either set
    /// or unset.
    struct Mask32<const LANES: usize>(crate::SimdI32<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI64](crate::SimdI64), where all bits in the lane must be either set
    /// or unset.
    struct Mask64<const LANES: usize>(crate::SimdI64<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdIsize](crate::SimdIsize), where all bits in the lane must be either set
    /// or unset.
    struct MaskSize<const LANES: usize>(crate::SimdIsize<LANES>);
}

macro_rules! impl_from {
    { $from:ident ($from_inner:ident) => $($to:ident ($to_inner:ident)),* } => {
        $(
        impl<const LANES: usize> From<$from<LANES>> for $to<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            fn from(value: $from<LANES>) -> Self {
                let mut new = Self::splat(false);
                for i in 0..LANES {
                    unsafe { new.set_unchecked(i, value.test_unchecked(i)) }
                }
                new
            }
        }
        )*
    }
}
impl_from! { Mask8 (SimdI8) => Mask16 (SimdI16), Mask32 (SimdI32), Mask64 (SimdI64), MaskSize (SimdIsize) }
impl_from! { Mask16 (SimdI16) => Mask32 (SimdI32), Mask64 (SimdI64), MaskSize (SimdIsize), Mask8 (SimdI8) }
impl_from! { Mask32 (SimdI32) => Mask64 (SimdI64), MaskSize (SimdIsize), Mask8 (SimdI8), Mask16 (SimdI16) }
impl_from! { Mask64 (SimdI64) => MaskSize (SimdIsize), Mask8 (SimdI8), Mask16 (SimdI16), Mask32 (SimdI32) }
impl_from! { MaskSize (SimdIsize) => Mask8 (SimdI8), Mask16 (SimdI16), Mask32 (SimdI32), Mask64 (SimdI64) }
