use crate::convert::{TryFrom, TryInto};
use crate::intrinsics::assert_unsafe_precondition;
use crate::num::NonZeroUsize;
use crate::{cmp, fmt, hash, mem, num};

/// A type storing a `usize` which is a power of two, and thus
/// represents a possible alignment in the rust abstract machine.
///
/// Note that particularly large alignments, while representable in this type,
/// are likely not to be supported by actual allocators and linkers.
#[unstable(feature = "ptr_alignment_type", issue = "102070")]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Alignment(AlignmentEnum);

// Alignment is `repr(usize)`, but via extra steps.
const _: () = assert!(mem::size_of::<Alignment>() == mem::size_of::<usize>());
const _: () = assert!(mem::align_of::<Alignment>() == mem::align_of::<usize>());

fn _alignment_can_be_structurally_matched(a: Alignment) -> bool {
    matches!(a, Alignment::MIN)
}

impl Alignment {
    /// The smallest possible alignment, 1.
    ///
    /// All addresses are always aligned at least this much.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_alignment_type)]
    /// use std::ptr::Alignment;
    ///
    /// assert_eq!(Alignment::MIN.as_usize(), 1);
    /// ```
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    pub const MIN: Self = Self(AlignmentEnum::_Align1Shl0);

    /// Returns the alignment for a type.
    ///
    /// This provides the same numerical value as [`mem::align_of`],
    /// but in an `Alignment` instead of a `usize`.
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[inline]
    pub const fn of<T>() -> Self {
        // SAFETY: rustc ensures that type alignment is always a power of two.
        unsafe { Alignment::new_unchecked(mem::align_of::<T>()) }
    }

    /// Creates an `Alignment` from a `usize`, or returns `None` if it's
    /// not a power of two.
    ///
    /// Note that `0` is not a power of two, nor a valid alignment.
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[inline]
    pub const fn new(align: usize) -> Option<Self> {
        if align.is_power_of_two() {
            // SAFETY: Just checked it only has one bit set
            Some(unsafe { Self::new_unchecked(align) })
        } else {
            None
        }
    }

    /// Creates an `Alignment` from a power-of-two `usize`.
    ///
    /// # Safety
    ///
    /// `align` must be a power of two.
    ///
    /// Equivalently, it must be `1 << exp` for some `exp` in `0..usize::BITS`.
    /// It must *not* be zero.
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[rustc_const_unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[inline]
    pub const unsafe fn new_unchecked(align: usize) -> Self {
        // SAFETY: Precondition passed to the caller.
        unsafe {
            assert_unsafe_precondition!(
               "Alignment::new_unchecked requires a power of two",
                (align: usize) => align.is_power_of_two()
            )
        };

        // SAFETY: By precondition, this must be a power of two, and
        // our variants encompass all possible powers of two.
        unsafe { mem::transmute::<usize, Alignment>(align) }
    }

    /// Returns the alignment as a [`usize`]
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[rustc_const_unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns the alignment as a [`NonZeroUsize`]
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[inline]
    pub const fn as_nonzero(self) -> NonZeroUsize {
        // SAFETY: All the discriminants are non-zero.
        unsafe { NonZeroUsize::new_unchecked(self.as_usize()) }
    }

    /// Returns the base-2 logarithm of the alignment.
    ///
    /// This is always exact, as `self` represents a power of two.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_alignment_type)]
    /// use std::ptr::Alignment;
    ///
    /// assert_eq!(Alignment::of::<u8>().log2(), 0);
    /// assert_eq!(Alignment::new(1024).unwrap().log2(), 10);
    /// ```
    #[unstable(feature = "ptr_alignment_type", issue = "102070")]
    #[inline]
    pub fn log2(self) -> u32 {
        self.as_nonzero().trailing_zeros()
    }
}

#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl fmt::Debug for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (1 << {:?})", self.as_nonzero(), self.log2())
    }
}

#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl TryFrom<NonZeroUsize> for Alignment {
    type Error = num::TryFromIntError;

    #[inline]
    fn try_from(align: NonZeroUsize) -> Result<Alignment, Self::Error> {
        align.get().try_into()
    }
}

#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl TryFrom<usize> for Alignment {
    type Error = num::TryFromIntError;

    #[inline]
    fn try_from(align: usize) -> Result<Alignment, Self::Error> {
        Self::new(align).ok_or(num::TryFromIntError(()))
    }
}

#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl From<Alignment> for NonZeroUsize {
    #[inline]
    fn from(align: Alignment) -> NonZeroUsize {
        align.as_nonzero()
    }
}

#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl From<Alignment> for usize {
    #[inline]
    fn from(align: Alignment) -> usize {
        align.as_usize()
    }
}

#[rustc_const_unstable(feature = "const_alloc_layout", issue = "67521")]
#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl cmp::Ord for Alignment {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_nonzero().get().cmp(&other.as_nonzero().get())
    }
}

#[rustc_const_unstable(feature = "const_alloc_layout", issue = "67521")]
#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl cmp::PartialOrd for Alignment {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[unstable(feature = "ptr_alignment_type", issue = "102070")]
impl hash::Hash for Alignment {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_nonzero().hash(state)
    }
}

macro_rules! alignment_enum {
    (
        $( #[$attrs:meta] )+
        enum $enum:ident {
            #[$width16:meta]
            $( $variant16:ident => $align16:literal, )+
            #[$width32:meta]
            $( $variant32:ident => $align32:literal, )+
            #[$width64:meta]
            $( $variant64:ident => $align64:literal, )+
        }
    ) => {
        #[$width16]
        $( #[$attrs] )+
        enum $enum {
            $( $variant16 = 1 << $align16, )+
        }

        #[$width32]
        $( #[$attrs] )+
        enum $enum {
            $( $variant16 = 1 << $align16, )+
            $( $variant32 = 1 << $align32, )+
        }

        #[$width64]
        $( #[$attrs] )+
        enum $enum {
            $( $variant16 = 1 << $align16, )+
            $( $variant32 = 1 << $align32, )+
            $( $variant64 = 1 << $align64, )+
        }
    };
}

alignment_enum! {
    #[derive(Copy, Clone, PartialEq, Eq)]
    #[repr(usize)]
    enum AlignmentEnum {
        #[cfg(target_pointer_width = "16")]
        _Align1Shl0 => 0,
        _Align1Shl1 => 1,
        _Align1Shl2 => 2,
        _Align1Shl3 => 3,
        _Align1Shl4 => 4,
        _Align1Shl5 => 5,
        _Align1Shl6 => 6,
        _Align1Shl7 => 7,
        _Align1Shl8 => 8,
        _Align1Shl9 => 9,
        _Align1Shl10 => 10,
        _Align1Shl11 => 11,
        _Align1Shl12 => 12,
        _Align1Shl13 => 13,
        _Align1Shl14 => 14,
        _Align1Shl15 => 15,
        #[cfg(target_pointer_width = "32")]
        _Align1Shl16 => 16,
        _Align1Shl17 => 17,
        _Align1Shl18 => 18,
        _Align1Shl19 => 19,
        _Align1Shl20 => 20,
        _Align1Shl21 => 21,
        _Align1Shl22 => 22,
        _Align1Shl23 => 23,
        _Align1Shl24 => 24,
        _Align1Shl25 => 25,
        _Align1Shl26 => 26,
        _Align1Shl27 => 27,
        _Align1Shl28 => 28,
        _Align1Shl29 => 29,
        _Align1Shl30 => 30,
        _Align1Shl31 => 31,
        #[cfg(target_pointer_width = "64")]
        _Align1Shl32 => 32,
        _Align1Shl33 => 33,
        _Align1Shl34 => 34,
        _Align1Shl35 => 35,
        _Align1Shl36 => 36,
        _Align1Shl37 => 37,
        _Align1Shl38 => 38,
        _Align1Shl39 => 39,
        _Align1Shl40 => 40,
        _Align1Shl41 => 41,
        _Align1Shl42 => 42,
        _Align1Shl43 => 43,
        _Align1Shl44 => 44,
        _Align1Shl45 => 45,
        _Align1Shl46 => 46,
        _Align1Shl47 => 47,
        _Align1Shl48 => 48,
        _Align1Shl49 => 49,
        _Align1Shl50 => 50,
        _Align1Shl51 => 51,
        _Align1Shl52 => 52,
        _Align1Shl53 => 53,
        _Align1Shl54 => 54,
        _Align1Shl55 => 55,
        _Align1Shl56 => 56,
        _Align1Shl57 => 57,
        _Align1Shl58 => 58,
        _Align1Shl59 => 59,
        _Align1Shl60 => 60,
        _Align1Shl61 => 61,
        _Align1Shl62 => 62,
        _Align1Shl63 => 63,
    }
}
