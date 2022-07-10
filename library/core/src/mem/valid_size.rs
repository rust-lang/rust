use crate::convert::TryFrom;
use crate::{fmt, num};

/// A type storing a possible object size (in bytes) in the rust abstract machine.
///
/// This can be thought of as a positive `isize`, or `usize` without the high bit
/// set.  This is important because [`pointer::offset`] is UB for *byte* sizes
/// too large for an `isize`, and there's a corresponding language limit on the
/// size of any allocated object.
///
/// Note that particularly large sizes, while representable in this type, are
/// likely not to be supported by actual allocators and machines.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
#[repr(transparent)]
#[cfg_attr(target_pointer_width = "16", rustc_layout_scalar_valid_range_end(0x7FFF))]
#[cfg_attr(target_pointer_width = "32", rustc_layout_scalar_valid_range_end(0x7FFF_FFFF))]
#[cfg_attr(target_pointer_width = "64", rustc_layout_scalar_valid_range_end(0x7FFF_FFFF_FFFF_FFFF))]
pub(crate) struct ValidSize(usize);

const MAX_SIZE: usize = isize::MAX as usize;

const _: () = unsafe { ValidSize::new_unchecked(MAX_SIZE); };

impl ValidSize {
    /// Creates a `ValidSize` from a `usize` that fits in an `isize`.
    ///
    /// # Safety
    ///
    /// `size` must be less than or equal to `isize::MAX`.
    ///
    /// Equivalently, it must not have its high bit set.
    #[inline]
    pub(crate) const unsafe fn new_unchecked(size: usize) -> Self {
        debug_assert!(size <= MAX_SIZE);

        // SAFETY: By precondition, this must be within our validity invariant.
        unsafe { ValidSize(size) }
    }

    #[inline]
    pub(crate) const fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ValidSize {
    type Error = num::TryFromIntError;

    #[inline]
    fn try_from(size: usize) -> Result<ValidSize, Self::Error> {
        if size <= MAX_SIZE {
            // SAFETY: Just checked it's within our validity invariant.
            unsafe { Ok(ValidSize(size)) }
        } else {
            Err(num::TryFromIntError(()))
        }
    }
}

impl fmt::Debug for ValidSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_usize().fmt(f)
    }
}
