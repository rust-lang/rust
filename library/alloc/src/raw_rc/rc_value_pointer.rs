use core::cell::UnsafeCell;
use core::ptr::NonNull;

use crate::raw_rc::RefCounts;

/// A pointer to the value location in a reference-counted allocation. The reference-counted
/// allocation is allowed to be deallocated, and the value is allowed to be uninitialized.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct RcValuePointer {
    inner: NonNull<()>,
}

impl RcValuePointer {
    /// Creates a new `RcValuePointer` from a raw pointer.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `ptr` is a valid pointer to the value location inside some valid
    /// reference-counted allocation.
    #[inline]
    pub(crate) unsafe fn new(ptr: NonNull<()>) -> Self {
        Self { inner: ptr }
    }

    #[inline]
    pub(crate) fn as_ptr(self) -> NonNull<()> {
        self.inner
    }

    #[inline]
    pub(crate) fn ref_counts_ptr(self) -> NonNull<RefCounts> {
        // SAFETY: `self.inner` is guaranteed to have a valid address inside a reference-counted
        // allocation, so we are safe to assume we can get a proper pointer to the corresponding
        // `RefCounts` object.
        unsafe { self.inner.cast::<RefCounts>().sub(1) }
    }

    #[inline]
    pub(crate) fn strong_count_ptr(self) -> NonNull<UnsafeCell<usize>> {
        let ref_counts_ptr = self.ref_counts_ptr();

        // SAFETY: `ref_counts_ptr` is guaranteed to be a valid pointer to a `RefCounts` object, so
        // we can safely acquire the pointer to the corresponding strong counter object.
        unsafe { NonNull::new_unchecked(&raw mut (*ref_counts_ptr.as_ptr()).strong) }
    }

    #[inline]
    pub(crate) fn weak_count_ptr(self) -> NonNull<UnsafeCell<usize>> {
        let ref_counts_ptr = self.ref_counts_ptr();

        // SAFETY: `ref_counts_ptr` is guaranteed to be a valid pointer to a `RefCounts` object, so
        // we can safely acquire the pointer to the corresponding weak counter object.
        unsafe { NonNull::new_unchecked(&raw mut (*ref_counts_ptr.as_ptr()).weak) }
    }
}
