use core::cell::UnsafeCell;
use core::ptr::NonNull;

use crate::raw_rc::RefCounts;
use crate::raw_rc::rc_layout::RcLayout;

/// A pointer to the value location in a reference-counted allocation. The reference-counted
/// allocation may be deallocated, and the value may be uninitialized. This type provides stronger
/// pointer semantics, reducing the risk of misuse. The guarantees this type provides can also
/// reduce the amount of unsafe code.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct RcValuePointer {
    inner: NonNull<()>,
}

impl RcValuePointer {
    /// Creates a new `RcValuePointer` from a raw pointer to a reference-counted allocation.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `allocation_ptr` points to a valid reference-counted allocation that
    /// can be described by `rc_layout`.
    #[inline]
    pub(crate) unsafe fn from_allocation_ptr(
        allocation_ptr: NonNull<()>,
        rc_layout: RcLayout,
    ) -> Self {
        // SAFETY: Caller guarantees that `allocation_ptr` points to some reference-counted
        // allocation that can be described by `rc_layout`, so we can acquire the corresponding
        // value pointer safely.
        unsafe { Self::from_value_ptr(allocation_ptr.byte_add(rc_layout.value_offset())) }
    }

    /// Creates a new `RcValuePointer` from a raw pointer to the value location in a
    /// reference-counted allocation.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `value_ptr` is a valid pointer to the value location inside some
    /// valid reference-counted allocation.
    #[inline]
    pub(crate) unsafe fn from_value_ptr(value_ptr: NonNull<()>) -> Self {
        Self { inner: value_ptr }
    }

    #[inline]
    pub(crate) fn as_ptr(self) -> NonNull<()> {
        self.inner
    }

    #[inline]
    pub(crate) fn ref_counts_ptr(self) -> NonNull<RefCounts> {
        // SAFETY: `self.inner` is guaranteed to have a valid address inside a reference-counted
        // allocation, so we can safely obtain a pointer to the corresponding `RefCounts` object.
        unsafe { self.inner.cast::<RefCounts>().sub(1) }
    }

    #[inline]
    pub(crate) fn strong_count_ptr(self) -> NonNull<UnsafeCell<usize>> {
        let ref_counts_ptr = self.ref_counts_ptr();

        // SAFETY: `ref_counts_ptr` is guaranteed to be a valid pointer to a `RefCounts` object, so
        // we can safely obtain the pointer to the corresponding strong counter object.
        unsafe { NonNull::new_unchecked(&raw mut (*ref_counts_ptr.as_ptr()).strong) }
    }

    #[inline]
    pub(crate) fn weak_count_ptr(self) -> NonNull<UnsafeCell<usize>> {
        let ref_counts_ptr = self.ref_counts_ptr();

        // SAFETY: `ref_counts_ptr` is guaranteed to be a valid pointer to a `RefCounts` object, so
        // we can safely obtain the pointer to the corresponding weak counter object.
        unsafe { NonNull::new_unchecked(&raw mut (*ref_counts_ptr.as_ptr()).weak) }
    }
}
