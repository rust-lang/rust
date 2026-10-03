use core::marker::{PhantomData, Unsize};
use core::mem::SizedTypeProperties;
use core::num::NonZero;
use core::ops::{CoerceUnsized, DispatchFromDyn};
use core::ptr::{NonNull, Pointee};

use crate::rcs::common::rc_layout;

/// An non-null pointer that either:
///
/// - Points to the value storage for `T` inside a referenced counted allocation.
/// - Is a dangling pointer used in `Weak`.
#[repr(transparent)]
pub(in crate::rcs) struct RcValuePointer<H, T>
where
    T: ?Sized,
{
    pub(in crate::rcs) ptr: NonNull<T>,
    _marker: PhantomData<H>,
}

impl<H, T> RcValuePointer<H, T>
where
    T: ?Sized,
{
    pub(in crate::rcs) const fn new(ptr: NonNull<T>) -> Self {
        Self { ptr, _marker: PhantomData }
    }

    pub(in crate::rcs) fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(in crate::rcs) fn cast<U>(self) -> RcValuePointer<H, U> {
        RcValuePointer::new(self.ptr.cast())
    }

    pub(in crate::rcs) fn erase(self) -> ErasedRcValuePointer<H> {
        ErasedRcValuePointer::new(self.ptr.cast())
    }

    /// # Safety
    ///
    /// - `self` points to a reference counted allocation with `H` as header.
    /// - The header is properly initialized.
    /// - The allocation outlives `'a`.
    pub(in crate::rcs) unsafe fn header<'a>(self) -> &'a H {
        // SAFETY: Guaranteed by caller.
        unsafe { self.erase().header() }
    }

    /// # Safety
    ///
    /// - `self` points to a reference counted allocation with `H` as header.
    pub(in crate::rcs) unsafe fn header_ptr(self) -> NonNull<H> {
        // SAFETY: Caller guarantees the validify of `self`.
        unsafe { self.erase().header_ptr() }
    }

    pub(in crate::rcs) fn is_dangling(self) -> bool {
        self.erase().is_dangling()
    }
}

impl<H, T> RcValuePointer<H, T> {
    pub(in crate::rcs) const DANGLING: Self = ErasedRcValuePointer::DANGLING.unerase();
}

impl<H, T> Clone for RcValuePointer<H, T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<H, T, U> CoerceUnsized<RcValuePointer<H, U>> for RcValuePointer<H, T>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

impl<H, T> Copy for RcValuePointer<H, T> where T: ?Sized {}

impl<H, T, U> DispatchFromDyn<RcValuePointer<H, U>> for RcValuePointer<H, T>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

/// An non-null pointer that points to the value storage for an unspecified type inside a referenced
/// counted allocation.
#[repr(transparent)]
pub(in crate::rcs) struct ErasedRcValuePointer<H> {
    pub(in crate::rcs) ptr: NonNull<()>,
    _marker: PhantomData<H>,
}

impl<H> ErasedRcValuePointer<H> {
    const DANGLING: Self = {
        // This check ensures the value is at least aligned to an even number so that `usize::MAX`
        // can be used as the address of a dangling pointer.
        assert!(H::ALIGN > 1);

        Self::new(NonNull::without_provenance(NonZero::<usize>::MAX))
    };

    pub(in crate::rcs) const fn new(ptr: NonNull<()>) -> Self {
        Self { ptr, _marker: PhantomData }
    }

    /// # Safety
    ///
    /// `allocation_ptr` points to a reference counted allocation for storing value with alignment
    /// of `value_alignment`.
    pub(super) unsafe fn from_allocation_ptr(
        allocation_ptr: NonNull<()>,
        value_alignment: usize,
    ) -> Self {
        // SAFETY: Caller guarantees the validity of `allocation_ptr` and `value_alignment`, we are
        // safe to acquire a pointer to the value storage.
        Self::new(unsafe { allocation_ptr.byte_add(rc_layout::value_offset::<H>(value_alignment)) })
    }

    /// # Safety
    ///
    /// - `self` points to a reference counted allocation with `H` as header.
    /// - The header is properly initialized.
    /// - The allocation outlives `'a`.
    pub(in crate::rcs) unsafe fn header<'a>(self) -> &'a H {
        // SAFETY: Guaranteed by caller.
        unsafe { self.header_ptr().as_ref() }
    }

    /// # Safety
    ///
    /// - `self` points to a reference counted allocation with `H` as header.
    unsafe fn header_ptr(self) -> NonNull<H> {
        // SAFETY: Caller guarantees the validify of `self`.
        unsafe { self.ptr.byte_sub(size_of::<H>()).cast() }
    }

    fn is_dangling(self) -> bool {
        self.ptr.addr().get() == usize::MAX
    }

    pub(in crate::rcs) const fn unerase<T>(self) -> RcValuePointer<H, T> {
        RcValuePointer::new(self.ptr.cast())
    }

    pub(in crate::rcs) const fn unerase_with<T>(
        self,
        metadata: <T as Pointee>::Metadata,
    ) -> RcValuePointer<H, T>
    where
        T: ?Sized,
    {
        RcValuePointer::new(NonNull::from_raw_parts(self.ptr, metadata))
    }
}

impl<H> Clone for ErasedRcValuePointer<H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H> Copy for ErasedRcValuePointer<H> {}
