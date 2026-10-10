use core::marker::Unsize;
use core::num::NonZero;
use core::ops::{CoerceUnsized, DispatchFromDyn};
use core::ptr::{NonNull, Pointee};

/// An non-null pointer that either:
///
/// - Points to the value storage for `T` inside a referenced counted allocation.
/// - Is a dangling pointer used in `Weak`.
#[repr(transparent)]
pub(in crate::rcs) struct RcValuePointer<T>
where
    T: ?Sized,
{
    pub(in crate::rcs) ptr: NonNull<T>,
}

impl<T> RcValuePointer<T>
where
    T: ?Sized,
{
    pub(in crate::rcs) const fn new(ptr: NonNull<T>) -> Self {
        Self { ptr }
    }

    pub(in crate::rcs) fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(in crate::rcs) fn cast<U>(self) -> RcValuePointer<U> {
        RcValuePointer::new(self.ptr.cast())
    }

    pub(in crate::rcs) fn erase(self) -> ErasedRcValuePointer {
        ErasedRcValuePointer::new(self.ptr.cast())
    }

    #[inline]
    pub(in crate::rcs) fn is_dangling(self) -> bool {
        self.erase().is_dangling()
    }
}

impl<T> RcValuePointer<T> {
    pub(in crate::rcs) const DANGLING: Self = ErasedRcValuePointer::DANGLING.unerase();
}

impl<T> Clone for RcValuePointer<T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, U> CoerceUnsized<RcValuePointer<U>> for RcValuePointer<T>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

impl<T> Copy for RcValuePointer<T> where T: ?Sized {}

impl<T, U> DispatchFromDyn<RcValuePointer<U>> for RcValuePointer<T>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

/// An non-null pointer that points to the value storage for an unspecified type inside a referenced
/// counted allocation.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(in crate::rcs) struct ErasedRcValuePointer {
    pub(in crate::rcs) ptr: NonNull<()>,
}

impl ErasedRcValuePointer {
    const DANGLING: Self = Self::new(NonNull::without_provenance(NonZero::<usize>::MAX));

    pub(in crate::rcs) const fn new(ptr: NonNull<()>) -> Self {
        Self { ptr }
    }

    #[inline]
    fn is_dangling(self) -> bool {
        self.ptr.addr().get() == usize::MAX
    }

    pub(in crate::rcs) const fn unerase<T>(self) -> RcValuePointer<T> {
        RcValuePointer::new(self.ptr.cast())
    }

    pub(in crate::rcs) const fn unerase_with<T>(
        self,
        metadata: <T as Pointee>::Metadata,
    ) -> RcValuePointer<T>
    where
        T: ?Sized,
    {
        RcValuePointer::new(NonNull::from_raw_parts(self.ptr, metadata))
    }
}
