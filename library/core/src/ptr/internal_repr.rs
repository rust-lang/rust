//! This encapsulates the layout knowledge for pointers, only exporting the two
//! safe functions that can be used to interact with the metadata directly.

use super::{NonNull, Pointee, Thin};

#[inline]
pub(super) const fn metadata<P: RawPointer>(ptr: P) -> <P::Pointee as Pointee>::Metadata {
    // SAFETY: Transmuting like this is safe since `P` and `PtrComponents`
    // have the same memory layouts. Only std can make this guarantee.
    unsafe {
        crate::intrinsics::transmute_unchecked::<
            P,
            PtrComponents<P::Family, <P::Pointee as Pointee>::Metadata>,
        >(ptr)
        .metadata
    }
}

/// Just like [`super::from_raw_parts`] and [`super::from_raw_parts_mut`], but
/// more flexible in terms of which types it can take, allowing smaller MIR.
// See <https://github.com/rust-lang/rust/issues/123174>
#[rustc_const_unstable(feature = "ptr_metadata", issue = "81513")]
#[inline]
pub(super) const fn from_raw_parts<P: RawPointer>(
    data_pointer: impl RawPointer<Pointee: Thin, Family = P::Family>,
    metadata: <P::Pointee as Pointee>::Metadata,
) -> P {
    // SAFETY: Transmuting like this is safe since `P` and `PtrComponents`
    // have the same memory layouts. Only std can make this guarantee.
    unsafe {
        crate::intrinsics::transmute_unchecked::<
            PtrComponents<_, <P::Pointee as Pointee>::Metadata>,
            P,
        >(PtrComponents { data_pointer, metadata })
    }
}

// Intentionally private with no derives, as it's only used via transmuting.
// This layout is not stable; only std can rely on it.
// (And should only do so in the two functions in this module.)
#[repr(C)]
struct PtrComponents<P, M = ()> {
    data_pointer: P,
    metadata: M,
}

/// Internal trait to avoid bad instantiations of [`PtrComponents`]
///
/// # Safety
///
/// Must have the same layout as `*const Self::Pointee` and be able to hold provenance.
///
/// Every type with the same associated `Family` must be soundly transmutable
/// between each other when the metadata is the same.
pub unsafe trait RawPointer: Copy {
    type Pointee: ?Sized + super::Pointee;
    type Family: RawPointer<Pointee = (), Family = Self::Family>;
}

// SAFETY: `*const T` is obviously a raw pointer
unsafe impl<T: ?Sized> RawPointer for *const T {
    type Pointee = T;
    type Family = *const ();
}
// SAFETY: `*mut T` is obviously a raw pointer
unsafe impl<T: ?Sized> RawPointer for *mut T {
    type Pointee = T;
    type Family = *mut ();
}
// SAFETY: `NonNull<T>` is a transparent newtype around a `*const T`.
unsafe impl<T: ?Sized> RawPointer for NonNull<T> {
    type Pointee = T;
    type Family = NonNull<()>;
}
