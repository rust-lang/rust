//! This module implements tagged pointers.
//!
//! In order to utilize the pointer packing, you must have two types: a pointer,
//! and a tag.
//!
//! The pointer must implement the `Pointer` trait, with the primary requirement
//! being conversion to and from a usize. Note that the pointer must be
//! dereferenceable, so raw pointers generally cannot implement the `Pointer`
//! trait. This implies that the pointer must also be nonzero.
//!
//! Many common pointer types already implement the `Pointer` trait.
//!
//! The tag must implement the `Tag` trait. We assert that the tag and `Pointer`
//! are compatible at compile time.

use std::mem::{self, ManuallyDrop};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

mod copy;
mod drop;

pub use copy::CopyTaggedPtr;
pub use drop::TaggedPtr;

/// This describes the pointer type encapsulated by [`TaggedPtr`] and
/// [`CopyTaggedPtr`].
///
/// # Safety
///
/// The usize returned from `into_usize` must be a valid, dereferenceable,
/// pointer to [`<Self as Deref>::Target`]. Note that pointers to
/// [`Self::Target`] must be thin, even though [`Self::Target`] may not be
/// `Sized`.
///
/// Note that the returned pointer from `into_usize` should be castable to `&mut
/// <Self as Deref>::Target` if `Self: DerefMut`.
///
/// The BITS constant must be correct. At least `BITS` bits, least-significant,
/// must be zero on all returned pointers from `into_usize`.
///
/// For example, if the alignment of [`Self::Target`] is 2, then `BITS` should be 1.
///
/// [`<Self as Deref>::Target`]: Deref::Target
/// [`Self::Target`]: Deref::Target
pub unsafe trait Pointer: Deref {
    /// Number of unused (always zero) **least significant bits** in this
    /// pointer, usually related to the pointees alignment.
    ///
    /// Most likely the value you want to use here is the following, unless
    /// your [`Self::Target`] type is unsized (e.g., `ty::List<T>` in rustc)
    /// or your pointer is over/under aligned, in which case you'll need to
    /// manually figure out what the right type to pass to [`bits_for`] is, or
    /// what the value to set here.
    ///
    /// ```rust
    /// # use std::ops::Deref;
    /// # type Self = &'static u64;
    /// bits_for::<Self::Target>()
    /// ```
    ///
    /// [`Self::Target`]: Deref::Target
    const BITS: usize;

    fn into_usize(self) -> usize;

    /// # Safety
    ///
    /// The passed `ptr` must be returned from `into_usize`.
    ///
    /// This acts as `ptr::read` semantically, it should not be called more than
    /// once on non-`Copy` `Pointer`s.
    unsafe fn from_usize(ptr: usize) -> Self;

    /// This provides a reference to the `Pointer` itself, rather than the
    /// `Deref::Target`. It is used for cases where we want to call methods that
    /// may be implement differently for the Pointer than the Pointee (e.g.,
    /// `Rc::clone` vs cloning the inner value).
    ///
    /// # Safety
    ///
    /// The passed `ptr` must be returned from `into_usize`.
    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R;
}

/// This describes tags that the `TaggedPtr` struct can hold.
///
/// # Safety
///
/// The BITS constant must be correct.
///
/// No more than `BITS` least significant bits may be set in the returned usize.
pub unsafe trait Tag: Copy {
    const BITS: usize;

    fn into_usize(self) -> usize;

    /// # Safety
    ///
    /// The passed `tag` must be returned from `into_usize`.
    unsafe fn from_usize(tag: usize) -> Self;
}

unsafe impl<T> Pointer for Box<T> {
    const BITS: usize = bits_for::<Self::Target>();

    #[inline]
    fn into_usize(self) -> usize {
        Box::into_raw(self) as usize
    }

    #[inline]
    unsafe fn from_usize(ptr: usize) -> Self {
        Box::from_raw(ptr as *mut T)
    }

    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        let raw = ManuallyDrop::new(Self::from_usize(ptr));
        f(&raw)
    }
}

unsafe impl<T> Pointer for Rc<T> {
    const BITS: usize = bits_for::<Self::Target>();

    #[inline]
    fn into_usize(self) -> usize {
        Rc::into_raw(self) as usize
    }

    #[inline]
    unsafe fn from_usize(ptr: usize) -> Self {
        Rc::from_raw(ptr as *const T)
    }

    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        let raw = ManuallyDrop::new(Self::from_usize(ptr));
        f(&raw)
    }
}

unsafe impl<T> Pointer for Arc<T> {
    const BITS: usize = bits_for::<Self::Target>();

    #[inline]
    fn into_usize(self) -> usize {
        Arc::into_raw(self) as usize
    }

    #[inline]
    unsafe fn from_usize(ptr: usize) -> Self {
        Arc::from_raw(ptr as *const T)
    }

    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        let raw = ManuallyDrop::new(Self::from_usize(ptr));
        f(&raw)
    }
}

unsafe impl<'a, T: 'a> Pointer for &'a T {
    const BITS: usize = bits_for::<Self::Target>();

    #[inline]
    fn into_usize(self) -> usize {
        self as *const T as usize
    }

    #[inline]
    unsafe fn from_usize(ptr: usize) -> Self {
        &*(ptr as *const T)
    }

    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        f(&*(&ptr as *const usize as *const Self))
    }
}

unsafe impl<'a, T: 'a> Pointer for &'a mut T {
    const BITS: usize = bits_for::<Self::Target>();
    #[inline]
    fn into_usize(self) -> usize {
        self as *mut T as usize
    }
    #[inline]
    unsafe fn from_usize(ptr: usize) -> Self {
        &mut *(ptr as *mut T)
    }
    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        f(&*(&ptr as *const usize as *const Self))
    }
}

/// Returns the number of bits available for use for tags in a pointer to `T`
/// (this is based on `T`'s alignment).
pub const fn bits_for<T>() -> usize {
    let bits = mem::align_of::<T>().trailing_zeros();

    // This is a replacement for `.try_into().unwrap()` unavailable in `const`
    // (it's fine to make an assert here, since this is only called in compile time)
    assert!((bits as u128) < usize::MAX as u128);

    bits as usize
}
