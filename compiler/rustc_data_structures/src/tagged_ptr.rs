//! This module implements tagged pointers.
//!
//! In order to utilize the pointer packing, you must have two types: a pointer,
//! and a tag.
//!
//! The pointer must implement the [`Pointer`] trait, with the primary
//! requirement being convertible to and from a raw pointer. Note that the
//! pointer must be dereferenceable, so raw pointers generally cannot implement
//! the [`Pointer`] trait. This implies that the pointer must also be non-null.
//!
//! Many common pointer types already implement the [`Pointer`] trait.
//!
//! The tag must implement the [`Tag`] trait.
//!
//! We assert that the tag and the [`Pointer`] types are compatible at compile
//! time.

use std::ops::Deref;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

use crate::aligned::Aligned;

mod copy;
mod drop;

pub use copy::CopyTaggedPtr;
pub use drop::TaggedPtr;

/// This describes the pointer type encapsulated by [`TaggedPtr`] and
/// [`CopyTaggedPtr`].
///
/// # Safety
///
/// The pointer returned from [`into_ptr`] must be a [valid], pointer to
/// [`<Self as Deref>::Target`].
///
/// Note that if `Self` implements [`DerefMut`] the pointer returned from
/// [`into_ptr`] must be valid for writes (and thus calling [`NonNull::as_mut`]
/// on it must be safe).
///
/// The [`BITS`] constant must be correct. [`BITS`] least-significant bits,
/// must be zero on all pointers returned from [`into_ptr`].
///
/// For example, if the alignment of [`Self::Target`] is 2, then `BITS` should be 1.
///
/// [`BITS`]: Pointer::BITS
/// [`into_ptr`]: Pointer::into_ptr
/// [valid]: std::ptr#safety
/// [`<Self as Deref>::Target`]: Deref::Target
/// [`Self::Target`]: Deref::Target
/// [`DerefMut`]: std::ops::DerefMut
pub unsafe trait Pointer: Deref {
    /// Number of unused (always zero) **least-significant bits** in this
    /// pointer, usually related to the pointees alignment.
    ///
    /// For example if [`BITS`] = `2`, then given `ptr = Self::into_ptr(..)`,
    /// `ptr.addr() & 0b11 == 0` must be true.
    ///
    /// Most likely the value you want to use here is the following, unless
    /// your [`Self::Target`] type is unsized (e.g., `ty::List<T>` in rustc)
    /// or your pointer is over/under aligned, in which case you'll need to
    /// manually figure out what the right type to pass to [`bits_for`] is, or
    /// what the value to set here.
    ///
    /// ```rust
    /// # use std::ops::Deref;
    /// # use rustc_data_structures::tagged_ptr::bits_for;
    /// # struct T;
    /// # impl Deref for T { type Target = u8; fn deref(&self) -> &u8 { &0 } }
    /// # impl T {
    /// const BITS: u32 = bits_for::<<Self as Deref>::Target>();
    /// # }
    /// ```
    ///
    /// [`BITS`]: Pointer::BITS
    /// [`Self::Target`]: Deref::Target
    const BITS: u32;

    /// Turns this pointer into a raw, non-null pointer.
    ///
    /// The inverse of this function is [`from_ptr`].
    ///
    /// This function guarantees that the least-significant [`Self::BITS`] bits
    /// are zero.
    ///
    /// [`from_ptr`]: Pointer::from_ptr
    /// [`Self::BITS`]: Pointer::BITS
    fn into_ptr(self) -> NonNull<Self::Target>;

    /// Re-creates the original pointer, from a raw pointer returned by [`into_ptr`].
    ///
    /// # Safety
    ///
    /// The passed `ptr` must be returned from [`into_ptr`].
    ///
    /// This acts as [`ptr::read::<Self>()`] semantically, it should not be called more than
    /// once on non-[`Copy`] `Pointer`s.
    ///
    /// [`into_ptr`]: Pointer::into_ptr
    /// [`ptr::read::<Self>()`]: std::ptr::read
    unsafe fn from_ptr(ptr: NonNull<Self::Target>) -> Self;
}

/// This describes tags that the [`TaggedPtr`] struct can hold.
///
/// # Safety
///
/// The [`BITS`] constant must be correct.
///
/// No more than [`BITS`] least-significant bits may be set in the returned usize.
///
/// [`BITS`]: Tag::BITS
pub unsafe trait Tag: Copy {
    /// Number of least-significant bits in the return value of [`into_usize`]
    /// which may be non-zero. In other words this is the bit width of the
    /// value.
    ///
    /// [`into_usize`]: Tag::into_usize
    const BITS: u32;

    /// Turns this tag into an integer.
    ///
    /// The inverse of this function is [`from_usize`].
    ///
    /// This function guarantees that only the least-significant [`Self::BITS`]
    /// bits can be non-zero.
    ///
    /// [`from_usize`]: Tag::from_usize
    /// [`Self::BITS`]: Tag::BITS
    fn into_usize(self) -> usize;

    /// Re-creates the tag from the integer returned by [`into_usize`].
    ///
    /// # Safety
    ///
    /// The passed `tag` must be returned from [`into_usize`].
    ///
    /// [`into_usize`]: Tag::into_usize
    unsafe fn from_usize(tag: usize) -> Self;
}

unsafe impl<T: ?Sized + Aligned> Pointer for Box<T> {
    const BITS: u32 = bits_for::<Self::Target>();

    #[inline]
    fn into_ptr(self) -> NonNull<T> {
        // Safety: pointers from `Box::into_raw` are valid & non-null
        unsafe { NonNull::new_unchecked(Box::into_raw(self)) }
    }

    #[inline]
    unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        // Safety: `ptr` comes from `into_ptr` which calls `Box::into_raw`
        unsafe { Box::from_raw(ptr.as_ptr()) }
    }
}

unsafe impl<T: ?Sized + Aligned> Pointer for Rc<T> {
    const BITS: u32 = bits_for::<Self::Target>();

    #[inline]
    fn into_ptr(self) -> NonNull<T> {
        // Safety: pointers from `Rc::into_raw` are valid & non-null
        unsafe { NonNull::new_unchecked(Rc::into_raw(self).cast_mut()) }
    }

    #[inline]
    unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        // Safety: `ptr` comes from `into_ptr` which calls `Rc::into_raw`
        unsafe { Rc::from_raw(ptr.as_ptr()) }
    }
}

unsafe impl<T: ?Sized + Aligned> Pointer for Arc<T> {
    const BITS: u32 = bits_for::<Self::Target>();

    #[inline]
    fn into_ptr(self) -> NonNull<T> {
        // Safety: pointers from `Arc::into_raw` are valid & non-null
        unsafe { NonNull::new_unchecked(Arc::into_raw(self).cast_mut()) }
    }

    #[inline]
    unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        // Safety: `ptr` comes from `into_ptr` which calls `Arc::into_raw`
        unsafe { Arc::from_raw(ptr.as_ptr()) }
    }
}

unsafe impl<'a, T: 'a + ?Sized + Aligned> Pointer for &'a T {
    const BITS: u32 = bits_for::<Self::Target>();

    #[inline]
    fn into_ptr(self) -> NonNull<T> {
        NonNull::from(self)
    }

    #[inline]
    unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        // Safety:
        // `ptr` comes from `into_ptr` which gets the pointer from a reference
        unsafe { ptr.as_ref() }
    }
}

unsafe impl<'a, T: 'a + ?Sized + Aligned> Pointer for &'a mut T {
    const BITS: u32 = bits_for::<Self::Target>();

    #[inline]
    fn into_ptr(self) -> NonNull<T> {
        NonNull::from(self)
    }

    #[inline]
    unsafe fn from_ptr(mut ptr: NonNull<T>) -> Self {
        // Safety:
        // `ptr` comes from `into_ptr` which gets the pointer from a reference
        unsafe { ptr.as_mut() }
    }
}

/// Returns the number of bits available for use for tags in a pointer to `T`
/// (this is based on `T`'s alignment).
pub const fn bits_for<T: ?Sized + Aligned>() -> u32 {
    crate::aligned::align_of::<T>().as_nonzero().trailing_zeros()
}

/// A tag type used in [`CopyTaggedPtr`] and [`TaggedPtr`] tests.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
enum Tag2 {
    B00 = 0b00,
    B01 = 0b01,
    B10 = 0b10,
    B11 = 0b11,
}

#[cfg(test)]
unsafe impl Tag for Tag2 {
    const BITS: u32 = 2;

    fn into_usize(self) -> usize {
        self as _
    }

    unsafe fn from_usize(tag: usize) -> Self {
        match tag {
            0b00 => Tag2::B00,
            0b01 => Tag2::B01,
            0b10 => Tag2::B10,
            0b11 => Tag2::B11,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
impl<HCX> crate::stable_hasher::HashStable<HCX> for Tag2 {
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut crate::stable_hasher::StableHasher) {
        (*self as u8).hash_stable(hcx, hasher);
    }
}
