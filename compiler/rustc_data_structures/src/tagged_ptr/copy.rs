use super::{Pointer, Tag};
use crate::stable_hasher::{HashStable, StableHasher};
use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A `Copy` TaggedPtr.
///
/// You should use this instead of the `TaggedPtr` type in all cases where
/// `P: Copy`.
///
/// If `COMPARE_PACKED` is true, then the pointers will be compared and hashed without
/// unpacking. Otherwise we don't implement PartialEq/Eq/Hash; if you want that,
/// wrap the TaggedPtr.
pub struct CopyTaggedPtr<P, T, const COMPARE_PACKED: bool>
where
    P: Pointer,
    T: Tag,
{
    packed: NonNull<P::Target>,
    tag_ghost: PhantomData<T>,
}

impl<P, T, const COMPARE_PACKED: bool> Copy for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
    P: Copy,
{
}

impl<P, T, const COMPARE_PACKED: bool> Clone for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
    P: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

// We pack the tag into the *upper* bits of the pointer to ease retrieval of the
// value; a left shift is a multiplication and those are embeddable in
// instruction encoding.
impl<P, T, const COMPARE_PACKED: bool> CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
{
    pub fn new(pointer: P, tag: T) -> Self {
        Self { packed: Self::pack(P::into_ptr(pointer), tag), tag_ghost: PhantomData }
    }

    const TAG_BIT_SHIFT: usize = usize::BITS as usize - T::BITS;
    const ASSERTION: () = { assert!(T::BITS <= P::BITS) };

    /// Pack pointer `ptr` that comes from [`P::into_ptr`] with a `tag`.
    ///
    /// [`P::into_ptr`]: Pointer::into_ptr
    fn pack(ptr: NonNull<P::Target>, tag: T) -> NonNull<P::Target> {
        // Trigger assert!
        let () = Self::ASSERTION;

        let packed_tag = tag.into_usize() << Self::TAG_BIT_SHIFT;

        ptr.map_addr(|addr| {
            // SAFETY:
            // - The pointer is `NonNull` => it's address is `NonZeroUsize`
            // - `P::BITS` least significant bits are always zero (`Pointer` contract)
            // - `T::BITS <= P::BITS` (from `Self::ASSERTION`)
            //
            // Thus `addr >> T::BITS` is guaranteed to be non-zero.
            //
            // `{non_zero} | packed_tag` can't make the value zero.

            let packed = (addr.get() >> T::BITS) | packed_tag;
            unsafe { NonZeroUsize::new_unchecked(packed) }
        })
    }

    pub(super) fn pointer_raw(&self) -> NonNull<P::Target> {
        self.packed.map_addr(|addr| unsafe { NonZeroUsize::new_unchecked(addr.get() << T::BITS) })
    }

    /// This provides a reference to the `P` pointer itself, rather than the
    /// `Deref::Target`. It is used for cases where we want to call methods
    /// that may be implement differently for the Pointer than the Pointee
    /// (e.g., `Rc::clone` vs cloning the inner value).
    pub(super) fn with_pointer_ref<R>(&self, f: impl FnOnce(&P) -> R) -> R {
        // Safety:
        // - `self.raw.pointer_raw()` is originally returned from `P::into_ptr`
        //   and as such is valid for `P::from_ptr`.
        //   - This also allows us to not care whatever `f` panics or not.
        // - Even though we create a copy of the pointer, we store it inside
        //   `ManuallyDrop` and only access it by-ref, so we don't double-drop.
        //
        // Semantically this is just `f(&self.pointer)` (where `self.pointer`
        // is non-packed original pointer).
        let ptr = unsafe { ManuallyDrop::new(P::from_ptr(self.pointer_raw())) };
        f(&ptr)
    }

    pub fn pointer(self) -> P
    where
        P: Copy,
    {
        // SAFETY: pointer_raw returns the original pointer
        //
        // Note that this isn't going to double-drop or anything because we have
        // P: Copy
        unsafe { P::from_ptr(self.pointer_raw()) }
    }

    pub fn pointer_ref(&self) -> &P::Target {
        // SAFETY: pointer_raw returns the original pointer
        unsafe { self.pointer_raw().as_ref() }
    }

    pub fn pointer_mut(&mut self) -> &mut P::Target
    where
        P: DerefMut,
    {
        // SAFETY: pointer_raw returns the original pointer
        unsafe { self.pointer_raw().as_mut() }
    }

    #[inline]
    pub fn tag(&self) -> T {
        unsafe { T::from_usize(self.packed.addr().get() >> Self::TAG_BIT_SHIFT) }
    }

    #[inline]
    pub fn set_tag(&mut self, tag: T) {
        self.packed = Self::pack(self.pointer_raw(), tag);
    }
}

impl<P, T, const COMPARE_PACKED: bool> Deref for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
{
    type Target = P::Target;

    fn deref(&self) -> &Self::Target {
        self.pointer_ref()
    }
}

impl<P, T, const COMPARE_PACKED: bool> DerefMut for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer + DerefMut,
    T: Tag,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.pointer_mut()
    }
}

impl<P, T, const COMPARE_PACKED: bool> fmt::Debug for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    P::Target: fmt::Debug,
    T: Tag + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyTaggedPtr")
            .field("pointer", &self.pointer_ref())
            .field("tag", &self.tag())
            .finish()
    }
}

impl<P, T> PartialEq for CopyTaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed
    }
}

impl<P, T> Eq for CopyTaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
}

impl<P, T> std::hash::Hash for CopyTaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.packed.hash(state);
    }
}

impl<P, T, HCX, const COMPARE_PACKED: bool> HashStable<HCX> for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer + HashStable<HCX>,
    T: Tag + HashStable<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.with_pointer_ref(|ptr| ptr.hash_stable(hcx, hasher));
        self.tag().hash_stable(hcx, hasher);
    }
}
