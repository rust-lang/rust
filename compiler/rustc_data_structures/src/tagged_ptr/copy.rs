use super::{Pointer, Tag};
use crate::stable_hasher::{HashStable, StableHasher};
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

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
    packed: NonZeroUsize,
    data: PhantomData<(P, T)>,
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
    const TAG_BIT_SHIFT: usize = usize::BITS as usize - T::BITS;
    const ASSERTION: () = {
        assert!(T::BITS <= P::BITS);
        // Used for the transmute_copy's below
        assert!(std::mem::size_of::<&P::Target>() == std::mem::size_of::<usize>());
    };

    pub fn new(pointer: P, tag: T) -> Self {
        // Trigger assert!
        let () = Self::ASSERTION;
        let packed_tag = tag.into_usize() << Self::TAG_BIT_SHIFT;

        Self {
            // SAFETY: We know that the pointer is non-null, as it must be
            // dereferenceable per `Pointer` safety contract.
            packed: unsafe {
                NonZeroUsize::new_unchecked((P::into_usize(pointer) >> T::BITS) | packed_tag)
            },
            data: PhantomData,
        }
    }

    pub(super) fn pointer_raw(&self) -> usize {
        self.packed.get() << T::BITS
    }
    pub fn pointer(self) -> P
    where
        P: Copy,
    {
        // SAFETY: pointer_raw returns the original pointer
        //
        // Note that this isn't going to double-drop or anything because we have
        // P: Copy
        unsafe { P::from_usize(self.pointer_raw()) }
    }
    pub fn pointer_ref(&self) -> &P::Target {
        // SAFETY: pointer_raw returns the original pointer
        unsafe { std::mem::transmute_copy(&self.pointer_raw()) }
    }
    pub fn pointer_mut(&mut self) -> &mut P::Target
    where
        P: std::ops::DerefMut,
    {
        // SAFETY: pointer_raw returns the original pointer
        unsafe { std::mem::transmute_copy(&self.pointer_raw()) }
    }
    #[inline]
    pub fn tag(&self) -> T {
        unsafe { T::from_usize(self.packed.get() >> Self::TAG_BIT_SHIFT) }
    }
    #[inline]
    pub fn set_tag(&mut self, tag: T) {
        let mut packed = self.packed.get();
        let new_tag = T::into_usize(tag) << Self::TAG_BIT_SHIFT;
        let tag_mask = (1 << T::BITS) - 1;
        packed &= !(tag_mask << Self::TAG_BIT_SHIFT);
        packed |= new_tag;
        self.packed = unsafe { NonZeroUsize::new_unchecked(packed) };
    }
}

impl<P, T, const COMPARE_PACKED: bool> std::ops::Deref for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
{
    type Target = P::Target;
    fn deref(&self) -> &Self::Target {
        self.pointer_ref()
    }
}

impl<P, T, const COMPARE_PACKED: bool> std::ops::DerefMut for CopyTaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer + std::ops::DerefMut,
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
        unsafe {
            Pointer::with_ref(self.pointer_raw(), |p: &P| p.hash_stable(hcx, hasher));
        }
        self.tag().hash_stable(hcx, hasher);
    }
}
