use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

use super::CopyTaggedPtr;
use super::{Pointer, Tag};
use crate::stable_hasher::{HashStable, StableHasher};

/// A TaggedPtr implementing `Drop`.
///
/// If `COMPARE_PACKED` is true, then the pointers will be compared and hashed without
/// unpacking. Otherwise we don't implement PartialEq/Eq/Hash; if you want that,
/// wrap the TaggedPtr.
pub struct TaggedPtr<P, T, const COMPARE_PACKED: bool>
where
    P: Pointer,
    T: Tag,
{
    raw: CopyTaggedPtr<P, T, COMPARE_PACKED>,
}

// We pack the tag into the *upper* bits of the pointer to ease retrieval of the
// value; a right shift is a multiplication and those are embeddable in
// instruction encoding.
impl<P, T, const CP: bool> TaggedPtr<P, T, CP>
where
    P: Pointer,
    T: Tag,
{
    pub fn new(pointer: P, tag: T) -> Self {
        TaggedPtr { raw: CopyTaggedPtr::new(pointer, tag) }
    }

    pub fn tag(&self) -> T {
        self.raw.tag()
    }

    pub fn set_tag(&mut self, tag: T) {
        self.raw.set_tag(tag)
    }
}

impl<P, T, const CP: bool> Clone for TaggedPtr<P, T, CP>
where
    P: Pointer + Clone,
    T: Tag,
{
    fn clone(&self) -> Self {
        let ptr = self.raw.with_pointer_ref(P::clone);

        Self::new(ptr, self.tag())
    }
}

impl<P, T, const CP: bool> Deref for TaggedPtr<P, T, CP>
where
    P: Pointer,
    T: Tag,
{
    type Target = P::Target;
    fn deref(&self) -> &Self::Target {
        self.raw.deref()
    }
}

impl<P, T, const CP: bool> DerefMut for TaggedPtr<P, T, CP>
where
    P: Pointer + DerefMut,
    T: Tag,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.raw.deref_mut()
    }
}

impl<P, T, const CP: bool> Drop for TaggedPtr<P, T, CP>
where
    P: Pointer,
    T: Tag,
{
    fn drop(&mut self) {
        // No need to drop the tag, as it's Copy
        unsafe {
            drop(P::from_ptr(self.raw.pointer_raw()));
        }
    }
}

impl<P, T, const CP: bool> fmt::Debug for TaggedPtr<P, T, CP>
where
    P: Pointer + fmt::Debug,
    T: Tag + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.raw.with_pointer_ref(|ptr| {
            f.debug_struct("TaggedPtr").field("pointer", ptr).field("tag", &self.tag()).finish()
        })
    }
}

impl<P, T> PartialEq for TaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn eq(&self, other: &Self) -> bool {
        self.raw.eq(&other.raw)
    }
}

impl<P, T> Eq for TaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
}

impl<P, T> Hash for TaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<P, T, HCX, const CP: bool> HashStable<HCX> for TaggedPtr<P, T, CP>
where
    P: Pointer + HashStable<HCX>,
    T: Tag + HashStable<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.raw.hash_stable(hcx, hasher);
    }
}
