use super::{Pointer, Tag};
use crate::stable_hasher::{HashStable, StableHasher};
use std::fmt;

use super::CopyTaggedPtr;

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

impl<P, T, const COMPARE_PACKED: bool> Clone for TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer + Clone,
    T: Tag,
{
    fn clone(&self) -> Self {
        unsafe { Self::new(P::with_ref(self.raw.pointer_raw(), |p| p.clone()), self.raw.tag()) }
    }
}

// We pack the tag into the *upper* bits of the pointer to ease retrieval of the
// value; a right shift is a multiplication and those are embeddable in
// instruction encoding.
impl<P, T, const COMPARE_PACKED: bool> TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
{
    pub fn new(pointer: P, tag: T) -> Self {
        TaggedPtr { raw: CopyTaggedPtr::new(pointer, tag) }
    }

    pub fn pointer_ref(&self) -> &P::Target {
        self.raw.pointer_ref()
    }
    pub fn pointer_mut(&mut self) -> &mut P::Target
    where
        P: std::ops::DerefMut,
    {
        self.raw.pointer_mut()
    }
    pub fn tag(&self) -> T {
        self.raw.tag()
    }
    pub fn set_tag(&mut self, tag: T) {
        self.raw.set_tag(tag);
    }
}

impl<P, T, const COMPARE_PACKED: bool> std::ops::Deref for TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
{
    type Target = P::Target;
    fn deref(&self) -> &Self::Target {
        self.raw.pointer_ref()
    }
}

impl<P, T, const COMPARE_PACKED: bool> std::ops::DerefMut for TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer + std::ops::DerefMut,
    T: Tag,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.raw.pointer_mut()
    }
}

impl<P, T, const COMPARE_PACKED: bool> Drop for TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    T: Tag,
{
    fn drop(&mut self) {
        // No need to drop the tag, as it's Copy
        unsafe {
            std::mem::drop(P::from_usize(self.raw.pointer_raw()));
        }
    }
}

impl<P, T, const COMPARE_PACKED: bool> fmt::Debug for TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer,
    P::Target: fmt::Debug,
    T: Tag + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaggedPtr")
            .field("pointer", &self.pointer_ref())
            .field("tag", &self.tag())
            .finish()
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

impl<P, T> std::hash::Hash for TaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<P, T, HCX, const COMPARE_PACKED: bool> HashStable<HCX> for TaggedPtr<P, T, COMPARE_PACKED>
where
    P: Pointer + HashStable<HCX>,
    T: Tag + HashStable<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.raw.hash_stable(hcx, hasher);
    }
}
