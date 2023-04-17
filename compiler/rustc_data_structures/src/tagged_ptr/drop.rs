use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

use super::CopyTaggedPtr;
use super::{Pointer, Tag};
use crate::stable_hasher::{HashStable, StableHasher};

/// A tagged pointer that supports pointers that implement [`Drop`].
///
/// This is essentially `{ pointer: P, tag: T }` packed in a single pointer.
///
/// You should use [`CopyTaggedPtr`] instead of the this type in all cases
/// where `P` implements [`Copy`].
///
/// If `COMPARE_PACKED` is true, then the pointers will be compared and hashed without
/// unpacking. Otherwise we don't implement [`PartialEq`], [`Eq`] and [`Hash`];
/// if you want that, wrap the [`TaggedPtr`].
pub struct TaggedPtr<P, T, const COMPARE_PACKED: bool>
where
    P: Pointer,
    T: Tag,
{
    raw: CopyTaggedPtr<P, T, COMPARE_PACKED>,
}

impl<P, T, const CP: bool> TaggedPtr<P, T, CP>
where
    P: Pointer,
    T: Tag,
{
    /// Tags `pointer` with `tag`.
    pub fn new(pointer: P, tag: T) -> Self {
        TaggedPtr { raw: CopyTaggedPtr::new(pointer, tag) }
    }

    /// Retrieves the tag.
    pub fn tag(&self) -> T {
        self.raw.tag()
    }

    /// Sets the tag to a new value.
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

/// Test that `new` does not compile if there is not enough alignment for the
/// tag in the pointer.
///
/// ```compile_fail,E0080
/// use rustc_data_structures::tagged_ptr::{TaggedPtr, Tag};
///
/// #[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// enum Tag2 { B00 = 0b00, B01 = 0b01, B10 = 0b10, B11 = 0b11 };
///
/// unsafe impl Tag for Tag2 {
///     const BITS: u32 = 2;
///
///     fn into_usize(self) -> usize { todo!() }
///     unsafe fn from_usize(tag: usize) -> Self { todo!() }
/// }
///
/// let value = 12u16;
/// let reference = &value;
/// let tag = Tag2::B01;
///
/// let _ptr = TaggedPtr::<_, _, true>::new(reference, tag);
/// ```
// For some reason miri does not get the compile error
// probably it `check`s instead of `build`ing?
#[cfg(not(miri))]
const _: () = ();

#[cfg(test)]
mod tests;
