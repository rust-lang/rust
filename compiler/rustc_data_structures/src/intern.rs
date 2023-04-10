use crate::stable_hasher::{HashStable, StableHasher};
use crate::tagged_ptr::{self, CopyTaggedPtr};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;

mod private {
    use std::marker::PhantomData;

    #[derive(Clone, Copy, Debug)]
    pub struct PrivateZst<T>(pub(super) PhantomData<T>);
}

trait InternedPtr<'a, T>: Copy {
    fn as_ref(self) -> &'a T;
}

impl<'a, T> InternedPtr<'a, T> for &'a T {
    fn as_ref(self) -> &'a T {
        self
    }
}

impl<'a, T, P: tagged_ptr::Pointer + Copy + InternedPtr<'a, T>, Tag: tagged_ptr::Tag>
    InternedPtr<'a, T> for CopyTaggedPtr<P, Tag, false>
{
    fn as_ref(self) -> &'a T {
        InternedPtr::as_ref(self.pointer())
    }
}

/// A reference to a value that is interned, and is known to be unique.
///
/// Note that it is possible to have a `T` and a `Interned<T>` that are (or
/// refer to) equal but different values. But if you have two different
/// `Interned<T>`s, they both refer to the same value, at a single location in
/// memory. This means that equality and hashing can be done on the value's
/// address rather than the value's contents, which can improve performance.
///
/// The `PrivateZst` field means you can pattern match with `Interned(v, _)`
/// but you can only construct a `Interned` with `new_unchecked`, and not
/// directly.
#[derive(Debug)]
#[rustc_pass_by_value]
pub struct Interned<'a, T, P = &'a T>(pub P, pub private::PrivateZst<&'a T>);

impl<'a, T> Interned<'a, T> {
    /// Create a new `Interned` value. The value referred to *must* be interned
    /// and thus be unique, and it *must* remain unique in the future. This
    /// function has `_unchecked` in the name but is not `unsafe`, because if
    /// the uniqueness condition is violated condition it will cause incorrect
    /// behaviour but will not affect memory safety.
    #[inline]
    pub const fn new_unchecked(t: &'a T) -> Self {
        Interned(t, private::PrivateZst(PhantomData))
    }
}

impl<'a, T, P: InternedPtr<'a, T>> Clone for Interned<'a, T, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, P: InternedPtr<'a, T>> Copy for Interned<'a, T, P> {}

impl<'a, T, P: InternedPtr<'a, T>> Deref for Interned<'a, T, P> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.as_ref()
    }
}

impl<'a, T, P: InternedPtr<'a, T>> PartialEq for Interned<'a, T, P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Pointer equality implies equality, due to the uniqueness constraint.
        ptr::eq(self.0.as_ref(), other.0.as_ref())
    }
}

impl<'a, T, P: InternedPtr<'a, T>> Eq for Interned<'a, T, P> {}

impl<'a, T: PartialOrd, P: InternedPtr<'a, T>> PartialOrd for Interned<'a, T, P> {
    fn partial_cmp(&self, other: &Interned<'a, T, P>) -> Option<Ordering> {
        // Pointer equality implies equality, due to the uniqueness constraint,
        // but the contents must be compared otherwise.
        if ptr::eq(self.0.as_ref(), other.0.as_ref()) {
            Some(Ordering::Equal)
        } else {
            let res = self.0.as_ref().partial_cmp(other.0.as_ref());
            debug_assert_ne!(res, Some(Ordering::Equal));
            res
        }
    }
}

impl<'a, T: Ord, P: InternedPtr<'a, T>> Ord for Interned<'a, T, P> {
    fn cmp(&self, other: &Interned<'a, T, P>) -> Ordering {
        // Pointer equality implies equality, due to the uniqueness constraint,
        // but the contents must be compared otherwise.
        if ptr::eq(self.0.as_ref(), other.0.as_ref()) {
            Ordering::Equal
        } else {
            let res = self.0.as_ref().cmp(other.0.as_ref());
            debug_assert_ne!(res, Ordering::Equal);
            res
        }
    }
}

impl<'a, T, P: InternedPtr<'a, T>> Hash for Interned<'a, T, P> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // Pointer hashing is sufficient, due to the uniqueness constraint.
        ptr::hash(self.0.as_ref(), s)
    }
}

impl<'a, T, CTX, P: InternedPtr<'a, T>> HashStable<CTX> for Interned<'a, T, P>
where
    T: HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.0.as_ref().hash_stable(hcx, hasher);
    }
}

#[cfg(test)]
mod tests;
