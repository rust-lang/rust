use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr;

mod private {
    #[derive(Clone, Copy, Debug)]
    pub struct PrivateZst;
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
#[cfg_attr(not(bootstrap), rustc_pass_by_value)]
pub struct Interned<'a, T>(pub &'a T, pub private::PrivateZst);

impl<'a, T> Interned<'a, T> {
    /// Create a new `Interned` value. The value referred to *must* be interned
    /// and thus be unique, and it *must* remain unique in the future. This
    /// function has `_unchecked` in the name but is not `unsafe`, because if
    /// the uniqueness condition is violated condition it will cause incorrect
    /// behaviour but will not affect memory safety.
    #[inline]
    pub const fn new_unchecked(t: &'a T) -> Self {
        Interned(t, private::PrivateZst)
    }
}

impl<'a, T> Clone for Interned<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for Interned<'a, T> {}

impl<'a, T> Deref for Interned<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0
    }
}

impl<'a, T> PartialEq for Interned<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Pointer equality implies equality, due to the uniqueness constraint.
        ptr::eq(self.0, other.0)
    }
}

impl<'a, T> Eq for Interned<'a, T> {}

// In practice you can't intern any `T` that doesn't implement `Eq`, because
// that's needed for hashing. Therefore, we won't be interning any `T` that
// implements `PartialOrd` without also implementing `Ord`. So we can have the
// bound `T: Ord` here and avoid duplication with the `Ord` impl below.
impl<'a, T: Ord> PartialOrd for Interned<'a, T> {
    fn partial_cmp(&self, other: &Interned<'a, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T: Ord> Ord for Interned<'a, T> {
    fn cmp(&self, other: &Interned<'a, T>) -> Ordering {
        // Pointer equality implies equality, due to the uniqueness constraint,
        // but the contents must be compared otherwise.
        if ptr::eq(self.0, other.0) {
            Ordering::Equal
        } else {
            let res = self.0.cmp(&other.0);
            debug_assert_ne!(res, Ordering::Equal);
            res
        }
    }
}

impl<'a, T> Hash for Interned<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // Pointer hashing is sufficient, due to the uniqueness constraint.
        ptr::hash(self.0, s)
    }
}

#[cfg(test)]
mod tests;
