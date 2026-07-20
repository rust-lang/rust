use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr;

use crate::stable_hash::{StableHash, StableHashCtxt, StableHasher};

mod private {
    #[derive(Clone, Copy, Debug)]
    pub struct PrivateZst;
}

/// This type is a reference with one special behaviour: the reference pointer (i.e. the address of
/// the value referred to) is used for equality and hashing, rather than the value's contents, as
/// would occur with a vanilla reference. There are two cases when this is useful.
///
/// - Types where uniqueness is guaranteed. This is most commonly achieved via interning -- hence
///   the name `Interned` -- though it may also be possible via other means. In this case, the use
///   of `Interned` is primarily a performance optimization, because pointer equality/hashing gives
///   the same results as value equality/hashing, but is faster. (The use of the `Interned` type
///   also provides documentation about the interned-ness.)
///
///   Note that in this case it is possible to have a `T` and a `Interned<T>` that are (or refer
///   to) equal but different values. But if you have two different `Interned<T>`s, they both refer
///   to the same value, at a single location in memory.
///
/// - Types with identity, where distinct values should always be considered unequal, even if they
///   have equal values. These are rare in Rust, but do occur sometimes. In this case, the use of
///   `Interned` gives different behaviour, because pointer equality/hashing gives different result
///   to value equality/hashing, and is also faster.
///
/// The `PrivateZst` field means you can pattern match with `Interned(v, _)` but you can only
/// construct a `Interned` with `new_unchecked`, and not directly. This means that all creation
/// points can be audited easily.
#[rustc_pass_by_value]
pub struct Interned<'a, T>(pub &'a T, pub private::PrivateZst);

impl<'a, T> Interned<'a, T> {
    /// Create a new `Interned` value. The value referred to *must* satisfy one of the following
    /// two conditions.
    /// - It must be unique and it must remain unique in the future.
    /// - It must be of a type with "identity" such that distinct values should always be
    ///   considered unequal.
    ///
    /// This function has `_unchecked` in the name but is not `unsafe`, because if neither of these
    /// conditions is met it will cause incorrect behaviour but will not affect memory safety.
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

impl<'a, T> Hash for Interned<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // Pointer hashing is sufficient.
        ptr::hash(self.0, s)
    }
}

impl<T> StableHash for Interned<'_, T>
where
    T: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.0.stable_hash(hcx, hasher);
    }
}

impl<T: Debug> Debug for Interned<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests;
