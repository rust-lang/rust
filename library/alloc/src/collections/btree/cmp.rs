//! Complements to the [`core::cmp`] module

use core::borrow::Borrow;
use core::cmp::Ordering;

/// Key equivalence trait.
///
/// This trait allows hash table lookup to be customized. It has one blanket
/// implementation that uses the regular solution with `Borrow` and `Eq`, just
/// like `HashMap` does, so that you can pass `&str` to lookup into a map with
/// `String` keys and so on.
///
/// # Contract
///
/// The implementor **must** hash like `Q`, if it is hashable.
pub(crate) trait Equivalent<Q: ?Sized> {
    // /// Compare self to `key` and return `true` if they are equal.
    // fn equivalent(&self, key: &Q) -> bool;
}

impl<K: ?Sized, Q: ?Sized> Equivalent<Q> for K
where
    K: Borrow<Q>,
    Q: Eq,
{
    // #[inline]
    // fn equivalent(&self, key: &Q) -> bool {
    //     PartialEq::eq(self.borrow(), key)
    // }
}

/// Key ordering trait.
///
/// This trait allows ordered map lookup to be customized. It has one blanket
/// implementation that uses the regular solution with `Borrow` and `Ord`, just
/// like `BTreeMap` does, so that you can pass `&str` to lookup into a map with
/// `String` keys and so on.
pub(crate) trait Comparable<Q: ?Sized>: Equivalent<Q> {
    /// Compare self to `key` and return their ordering.
    fn compare(&self, key: &Q) -> Ordering;
}

impl<K: ?Sized, Q: ?Sized> Comparable<Q> for K
where
    K: Borrow<Q>,
    Q: Ord,
{
    #[inline]
    fn compare(&self, key: &Q) -> Ordering {
        Ord::cmp(self.borrow(), key)
    }
}
