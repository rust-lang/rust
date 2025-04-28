#![allow(unused_imports)]
use std::fmt;
use std::iter::ExactSizeIterator;
use std::ops::{Range, RangeInclusive};

use rustc_macros::{Decodable_Generic, Encodable_Generic};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

#[cfg(feature = "nightly")]
use super::old_dense_bit_set::{
    DenseBitSet as OldDenseBitSet, GrowableBitSet as OldGrowableBitSet,
};
use super::thin_bit_set::{GrowableBitSet as ThinGrowableBitSet, ThinBitSet};
use super::{BitIter, BitRelations, ChunkedBitSet, Word, bit_relations_inherent_impls};
use crate::Idx;

/// A product of [`ThinBitSet`] and [`OldDenseBitSet`]. Every operation is executed on both a
/// [`ThinBitSet`] and a [`OldDenseBitSet`] and the results are asserted to be equal.
///
/// Used to test [`ThinBitSet`].
#[derive(PartialEq, Eq, Hash)]
pub struct DenseBitSet<T> {
    thin: ThinBitSet<T>,
    old: OldDenseBitSet<T>,
}

macro_rules! compare_results(
    ($self:ident, $method:tt$(, $($args:expr),*)?) => {
        {
            let thin_res = $self.thin.$method($($($args),*)?);
            let old_res = $self.old.$method($($($args),*)?);
            $self.assert_valid();
            assert_eq!(thin_res, old_res, "TwinBitSet give different results in {}", stringify!($method));
            thin_res
        }
    };
);

macro_rules! compare_results_clone_arg(
    ($self:ident, $method:tt, $arg:expr) => {
        {
            let thin_res = $self.thin.$method($arg.clone());
            let old_res = $self.old.$method($arg);
            $self.assert_valid();
            assert_eq!(thin_res, old_res, "TwinBitSet give different results in {}", stringify!($method));
            thin_res
        }
    };
);

macro_rules! compare_results_with_self(
    ($self:ident, $method:tt, $other:expr) => {
        {
            let thin_res = $self.thin.$method(&$other.thin);
            let old_res = $self.old.$method(&$other.old);
            $self.assert_valid();
            assert_eq!(thin_res, old_res, "TwinBitSet give different results in {}", stringify!($method));
            thin_res
        }
    };
);

impl<T: Idx> DenseBitSet<T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> DenseBitSet<T> {
        let new = Self {
            thin: ThinBitSet::new_empty(domain_size),
            old: OldDenseBitSet::new_empty(domain_size),
        };
        new.assert_valid();
        new
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> DenseBitSet<T> {
        let new = Self {
            thin: ThinBitSet::new_filled(domain_size),
            old: OldDenseBitSet::new_filled(domain_size),
        };
        new.assert_valid();
        new
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        compare_results!(self, clear)
    }

    /// Count the number of set bits in the set.
    pub fn count(&self) -> usize {
        compare_results!(self, count)
    }

    /// Returns `true` if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        if self.old.domain_size() <= elem.index() {
            println!("{}: {:?}", self.old.domain_size(), self.old);
            println!("{}: {:?}", self.thin.capacity(), self.thin);
            assert!(!self.thin.contains(elem));
            return false;
        }
        compare_results!(self, contains, elem)
    }

    /// Is `self` is a (non-strict) superset of `other`?
    #[inline]
    pub fn superset(&self, other: &DenseBitSet<T>) -> bool {
        compare_results_with_self!(self, superset, other)
    }

    /// Is the set empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        compare_results!(self, is_empty)
    }

    /// Insert `elem`. Returns whether the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        compare_results!(self, insert, elem)
    }

    #[inline]
    pub fn insert_range(&mut self, elems: Range<T>) {
        compare_results_clone_arg!(self, insert_range, elems)
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self, domain_size: usize) {
        compare_results!(self, insert_all, domain_size)
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        compare_results!(self, remove, elem)
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        assert!(self.thin.iter().eq(self.old.iter()));
        self.thin.iter()
    }

    pub fn last_set_in(&self, range: RangeInclusive<T>) -> Option<T> {
        compare_results_clone_arg!(self, last_set_in, range)
    }

    bit_relations_inherent_impls! {}

    /// Sets `self = self | !other`.
    pub fn union_not(&mut self, other: &DenseBitSet<T>) {
        compare_results_with_self!(self, union_not, other)
    }

    #[inline]
    pub(crate) fn words(&self) -> impl ExactSizeIterator<Item = Word> {
        assert!(self.thin.words().eq(self.old.words()));
        self.thin.words()
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        assert!(self.thin.capacity() >= self.old.capacity());
        self.thin.capacity()
    }

    #[inline]
    pub fn insert_range_inclusive(&mut self, elems: RangeInclusive<T>) {
        compare_results_clone_arg!(self, insert_range_inclusive, elems)
    }

    #[inline]
    fn assert_valid(&self) {
        assert!(self.thin.iter().eq(self.old.iter()));
    }
}

// dense REL dense
impl<T: Idx> BitRelations<DenseBitSet<T>> for DenseBitSet<T> {
    fn union(&mut self, other: &DenseBitSet<T>) -> bool {
        compare_results_with_self!(self, union, other)
    }

    fn subtract(&mut self, other: &DenseBitSet<T>) -> bool {
        compare_results_with_self!(self, subtract, other)
    }

    fn intersect(&mut self, other: &DenseBitSet<T>) -> bool {
        compare_results_with_self!(self, intersect, other)
    }
}

impl<T: Idx> From<GrowableBitSet<T>> for DenseBitSet<T> {
    fn from(bit_set: GrowableBitSet<T>) -> Self {
        Self { thin: bit_set.thin.into(), old: bit_set.old.into() }
    }
}

impl<T: Idx> BitRelations<ChunkedBitSet<T>> for DenseBitSet<T> {
    fn union(&mut self, other: &ChunkedBitSet<T>) -> bool {
        compare_results!(self, union, other)
    }

    fn subtract(&mut self, other: &ChunkedBitSet<T>) -> bool {
        compare_results!(self, subtract, other)
    }

    fn intersect(&mut self, other: &ChunkedBitSet<T>) -> bool {
        compare_results!(self, intersect, other)
    }
}

impl<T: Idx> fmt::Debug for DenseBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.thin.fmt(w)
    }
}

impl<T> Clone for DenseBitSet<T> {
    fn clone(&self) -> Self {
        let new = Self { thin: self.thin.clone(), old: self.old.clone() };
        //assert!(new.thin.words().eq(new.old.words()));
        new
    }
}

/*impl<T: PartialEq> PartialEq for DenseBitSet<T> {
    fn eq(&self, other: &Self) -> bool {
        let thin_res = self.thin == other.thin;
        let old_res = self.old == other.old;
        assert_eq!(thin_res, old_res);
        thin_res
    }
}

impl<T: Eq> Eq for DenseBitSet<T> {}

impl<T: Idx> Hash for DenseBitSet<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.assert_valid();
        self.old.hash(hasher);
    }
}
*/

impl<S: Encoder, T: Encodable<S>> Encodable<S> for DenseBitSet<T> {
    fn encode(&self, s: &mut S) {
        self.old.encode(s);
    }
}

impl<D: Decoder, T: Idx + Decodable<D>> Decodable<D> for DenseBitSet<T> {
    fn decode(d: &mut D) -> Self {
        let old = OldDenseBitSet::<T>::decode(d);
        let mut thin = ThinBitSet::<T>::new_empty(old.domain_size());
        for x in old.iter() {
            thin.insert(x);
        }
        assert!(old.iter().eq(thin.iter()), "{:x?} != {:x?}", old, thin);
        Self { old, thin }
    }
}

/// A resizable bitset type with a dense representation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size.
#[derive(Clone, PartialEq)]
pub struct GrowableBitSet<T: Idx> {
    thin: ThinGrowableBitSet<T>,
    old: OldGrowableBitSet<T>,
}

impl<T: Idx> Default for GrowableBitSet<T> {
    fn default() -> Self {
        GrowableBitSet::new_empty()
    }
}

impl<T: Idx> GrowableBitSet<T> {
    /// Ensure that the set can hold at least `min_domain_size` elements.
    pub fn ensure(&mut self, min_domain_size: usize) {
        compare_results!(self, ensure, min_domain_size)
    }

    pub fn new_empty() -> GrowableBitSet<T> {
        let new = GrowableBitSet {
            thin: ThinGrowableBitSet::new_empty(),
            old: OldGrowableBitSet::new_empty(),
        };
        new.assert_valid();
        new
    }

    pub fn with_capacity(capacity: usize) -> GrowableBitSet<T> {
        let new = GrowableBitSet {
            thin: ThinGrowableBitSet::with_capacity(capacity),
            old: OldGrowableBitSet::with_capacity(capacity),
        };
        new.assert_valid();
        new
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        compare_results!(self, insert, elem)
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        compare_results!(self, remove, elem)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        compare_results!(self, is_empty)
    }

    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        compare_results!(self, contains, elem)
    }

    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        self.assert_valid();
        self.thin.iter()
    }

    #[inline]
    fn assert_valid(&self) {
        assert!(self.thin.iter().eq(self.old.iter()));
    }

    #[inline]
    pub fn len(&self) -> usize {
        compare_results!(self, len)
    }
}

/*impl<T: PartialEq + Idx> PartialEq for GrowableBitSet<T> {
    fn eq(&self, other: &Self) -> bool {
        let thin_res = self.thin == other.thin;
        let old_res = self.old == other.old;
        assert_eq!(thin_res, old_res);
        thin_res
    }
}
*/

impl<T: Idx> fmt::Debug for GrowableBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.thin.fmt(w)
    }
}

/*impl<T: Idx> Clone for GrowableBitSet<T> {
    fn clone(&self) -> Self {
        let new = Self { thin: self.thin.clone(), old: self.old.clone() };
        new
    }
}
*/

impl<T: Idx> From<DenseBitSet<T>> for GrowableBitSet<T> {
    fn from(bit_set: DenseBitSet<T>) -> Self {
        Self { thin: bit_set.thin.into(), old: bit_set.old.into() }
    }
}
