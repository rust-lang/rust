#[cfg(feature = "rustc_serialize")]
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use std::slice;
use std::vec;

/// Represents some newtyped `usize` wrapper.
///
/// Purpose: avoid mixing indexes for different bitvector domains.
pub trait Idx: Copy + 'static + Eq + PartialEq + Debug + Hash {
    fn new(idx: usize) -> Self;

    fn index(self) -> usize;

    #[inline]
    fn increment_by(&mut self, amount: usize) {
        *self = self.plus(amount);
    }

    #[inline]
    #[must_use = "Use `increment_by` if you wanted to update the index in-place"]
    fn plus(self, amount: usize) -> Self {
        Self::new(self.index() + amount)
    }
}

impl Idx for usize {
    #[inline]
    fn new(idx: usize) -> Self {
        idx
    }
    #[inline]
    fn index(self) -> usize {
        self
    }
}

impl Idx for u32 {
    #[inline]
    fn new(idx: usize) -> Self {
        assert!(idx <= u32::MAX as usize);
        idx as u32
    }
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

/// An owned contiguous collection of `T`s, indexed by `I` rather than by `usize`.
///
/// While it's possible to use `u32` or `usize` directly for `I`,
/// you almost certainly want to use a [`newtype_index!`]-generated type instead.
///
/// [`newtype_index!`]: ../macro.newtype_index.html
#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IndexVec<I: Idx, T> {
    pub raw: Vec<T>,
    _marker: PhantomData<fn(&I)>,
}

/// A view into contiguous `T`s, indexed by `I` rather than by `usize`.
///
/// One common pattern you'll see is code that uses [`IndexVec::from_elem`]
/// to create the storage needed for a particular "universe" (aka the set of all
/// the possible keys that need an associated value) then passes that working
/// area as `&mut IndexSlice<I, T>` to clarify that nothing will be added nor
/// removed during processing (and, as a bonus, to chase fewer pointers).
#[derive(PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IndexSlice<I: Idx, T> {
    _marker: PhantomData<fn(&I)>,
    pub raw: [T],
}

// Whether `IndexVec` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexVec<I, T> where T: Send {}

// Whether `IndexSlice` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexSlice<I, T> where T: Send {}

#[cfg(feature = "rustc_serialize")]
impl<S: Encoder, I: Idx, T: Encodable<S>> Encodable<S> for IndexVec<I, T> {
    fn encode(&self, s: &mut S) {
        Encodable::encode(&self.raw, s);
    }
}

#[cfg(feature = "rustc_serialize")]
impl<D: Decoder, I: Idx, T: Decodable<D>> Decodable<D> for IndexVec<I, T> {
    fn decode(d: &mut D) -> Self {
        IndexVec { raw: Decodable::decode(d), _marker: PhantomData }
    }
}

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexSlice<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

impl<I: Idx, T> IndexVec<I, T> {
    #[inline]
    pub fn new() -> Self {
        IndexVec { raw: Vec::new(), _marker: PhantomData }
    }

    #[inline]
    pub fn from_raw(raw: Vec<T>) -> Self {
        IndexVec { raw, _marker: PhantomData }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        IndexVec { raw: Vec::with_capacity(capacity), _marker: PhantomData }
    }

    /// Creates a new vector with a copy of `elem` for each index in `universe`.
    ///
    /// Thus `IndexVec::from_elem(elem, &universe)` is equivalent to
    /// `IndexVec::<I, _>::from_elem_n(elem, universe.len())`. That can help
    /// type inference as it ensures that the resulting vector uses the same
    /// index type as `universe`, rather than something potentially surprising.
    ///
    /// For example, if you want to store data for each local in a MIR body,
    /// using `let mut uses = IndexVec::from_elem(vec![], &body.local_decls);`
    /// ensures that `uses` is an `IndexVec<Local, _>`, and thus can give
    /// better error messages later if one accidentally mismatches indices.
    #[inline]
    pub fn from_elem<S>(elem: T, universe: &IndexSlice<I, S>) -> Self
    where
        T: Clone,
    {
        IndexVec { raw: vec![elem; universe.len()], _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        IndexVec { raw: vec![elem; n], _marker: PhantomData }
    }

    /// Create an `IndexVec` with `n` elements, where the value of each
    /// element is the result of `func(i)`. (The underlying vector will
    /// be allocated only once, with a capacity of at least `n`.)
    #[inline]
    pub fn from_fn_n(func: impl FnMut(I) -> T, n: usize) -> Self {
        let indices = (0..n).map(I::new);
        Self::from_raw(indices.map(func).collect())
    }

    #[inline]
    pub fn as_slice(&self) -> &IndexSlice<I, T> {
        IndexSlice::from_raw(&self.raw)
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut IndexSlice<I, T> {
        IndexSlice::from_raw_mut(&mut self.raw)
    }

    #[inline]
    pub fn push(&mut self, d: T) -> I {
        let idx = I::new(self.len());
        self.raw.push(d);
        idx
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.raw.pop()
    }

    #[inline]
    pub fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

    #[inline]
    pub fn into_iter_enumerated(
        self,
    ) -> impl DoubleEndedIterator<Item = (I, T)> + ExactSizeIterator {
        self.raw.into_iter().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> impl Iterator<Item = T> + '_ {
        self.raw.drain(range)
    }

    #[inline]
    pub fn drain_enumerated<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> impl Iterator<Item = (I, T)> + '_ {
        let begin = match range.start_bound() {
            std::ops::Bound::Included(i) => *i,
            std::ops::Bound::Excluded(i) => i.checked_add(1).unwrap(),
            std::ops::Bound::Unbounded => 0,
        };
        self.raw.drain(range).enumerate().map(move |(n, t)| (I::new(begin + n), t))
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.raw.shrink_to_fit()
    }

    #[inline]
    pub fn truncate(&mut self, a: usize) {
        self.raw.truncate(a)
    }

    pub fn convert_index_type<Ix: Idx>(self) -> IndexVec<Ix, T> {
        IndexVec { raw: self.raw, _marker: PhantomData }
    }

    /// Grows the index vector so that it contains an entry for
    /// `elem`; if that is already true, then has no
    /// effect. Otherwise, inserts new values as needed by invoking
    /// `fill_value`.
    ///
    /// Returns a reference to the `elem` entry.
    #[inline]
    pub fn ensure_contains_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) -> &mut T {
        let min_new_len = elem.index() + 1;
        if self.len() < min_new_len {
            self.raw.resize_with(min_new_len, fill_value);
        }

        &mut self[elem]
    }

    #[inline]
    pub fn resize_to_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) {
        let min_new_len = elem.index() + 1;
        self.raw.resize_with(min_new_len, fill_value);
    }
}

impl<I: Idx, T> Deref for IndexVec<I, T> {
    type Target = IndexSlice<I, T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<I: Idx, T> DerefMut for IndexVec<I, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<I: Idx, T> Borrow<IndexSlice<I, T>> for IndexVec<I, T> {
    fn borrow(&self) -> &IndexSlice<I, T> {
        self
    }
}

impl<I: Idx, T> BorrowMut<IndexSlice<I, T>> for IndexVec<I, T> {
    fn borrow_mut(&mut self) -> &mut IndexSlice<I, T> {
        self
    }
}

impl<I: Idx, T: Clone> ToOwned for IndexSlice<I, T> {
    type Owned = IndexVec<I, T>;

    fn to_owned(&self) -> IndexVec<I, T> {
        IndexVec::from_raw(self.raw.to_owned())
    }

    fn clone_into(&self, target: &mut IndexVec<I, T>) {
        self.raw.clone_into(&mut target.raw)
    }
}

impl<I: Idx, T> IndexSlice<I, T> {
    #[inline]
    pub fn empty() -> &'static Self {
        Default::default()
    }

    #[inline]
    pub fn from_raw(raw: &[T]) -> &Self {
        let ptr: *const [T] = raw;
        // SAFETY: `IndexSlice` is `repr(transparent)` over a normal slice
        unsafe { &*(ptr as *const Self) }
    }

    #[inline]
    pub fn from_raw_mut(raw: &mut [T]) -> &mut Self {
        let ptr: *mut [T] = raw;
        // SAFETY: `IndexSlice` is `repr(transparent)` over a normal slice
        unsafe { &mut *(ptr as *mut Self) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Gives the next index that will be assigned when `push` is called.
    ///
    /// Manual bounds checks can be done using `idx < slice.next_index()`
    /// (as opposed to `idx.index() < slice.len()`).
    #[inline]
    pub fn next_index(&self) -> I {
        I::new(self.len())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.raw.iter()
    }

    #[inline]
    pub fn iter_enumerated(
        &self,
    ) -> impl DoubleEndedIterator<Item = (I, &T)> + ExactSizeIterator + '_ {
        self.raw.iter().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn indices(
        &self,
    ) -> impl DoubleEndedIterator<Item = I> + ExactSizeIterator + Clone + 'static {
        (0..self.len()).map(|n| I::new(n))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.raw.iter_mut()
    }

    #[inline]
    pub fn iter_enumerated_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (I, &mut T)> + ExactSizeIterator + '_ {
        self.raw.iter_mut().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn last_index(&self) -> Option<I> {
        self.len().checked_sub(1).map(I::new)
    }

    #[inline]
    pub fn swap(&mut self, a: I, b: I) {
        self.raw.swap(a.index(), b.index())
    }

    #[inline]
    pub fn get(&self, index: I) -> Option<&T> {
        self.raw.get(index.index())
    }

    #[inline]
    pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
        self.raw.get_mut(index.index())
    }

    /// Returns mutable references to two distinct elements, `a` and `b`.
    ///
    /// Panics if `a == b`.
    #[inline]
    pub fn pick2_mut(&mut self, a: I, b: I) -> (&mut T, &mut T) {
        let (ai, bi) = (a.index(), b.index());
        assert!(ai != bi);

        if ai < bi {
            let (c1, c2) = self.raw.split_at_mut(bi);
            (&mut c1[ai], &mut c2[0])
        } else {
            let (c2, c1) = self.pick2_mut(b, a);
            (c1, c2)
        }
    }

    /// Returns mutable references to three distinct elements.
    ///
    /// Panics if the elements are not distinct.
    #[inline]
    pub fn pick3_mut(&mut self, a: I, b: I, c: I) -> (&mut T, &mut T, &mut T) {
        let (ai, bi, ci) = (a.index(), b.index(), c.index());
        assert!(ai != bi && bi != ci && ci != ai);
        let len = self.raw.len();
        assert!(ai < len && bi < len && ci < len);
        let ptr = self.raw.as_mut_ptr();
        unsafe { (&mut *ptr.add(ai), &mut *ptr.add(bi), &mut *ptr.add(ci)) }
    }
}

impl<I: Idx, J: Idx> IndexSlice<I, J> {
    /// Invert a bijective mapping, i.e. `invert(map)[y] = x` if `map[x] = y`,
    /// assuming the values in `self` are a permutation of `0..self.len()`.
    ///
    /// This is used to go between `memory_index` (source field order to memory order)
    /// and `inverse_memory_index` (memory order to source field order).
    /// See also `FieldsShape::Arbitrary::memory_index` for more details.
    // FIXME(eddyb) build a better abstraction for permutations, if possible.
    pub fn invert_bijective_mapping(&self) -> IndexVec<J, I> {
        debug_assert_eq!(
            self.iter().map(|x| x.index() as u128).sum::<u128>(),
            (0..self.len() as u128).sum::<u128>(),
            "The values aren't 0..N in input {self:?}",
        );

        let mut inverse = IndexVec::from_elem_n(Idx::new(0), self.len());
        for (i1, &i2) in self.iter_enumerated() {
            inverse[i2] = i1;
        }

        debug_assert_eq!(
            inverse.iter().map(|x| x.index() as u128).sum::<u128>(),
            (0..inverse.len() as u128).sum::<u128>(),
            "The values aren't 0..N in result {self:?}",
        );

        inverse
    }
}

/// `IndexVec` is often used as a map, so it provides some map-like APIs.
impl<I: Idx, T> IndexVec<I, Option<T>> {
    #[inline]
    pub fn insert(&mut self, index: I, value: T) -> Option<T> {
        self.ensure_contains_elem(index, || None).replace(value)
    }

    #[inline]
    pub fn get_or_insert_with(&mut self, index: I, value: impl FnOnce() -> T) -> &mut T {
        self.ensure_contains_elem(index, || None).get_or_insert_with(value)
    }

    #[inline]
    pub fn remove(&mut self, index: I) -> Option<T> {
        self.get_mut(index)?.take()
    }
}

impl<I: Idx, T: Clone> IndexVec<I, T> {
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.raw.resize(new_len, value)
    }
}

impl<I: Idx, T: Ord> IndexSlice<I, T> {
    #[inline]
    pub fn binary_search(&self, value: &T) -> Result<I, I> {
        match self.raw.binary_search(value) {
            Ok(i) => Ok(Idx::new(i)),
            Err(i) => Err(Idx::new(i)),
        }
    }
}

impl<I: Idx, T> Index<I> for IndexSlice<I, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &T {
        &self.raw[index.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IndexSlice<I, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut T {
        &mut self.raw[index.index()]
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> Default for &IndexSlice<I, T> {
    #[inline]
    fn default() -> Self {
        IndexSlice::from_raw(Default::default())
    }
}

impl<I: Idx, T> Default for &mut IndexSlice<I, T> {
    #[inline]
    fn default() -> Self {
        IndexSlice::from_raw_mut(Default::default())
    }
}

impl<I: Idx, T> Extend<T> for IndexVec<I, T> {
    #[inline]
    fn extend<J: IntoIterator<Item = T>>(&mut self, iter: J) {
        self.raw.extend(iter);
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_one(&mut self, item: T) {
        self.raw.push(item);
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_reserve(&mut self, additional: usize) {
        self.raw.reserve(additional);
    }
}

impl<I: Idx, T> FromIterator<T> for IndexVec<I, T> {
    #[inline]
    fn from_iter<J>(iter: J) -> Self
    where
        J: IntoIterator<Item = T>,
    {
        IndexVec { raw: FromIterator::from_iter(iter), _marker: PhantomData }
    }
}

impl<I: Idx, T, const N: usize> From<[T; N]> for IndexVec<I, T> {
    #[inline]
    fn from(array: [T; N]) -> Self {
        IndexVec::from_raw(array.into())
    }
}

impl<I: Idx, T> IntoIterator for IndexVec<I, T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a mut IndexVec<I, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a IndexSlice<I, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a mut IndexSlice<I, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}

#[cfg(test)]
mod tests;
