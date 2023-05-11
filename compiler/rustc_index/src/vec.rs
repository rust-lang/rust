#[cfg(feature = "rustc_serialize")]
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, RangeBounds};
use std::slice;
use std::vec;

use crate::{Idx, IndexSlice};

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

impl<I: Idx, T> IndexVec<I, T> {
    #[inline]
    pub const fn new() -> Self {
        IndexVec::from_raw(Vec::new())
    }

    #[inline]
    pub const fn from_raw(raw: Vec<T>) -> Self {
        IndexVec { raw, _marker: PhantomData }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        IndexVec::from_raw(Vec::with_capacity(capacity))
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
        IndexVec::from_raw(vec![elem; universe.len()])
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        IndexVec::from_raw(vec![elem; n])
    }

    /// Create an `IndexVec` with `n` elements, where the value of each
    /// element is the result of `func(i)`. (The underlying vector will
    /// be allocated only once, with a capacity of at least `n`.)
    #[inline]
    pub fn from_fn_n(func: impl FnMut(I) -> T, n: usize) -> Self {
        IndexVec::from_raw((0..n).map(I::new).map(func).collect())
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
        let idx = self.next_index();
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
        IndexVec::from_raw(self.raw)
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
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.raw.resize(new_len, value)
    }

    #[inline]
    pub fn resize_to_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) {
        let min_new_len = elem.index() + 1;
        self.raw.resize_with(min_new_len, fill_value);
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

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
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
        IndexVec::from_raw(Vec::from_iter(iter))
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
        self.iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a mut IndexVec<I, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    #[inline]
    fn default() -> Self {
        IndexVec::new()
    }
}

impl<I: Idx, T, const N: usize> From<[T; N]> for IndexVec<I, T> {
    #[inline]
    fn from(array: [T; N]) -> Self {
        IndexVec::from_raw(array.into())
    }
}

#[cfg(feature = "rustc_serialize")]
impl<S: Encoder, I: Idx, T: Encodable<S>> Encodable<S> for IndexVec<I, T> {
    fn encode(&self, s: &mut S) {
        Encodable::encode(&self.raw, s);
    }
}

#[cfg(feature = "rustc_serialize")]
impl<D: Decoder, I: Idx, T: Decodable<D>> Decodable<D> for IndexVec<I, T> {
    fn decode(d: &mut D) -> Self {
        IndexVec::from_raw(Vec::<T>::decode(d))
    }
}

// Whether `IndexVec` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexVec<I, T> where T: Send {}

#[cfg(test)]
mod tests;
