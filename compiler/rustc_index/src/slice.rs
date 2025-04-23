use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, RangeBounds};
use std::slice::GetDisjointMutError::*;
use std::slice::{self, SliceIndex};

use crate::{Idx, IndexVec, IntoSliceIdx};

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

impl<I: Idx, T> IndexSlice<I, T> {
    #[inline]
    pub const fn empty<'a>() -> &'a Self {
        Self::from_raw(&[])
    }

    #[inline]
    pub const fn from_raw(raw: &[T]) -> &Self {
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
    pub const fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
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
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.raw.iter()
    }

    #[inline]
    pub fn iter_enumerated(&self) -> impl DoubleEndedIterator<Item = (I, &T)> + ExactSizeIterator {
        // Allow the optimizer to elide the bounds checking when creating each index.
        let _ = I::new(self.len());
        self.raw.iter().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn indices(
        &self,
    ) -> impl DoubleEndedIterator<Item = I> + ExactSizeIterator + Clone + 'static {
        // Allow the optimizer to elide the bounds checking when creating each index.
        let _ = I::new(self.len());
        (0..self.len()).map(|n| I::new(n))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.raw.iter_mut()
    }

    #[inline]
    pub fn iter_enumerated_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (I, &mut T)> + ExactSizeIterator {
        // Allow the optimizer to elide the bounds checking when creating each index.
        let _ = I::new(self.len());
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
    pub fn copy_within(
        &mut self,
        src: impl IntoSliceIdx<I, [T], Output: RangeBounds<usize>>,
        dest: I,
    ) where
        T: Copy,
    {
        self.raw.copy_within(src.into_slice_idx(), dest.index());
    }

    #[inline]
    pub fn get<R: IntoSliceIdx<I, [T]>>(
        &self,
        index: R,
    ) -> Option<&<R::Output as SliceIndex<[T]>>::Output> {
        self.raw.get(index.into_slice_idx())
    }

    #[inline]
    pub fn get_mut<R: IntoSliceIdx<I, [T]>>(
        &mut self,
        index: R,
    ) -> Option<&mut <R::Output as SliceIndex<[T]>>::Output> {
        self.raw.get_mut(index.into_slice_idx())
    }

    /// Returns mutable references to two distinct elements, `a` and `b`.
    ///
    /// Panics if `a == b` or if some of them are out of bounds.
    #[inline]
    pub fn pick2_mut(&mut self, a: I, b: I) -> (&mut T, &mut T) {
        let (ai, bi) = (a.index(), b.index());

        match self.raw.get_disjoint_mut([ai, bi]) {
            Ok([a, b]) => (a, b),
            Err(OverlappingIndices) => panic!("Indices {ai:?} and {bi:?} are not disjoint!"),
            Err(IndexOutOfBounds) => {
                panic!("Some indices among ({ai:?}, {bi:?}) are out of bounds")
            }
        }
    }

    /// Returns mutable references to three distinct elements.
    ///
    /// Panics if the elements are not distinct or if some of them are out of bounds.
    #[inline]
    pub fn pick3_mut(&mut self, a: I, b: I, c: I) -> (&mut T, &mut T, &mut T) {
        let (ai, bi, ci) = (a.index(), b.index(), c.index());

        match self.raw.get_disjoint_mut([ai, bi, ci]) {
            Ok([a, b, c]) => (a, b, c),
            Err(OverlappingIndices) => {
                panic!("Indices {ai:?}, {bi:?} and {ci:?} are not disjoint!")
            }
            Err(IndexOutOfBounds) => {
                panic!("Some indices among ({ai:?}, {bi:?}, {ci:?}) are out of bounds")
            }
        }
    }

    #[inline]
    pub fn binary_search(&self, value: &T) -> Result<I, I>
    where
        T: Ord,
    {
        match self.raw.binary_search(value) {
            Ok(i) => Ok(Idx::new(i)),
            Err(i) => Err(Idx::new(i)),
        }
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

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexSlice<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

impl<I: Idx, T, R: IntoSliceIdx<I, [T]>> Index<R> for IndexSlice<I, T> {
    type Output = <R::Output as SliceIndex<[T]>>::Output;

    #[inline]
    fn index(&self, index: R) -> &Self::Output {
        &self.raw[index.into_slice_idx()]
    }
}

impl<I: Idx, T, R: IntoSliceIdx<I, [T]>> IndexMut<R> for IndexSlice<I, T> {
    #[inline]
    fn index_mut(&mut self, index: R) -> &mut Self::Output {
        &mut self.raw[index.into_slice_idx()]
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

impl<I: Idx, T: Clone> ToOwned for IndexSlice<I, T> {
    type Owned = IndexVec<I, T>;

    fn to_owned(&self) -> IndexVec<I, T> {
        IndexVec::from_raw(self.raw.to_owned())
    }

    fn clone_into(&self, target: &mut IndexVec<I, T>) {
        self.raw.clone_into(&mut target.raw)
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

// Whether `IndexSlice` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexSlice<I, T> where T: Send {}
