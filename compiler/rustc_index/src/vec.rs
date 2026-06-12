use std::borrow::{Borrow, BorrowMut};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut, RangeBounds};
use std::{fmt, slice, vec};

#[cfg(feature = "nightly")]
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use crate::{Idx, IndexSlice};

/// An owned contiguous collection of `T`s, indexed by `I` rather than by `usize`.
///
/// ## Why use this instead of a `Vec`?
///
/// An `IndexVec` allows element access only via a specific associated index type, meaning that
/// trying to use the wrong index type (possibly accessing an invalid element) will fail at
/// compile time.
///
/// It also documents what the index is indexing: in a `HashMap<usize, Something>` it's not
/// immediately clear what the `usize` means, while a `HashMap<FieldIdx, Something>` makes it obvious.
///
/// ```compile_fail
/// use rustc_index::{Idx, IndexVec};
///
/// fn f<I1: Idx, I2: Idx>(vec1: IndexVec<I1, u8>, idx1: I1, idx2: I2) {
///   &vec1[idx1]; // Ok
///   &vec1[idx2]; // Compile error!
/// }
/// ```
///
/// While it's possible to use `u32` or `usize` directly for `I`,
/// you almost certainly want to use a [`newtype_index!`]-generated type instead.
///
/// This allows to index the IndexVec with the new index type.
///
/// [`newtype_index!`]: ../macro.newtype_index.html
pub struct IndexVec<I: Idx, T> {
    data: *mut T,
    len: I,
    capacity: I,

    _marker: PhantomData<fn(&I)>,
    _marker2: PhantomData<T>,
}

impl<I: Idx, T: Clone> Clone for IndexVec<I, T> {
    fn clone(&self) -> Self {
        IndexVec::from_raw(self.as_slice().raw.to_vec())
    }
}

impl<I: Idx, T: PartialEq> PartialEq for IndexVec<I, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<I: Idx, T: PartialEq> Eq for IndexVec<I, T> {}

impl<I: Idx, T: Hash> Hash for IndexVec<I, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

#[cfg(feature = "nightly")]
unsafe impl<I: Idx, #[may_dangle] T> Drop for IndexVec<I, T> {
    fn drop(&mut self) {
        std::mem::take(self).into_vec();
    }
}

#[cfg(not(feature = "nightly"))]
impl<I: Idx, T> Drop for IndexVec<I, T> {
    fn drop(&mut self) {
        std::mem::take(self).into_vec();
    }
}

impl<I: Idx, T> IndexVec<I, T> {
    /// Constructs a new, empty `IndexVec<I, T>`.
    #[inline]
    pub fn new() -> Self {
        IndexVec::from_raw(Vec::new())
    }

    /// Constructs a new `IndexVec<I, T>` from a `Vec<T>`.
    #[inline]
    pub fn from_raw(raw: Vec<T>) -> Self {
        let (data, len, capacity) = raw.into_raw_parts();

        IndexVec {
            data,
            len: I::new(len),
            capacity: I::new(capacity),

            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        let me = ManuallyDrop::new(self);
        // fixme this is unsound because we rely on correct Idx trait impls
        unsafe { Vec::from_raw_parts(me.data, me.len.index(), me.capacity.index()) }
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

    /// Creates a new IndexVec with n copies of the `elem`.
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
        // Allow the optimizer to elide the bounds checking when creating each index.
        let _ = I::new(n);
        IndexVec::from_raw((0..n).map(I::new).map(func).collect())
    }

    #[inline]
    pub fn as_slice(&self) -> &IndexSlice<I, T> {
        IndexSlice::from_raw(unsafe { std::slice::from_raw_parts(self.data, self.len.index()) })
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut IndexSlice<I, T> {
        IndexSlice::from_raw_mut(unsafe {
            std::slice::from_raw_parts_mut(self.data, self.len.index())
        })
    }

    /// Pushes an element to the array returning the index where it was pushed to.
    #[inline]
    pub fn push(&mut self, d: T) -> I {
        let idx = self.next_index();
        self.mutate(|vec| vec.push(d));
        idx
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.mutate(|raw| raw.pop())
    }

    #[inline]
    pub fn into_iter(self) -> vec::IntoIter<T> {
        self.into_vec().into_iter()
    }

    #[inline]
    pub fn into_iter_enumerated(
        self,
    ) -> impl DoubleEndedIterator<Item = (I, T)> + ExactSizeIterator {
        // Allow the optimizer to elide the bounds checking when creating each index.
        let _ = I::new(self.len());
        self.into_iter().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn drain_into<R: RangeBounds<usize>>(&mut self, range: R, target: &mut IndexVec<I, T>) {
        self.mutate(|raw| target.extend(raw.drain(range)))
    }

    pub fn mutate<U, F: FnOnce(&mut Vec<T>) -> U>(&mut self, f: F) -> U {
        let mut vec = std::mem::take(self).into_vec();
        let v = f(&mut vec);
        let _ = std::mem::replace(self, IndexVec::from_raw(vec));
        v
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.mutate(|vec| vec.shrink_to_fit());
    }

    #[inline]
    pub fn truncate(&mut self, a: usize) {
        self.mutate(|vec| vec.truncate(a))
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
            self.mutate(|vec| vec.resize_with(min_new_len, fill_value));
        }

        &mut self[elem]
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.mutate(|vec| vec.resize(new_len, value))
    }

    #[inline]
    pub fn resize_to_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) {
        let min_new_len = elem.index() + 1;
        self.mutate(|vec| vec.resize_with(min_new_len, fill_value));
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.mutate(|vec| other.mutate(|other| vec.append(other)));
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

    #[inline]
    pub fn contains(&self, index: I) -> bool {
        self.get(index).and_then(Option::as_ref).is_some()
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
        self.mutate(|vec| vec.extend(iter));
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_one(&mut self, item: T) {
        self.mutate(|vec| vec.push(item));
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_reserve(&mut self, additional: usize) {
        self.mutate(|vec| vec.reserve(additional));
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
        self.into_vec().into_iter()
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

#[cfg(feature = "nightly")]
impl<S: Encoder, I: Idx, T: Encodable<S>> Encodable<S> for IndexVec<I, T> {
    fn encode(&self, s: &mut S) {
        Encodable::encode(&self.raw, s);
    }
}

#[cfg(feature = "nightly")]
impl<D: Decoder, I: Idx, T: Decodable<D>> Decodable<D> for IndexVec<I, T> {
    fn decode(d: &mut D) -> Self {
        IndexVec::from_raw(Vec::<T>::decode(d))
    }
}

// Whether `IndexVec` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexVec<I, T> where T: Send {}

#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use super::IndexVec;
    use crate::static_assert_size;
    static_assert_size!(IndexVec<u32, u8>, 16);
    static_assert_size!(IndexVec<usize, u8>, 24);
}

#[cfg(test)]
mod tests;
