// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A dynamically-sized view into a contiguous sequence, `[T]`.
//!
//! Slices are a view into a block of memory represented as a pointer and a
//! length.
//!
//! ```
//! // slicing a Vec
//! let vec = vec![1, 2, 3];
//! let int_slice = &vec[..];
//! // coercing an array to a slice
//! let str_slice: &[&str] = &["one", "two", "three"];
//! ```
//!
//! Slices are either mutable or shared. The shared slice type is `&[T]`,
//! while the mutable slice type is `&mut [T]`, where `T` represents the element
//! type. For example, you can mutate the block of memory that a mutable slice
//! points to:
//!
//! ```
//! let x = &mut [1, 2, 3];
//! x[1] = 7;
//! assert_eq!(x, &[1, 7, 3]);
//! ```
//!
//! Here are some of the things this module contains:
//!
//! ## Structs
//!
//! There are several structs that are useful for slices, such as [`Iter`], which
//! represents iteration over a slice.
//!
//! ## Trait Implementations
//!
//! There are several implementations of common traits for slices. Some examples
//! include:
//!
//! * [`Clone`]
//! * [`Eq`], [`Ord`] - for slices whose element type are [`Eq`] or [`Ord`].
//! * [`Hash`] - for slices whose element type is [`Hash`].
//!
//! ## Iteration
//!
//! The slices implement `IntoIterator`. The iterator yields references to the
//! slice elements.
//!
//! ```
//! let numbers = &[0, 1, 2];
//! for n in numbers {
//!     println!("{} is a number!", n);
//! }
//! ```
//!
//! The mutable slice yields mutable references to the elements:
//!
//! ```
//! let mut scores = [7, 8, 9];
//! for score in &mut scores[..] {
//!     *score += 1;
//! }
//! ```
//!
//! This iterator yields mutable references to the slice's elements, so while
//! the element type of the slice is `i32`, the element type of the iterator is
//! `&mut i32`.
//!
//! * [`.iter()`] and [`.iter_mut()`] are the explicit methods to return the default
//!   iterators.
//! * Further methods that return iterators are [`.split()`], [`.splitn()`],
//!   [`.chunks()`], [`.windows()`] and more.
//!
//! *[See also the slice primitive type](../../std/primitive.slice.html).*
//!
//! [`Clone`]: ../../std/clone/trait.Clone.html
//! [`Eq`]: ../../std/cmp/trait.Eq.html
//! [`Ord`]: ../../std/cmp/trait.Ord.html
//! [`Iter`]: struct.Iter.html
//! [`Hash`]: ../../std/hash/trait.Hash.html
//! [`.iter()`]: ../../std/primitive.slice.html#method.iter
//! [`.iter_mut()`]: ../../std/primitive.slice.html#method.iter_mut
//! [`.split()`]: ../../std/primitive.slice.html#method.split
//! [`.splitn()`]: ../../std/primitive.slice.html#method.splitn
//! [`.chunks()`]: ../../std/primitive.slice.html#method.chunks
//! [`.windows()`]: ../../std/primitive.slice.html#method.windows
#![stable(feature = "rust1", since = "1.0.0")]

// Many of the usings in this module are only used in the test configuration.
// It's cleaner to just turn off the unused_imports warning than to fix them.
#![cfg_attr(test, allow(unused_imports, dead_code))]

use alloc::boxed::Box;
use core::cmp::Ordering::{self, Greater};
use core::mem::size_of;
use core::mem;
use core::ptr;
use core::slice as core_slice;

use borrow::{Borrow, BorrowMut, ToOwned};
use vec::Vec;

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{Chunks, Windows};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{Iter, IterMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{SplitMut, ChunksMut, Split};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{SplitN, RSplitN, SplitNMut, RSplitNMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{from_raw_parts, from_raw_parts_mut};
#[unstable(feature = "slice_get_slice", issue = "35729")]
pub use core::slice::SliceIndex;

////////////////////////////////////////////////////////////////////////////////
// Basic slice extension methods
////////////////////////////////////////////////////////////////////////////////

// HACK(japaric) needed for the implementation of `vec!` macro during testing
// NB see the hack module in this file for more details
#[cfg(test)]
pub use self::hack::into_vec;

// HACK(japaric) needed for the implementation of `Vec::clone` during testing
// NB see the hack module in this file for more details
#[cfg(test)]
pub use self::hack::to_vec;

// HACK(japaric): With cfg(test) `impl [T]` is not available, these three
// functions are actually methods that are in `impl [T]` but not in
// `core::slice::SliceExt` - we need to supply these functions for the
// `test_permutations` test
mod hack {
    use alloc::boxed::Box;
    use core::mem;

    #[cfg(test)]
    use string::ToString;
    use vec::Vec;

    pub fn into_vec<T>(mut b: Box<[T]>) -> Vec<T> {
        unsafe {
            let xs = Vec::from_raw_parts(b.as_mut_ptr(), b.len(), b.len());
            mem::forget(b);
            xs
        }
    }

    #[inline]
    pub fn to_vec<T>(s: &[T]) -> Vec<T>
        where T: Clone
    {
        let mut vector = Vec::with_capacity(s.len());
        vector.extend_from_slice(s);
        vector
    }
}

#[lang = "slice"]
#[cfg(not(test))]
impl<T> [T] {
    /// Returns the number of elements in the slice.
    ///
    /// # Example
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len(&self) -> usize {
        core_slice::SliceExt::len(self)
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Example
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert!(!a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_empty(&self) -> bool {
        core_slice::SliceExt::is_empty(self)
    }

    /// Returns the first element of a slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&10), v.first());
    ///
    /// let w: &[i32] = &[];
    /// assert_eq!(None, w.first());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn first(&self) -> Option<&T> {
        core_slice::SliceExt::first(self)
    }

    /// Returns a mutable pointer to the first element of a slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(first) = x.first_mut() {
    ///     *first = 5;
    /// }
    /// assert_eq!(x, &[5, 1, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        core_slice::SliceExt::first_mut(self)
    }

    /// Returns the first and all the rest of the elements of a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[0, 1, 2];
    ///
    /// if let Some((first, elements)) = x.split_first() {
    ///     assert_eq!(first, &0);
    ///     assert_eq!(elements, &[1, 2]);
    /// }
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[inline]
    pub fn split_first(&self) -> Option<(&T, &[T])> {
        core_slice::SliceExt::split_first(self)
    }

    /// Returns the first and all the rest of the elements of a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some((first, elements)) = x.split_first_mut() {
    ///     *first = 3;
    ///     elements[0] = 4;
    ///     elements[1] = 5;
    /// }
    /// assert_eq!(x, &[3, 4, 5]);
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[inline]
    pub fn split_first_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        core_slice::SliceExt::split_first_mut(self)
    }

    /// Returns the last and all the rest of the elements of a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[0, 1, 2];
    ///
    /// if let Some((last, elements)) = x.split_last() {
    ///     assert_eq!(last, &2);
    ///     assert_eq!(elements, &[0, 1]);
    /// }
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[inline]
    pub fn split_last(&self) -> Option<(&T, &[T])> {
        core_slice::SliceExt::split_last(self)

    }

    /// Returns the last and all the rest of the elements of a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some((last, elements)) = x.split_last_mut() {
    ///     *last = 3;
    ///     elements[0] = 4;
    ///     elements[1] = 5;
    /// }
    /// assert_eq!(x, &[4, 5, 3]);
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[inline]
    pub fn split_last_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        core_slice::SliceExt::split_last_mut(self)
    }

    /// Returns the last element of a slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&30), v.last());
    ///
    /// let w: &[i32] = &[];
    /// assert_eq!(None, w.last());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn last(&self) -> Option<&T> {
        core_slice::SliceExt::last(self)
    }

    /// Returns a mutable pointer to the last item in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(last) = x.last_mut() {
    ///     *last = 10;
    /// }
    /// assert_eq!(x, &[0, 1, 10]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        core_slice::SliceExt::last_mut(self)
    }

    /// Returns a reference to an element or subslice depending on the type of
    /// index.
    ///
    /// - If given a position, returns a reference to the element at that
    ///   position or `None` if out of bounds.
    /// - If given a range, returns the subslice corresponding to that range,
    ///   or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&40), v.get(1));
    /// assert_eq!(Some(&[10, 40][..]), v.get(0..2));
    /// assert_eq!(None, v.get(3));
    /// assert_eq!(None, v.get(0..4));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
        where I: SliceIndex<T>
    {
        core_slice::SliceExt::get(self, index)
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index (see [`get()`]) or `None` if the index is out of bounds.
    ///
    /// [`get()`]: #method.get
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(elem) = x.get_mut(1) {
    ///     *elem = 42;
    /// }
    /// assert_eq!(x, &[0, 42, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
        where I: SliceIndex<T>
    {
        core_slice::SliceExt::get_mut(self, index)
    }

    /// Returns a reference to an element or subslice, without doing bounds
    /// checking. So use it very carefully!
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(x.get_unchecked(1), &2);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<T>
    {
        core_slice::SliceExt::get_unchecked(self, index)
    }

    /// Returns a mutable reference to an element or subslice, without doing
    /// bounds checking. So use it very carefully!
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [1, 2, 4];
    ///
    /// unsafe {
    ///     let elem = x.get_unchecked_mut(1);
    ///     *elem = 13;
    /// }
    /// assert_eq!(x, &[1, 13, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
        where I: SliceIndex<T>
    {
        core_slice::SliceExt::get_unchecked_mut(self, index)
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the slice may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[1, 2, 4];
    /// let x_ptr = x.as_ptr();
    ///
    /// unsafe {
    ///     for i in 0..x.len() {
    ///         assert_eq!(x.get_unchecked(i), &*x_ptr.offset(i as isize));
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        core_slice::SliceExt::as_ptr(self)
    }

    /// Returns an unsafe mutable pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the slice may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [1, 2, 4];
    /// let x_ptr = x.as_mut_ptr();
    ///
    /// unsafe {
    ///     for i in 0..x.len() {
    ///         *x_ptr.offset(i as isize) += 2;
    ///     }
    /// }
    /// assert_eq!(x, &[3, 4, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        core_slice::SliceExt::as_mut_ptr(self)
    }

    /// Swaps two elements in a slice.
    ///
    /// # Arguments
    ///
    /// * a - The index of the first element
    /// * b - The index of the second element
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = ["a", "b", "c", "d"];
    /// v.swap(1, 3);
    /// assert!(v == ["a", "d", "c", "b"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        core_slice::SliceExt::swap(self, a, b)
    }

    /// Reverse the order of elements in a slice, in place.
    ///
    /// # Example
    ///
    /// ```
    /// let mut v = [1, 2, 3];
    /// v.reverse();
    /// assert!(v == [3, 2, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn reverse(&mut self) {
        core_slice::SliceExt::reverse(self)
    }

    /// Returns an iterator over the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[1, 2, 4];
    /// let mut iterator = x.iter();
    ///
    /// assert_eq!(iterator.next(), Some(&1));
    /// assert_eq!(iterator.next(), Some(&2));
    /// assert_eq!(iterator.next(), Some(&4));
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn iter(&self) -> Iter<T> {
        core_slice::SliceExt::iter(self)
    }

    /// Returns an iterator that allows modifying each value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [1, 2, 4];
    /// for elem in x.iter_mut() {
    ///     *elem += 2;
    /// }
    /// assert_eq!(x, &[3, 4, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        core_slice::SliceExt::iter_mut(self)
    }

    /// Returns an iterator over all contiguous windows of length
    /// `size`. The windows overlap. If the slice is shorter than
    /// `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// let slice = ['r', 'u', 's', 't'];
    /// let mut iter = slice.windows(2);
    /// assert_eq!(iter.next().unwrap(), &['r', 'u']);
    /// assert_eq!(iter.next().unwrap(), &['u', 's']);
    /// assert_eq!(iter.next().unwrap(), &['s', 't']);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If the slice is shorter than `size`:
    ///
    /// ```
    /// let slice = ['f', 'o', 'o'];
    /// let mut iter = slice.windows(4);
    /// assert!(iter.next().is_none());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn windows(&self, size: usize) -> Windows<T> {
        core_slice::SliceExt::windows(self, size)
    }

    /// Returns an iterator over `size` elements of the slice at a
    /// time. The chunks are slices and do not overlap. If `size` does
    /// not divide the length of the slice, then the last chunk will
    /// not have length `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.chunks(2);
    /// assert_eq!(iter.next().unwrap(), &['l', 'o']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e']);
    /// assert_eq!(iter.next().unwrap(), &['m']);
    /// assert!(iter.next().is_none());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chunks(&self, size: usize) -> Chunks<T> {
        core_slice::SliceExt::chunks(self, size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last chunk will not
    /// have length `chunk_size`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.chunks_mut(2) {
    ///     for elem in chunk.iter_mut() {
    ///         *elem += count;
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[1, 1, 2, 2, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        core_slice::SliceExt::chunks_mut(self, chunk_size)
    }

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 50];
    /// let (v1, v2) = v.split_at(2);
    /// assert_eq!([10, 40], v1);
    /// assert_eq!([30, 20, 50], v2);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        core_slice::SliceExt::split_at(self, mid)
    }

    /// Divides one `&mut` into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 2, 3, 4, 5, 6];
    ///
    /// // scoped to restrict the lifetime of the borrows
    /// {
    ///    let (left, right) = v.split_at_mut(0);
    ///    assert!(left == []);
    ///    assert!(right == [1, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at_mut(2);
    ///     assert!(left == [1, 2]);
    ///     assert!(right == [3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at_mut(6);
    ///     assert!(left == [1, 2, 3, 4, 5, 6]);
    ///     assert!(right == []);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        core_slice::SliceExt::split_at_mut(self, mid)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`. The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = [10, 40, 33, 20];
    /// let mut iter = slice.split(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10, 40]);
    /// assert_eq!(iter.next().unwrap(), &[20]);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If the first element is matched, an empty slice will be the first item
    /// returned by the iterator. Similarly, if the last element in the slice
    /// is matched, an empty slice will be the last item returned by the
    /// iterator:
    ///
    /// ```
    /// let slice = [10, 40, 33];
    /// let mut iter = slice.split(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10, 40]);
    /// assert_eq!(iter.next().unwrap(), &[]);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If two matched elements are directly adjacent, an empty slice will be
    /// present between them:
    ///
    /// ```
    /// let slice = [10, 6, 33, 20];
    /// let mut iter = slice.split(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10]);
    /// assert_eq!(iter.next().unwrap(), &[]);
    /// assert_eq!(iter.next().unwrap(), &[20]);
    /// assert!(iter.next().is_none());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split<F>(&self, pred: F) -> Split<T, F>
        where F: FnMut(&T) -> bool
    {
        core_slice::SliceExt::split(self, pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`. The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.split_mut(|num| *num % 3 == 0) {
    ///     group[0] = 1;
    /// }
    /// assert_eq!(v, [1, 40, 30, 1, 60, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_mut<F>(&mut self, pred: F) -> SplitMut<T, F>
        where F: FnMut(&T) -> bool
    {
        core_slice::SliceExt::split_mut(self, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to returning at most `n` items. The matched element is
    /// not contained in the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// Print the slice split once by numbers divisible by 3 (i.e. `[10, 40]`,
    /// `[20, 60, 50]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.splitn(2, |num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn<F>(&self, n: usize, pred: F) -> SplitN<T, F>
        where F: FnMut(&T) -> bool
    {
        core_slice::SliceExt::splitn(self, n, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to returning at most `n` items. The matched element is
    /// not contained in the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.splitn_mut(2, |num| *num % 3 == 0) {
    ///     group[0] = 1;
    /// }
    /// assert_eq!(v, [1, 40, 30, 1, 60, 50]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn_mut<F>(&mut self, n: usize, pred: F) -> SplitNMut<T, F>
        where F: FnMut(&T) -> bool
    {
        core_slice::SliceExt::splitn_mut(self, n, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to returning at most `n` items. This starts at the end of
    /// the slice and works backwards.  The matched element is not contained in
    /// the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// Print the slice split once, starting from the end, by numbers divisible
    /// by 3 (i.e. `[50]`, `[10, 40, 30, 20]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.rsplitn(2, |num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn<F>(&self, n: usize, pred: F) -> RSplitN<T, F>
        where F: FnMut(&T) -> bool
    {
        core_slice::SliceExt::rsplitn(self, n, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to returning at most `n` items. This starts at the end of
    /// the slice and works backwards. The matched element is not contained in
    /// the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in s.rsplitn_mut(2, |num| *num % 3 == 0) {
    ///     group[0] = 1;
    /// }
    /// assert_eq!(s, [1, 40, 30, 20, 60, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn_mut<F>(&mut self, n: usize, pred: F) -> RSplitNMut<T, F>
        where F: FnMut(&T) -> bool
    {
        core_slice::SliceExt::rsplitn_mut(self, n, pred)
    }

    /// Returns `true` if the slice contains an element with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.contains(&30));
    /// assert!(!v.contains(&50));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains(&self, x: &T) -> bool
        where T: PartialEq
    {
        core_slice::SliceExt::contains(self, x)
    }

    /// Returns `true` if `needle` is a prefix of the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.starts_with(&[10]));
    /// assert!(v.starts_with(&[10, 40]));
    /// assert!(!v.starts_with(&[50]));
    /// assert!(!v.starts_with(&[10, 50]));
    /// ```
    ///
    /// Always returns `true` if `needle` is an empty slice:
    ///
    /// ```
    /// let v = &[10, 40, 30];
    /// assert!(v.starts_with(&[]));
    /// let v: &[u8] = &[];
    /// assert!(v.starts_with(&[]));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn starts_with(&self, needle: &[T]) -> bool
        where T: PartialEq
    {
        core_slice::SliceExt::starts_with(self, needle)
    }

    /// Returns `true` if `needle` is a suffix of the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.ends_with(&[30]));
    /// assert!(v.ends_with(&[40, 30]));
    /// assert!(!v.ends_with(&[50]));
    /// assert!(!v.ends_with(&[50, 30]));
    /// ```
    ///
    /// Always returns `true` if `needle` is an empty slice:
    ///
    /// ```
    /// let v = &[10, 40, 30];
    /// assert!(v.ends_with(&[]));
    /// let v: &[u8] = &[];
    /// assert!(v.ends_with(&[]));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ends_with(&self, needle: &[T]) -> bool
        where T: PartialEq
    {
        core_slice::SliceExt::ends_with(self, needle)
    }

    /// Binary search a sorted slice for a given element.
    ///
    /// If the value is found then `Ok` is returned, containing the
    /// index of the matching element; if the value is not found then
    /// `Err` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// # Example
    ///
    /// Looks up a series of four elements. The first is found, with a
    /// uniquely determined position; the second and third are not
    /// found; the fourth could match any position in `[1, 4]`.
    ///
    /// ```
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    ///
    /// assert_eq!(s.binary_search(&13),  Ok(9));
    /// assert_eq!(s.binary_search(&4),   Err(7));
    /// assert_eq!(s.binary_search(&100), Err(13));
    /// let r = s.binary_search(&1);
    /// assert!(match r { Ok(1...4) => true, _ => false, });
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
        where T: Ord
    {
        core_slice::SliceExt::binary_search(self, x)
    }

    /// Binary search a sorted slice with a comparator function.
    ///
    /// The comparator function should implement an order consistent
    /// with the sort order of the underlying slice, returning an
    /// order code that indicates whether its argument is `Less`,
    /// `Equal` or `Greater` the desired target.
    ///
    /// If a matching value is found then returns `Ok`, containing
    /// the index for the matched element; if no match is found then
    /// `Err` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// # Example
    ///
    /// Looks up a series of four elements. The first is found, with a
    /// uniquely determined position; the second and third are not
    /// found; the fourth could match any position in `[1, 4]`.
    ///
    /// ```
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    ///
    /// let seek = 13;
    /// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Ok(9));
    /// let seek = 4;
    /// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Err(7));
    /// let seek = 100;
    /// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Err(13));
    /// let seek = 1;
    /// let r = s.binary_search_by(|probe| probe.cmp(&seek));
    /// assert!(match r { Ok(1...4) => true, _ => false, });
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn binary_search_by<'a, F>(&'a self, f: F) -> Result<usize, usize>
        where F: FnMut(&'a T) -> Ordering
    {
        core_slice::SliceExt::binary_search_by(self, f)
    }

    /// Binary search a sorted slice with a key extraction function.
    ///
    /// Assumes that the slice is sorted by the key, for instance with
    /// [`sort_by_key`] using the same key extraction function.
    ///
    /// If a matching value is found then returns `Ok`, containing the
    /// index for the matched element; if no match is found then `Err`
    /// is returned, containing the index where a matching element could
    /// be inserted while maintaining sorted order.
    ///
    /// [`sort_by_key`]: #method.sort_by_key
    ///
    /// # Examples
    ///
    /// Looks up a series of four elements in a slice of pairs sorted by
    /// their second elements. The first is found, with a uniquely
    /// determined position; the second and third are not found; the
    /// fourth could match any position in `[1, 4]`.
    ///
    /// ```
    /// let s = [(0, 0), (2, 1), (4, 1), (5, 1), (3, 1),
    ///          (1, 2), (2, 3), (4, 5), (5, 8), (3, 13),
    ///          (1, 21), (2, 34), (4, 55)];
    ///
    /// assert_eq!(s.binary_search_by_key(&13, |&(a,b)| b),  Ok(9));
    /// assert_eq!(s.binary_search_by_key(&4, |&(a,b)| b),   Err(7));
    /// assert_eq!(s.binary_search_by_key(&100, |&(a,b)| b), Err(13));
    /// let r = s.binary_search_by_key(&1, |&(a,b)| b);
    /// assert!(match r { Ok(1...4) => true, _ => false, });
    /// ```
    #[stable(feature = "slice_binary_search_by_key", since = "1.10.0")]
    #[inline]
    pub fn binary_search_by_key<'a, B, F>(&'a self, b: &B, f: F) -> Result<usize, usize>
        where F: FnMut(&'a T) -> B,
              B: Ord
    {
        core_slice::SliceExt::binary_search_by_key(self, b, f)
    }

    /// This is equivalent to `self.sort_by(|a, b| a.cmp(b))`.
    ///
    /// This sort is stable and `O(n log n)` worst-case, but allocates
    /// temporary storage half the size of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5, 4, 1, -3, 2];
    ///
    /// v.sort();
    /// assert!(v == [-5, -3, 1, 2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sort(&mut self)
        where T: Ord
    {
        self.sort_by(|a, b| a.cmp(b))
    }

    /// Sorts the slice, in place, using `f` to extract a key by which to
    /// order the sort by.
    ///
    /// This sort is stable and `O(n log n)` worst-case, but allocates
    /// temporary storage half the size of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 1, -3, 2];
    ///
    /// v.sort_by_key(|k| k.abs());
    /// assert!(v == [1, 2, -3, 4, -5]);
    /// ```
    #[stable(feature = "slice_sort_by_key", since = "1.7.0")]
    #[inline]
    pub fn sort_by_key<B, F>(&mut self, mut f: F)
        where F: FnMut(&T) -> B, B: Ord
    {
        self.sort_by(|a, b| f(a).cmp(&f(b)))
    }

    /// Sorts the slice, in place, using `compare` to compare
    /// elements.
    ///
    /// This sort is stable and `O(n log n)` worst-case, but allocates
    /// temporary storage half the size of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [5, 4, 1, 3, 2];
    /// v.sort_by(|a, b| a.cmp(b));
    /// assert!(v == [1, 2, 3, 4, 5]);
    ///
    /// // reverse sorting
    /// v.sort_by(|a, b| b.cmp(a));
    /// assert!(v == [5, 4, 3, 2, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sort_by<F>(&mut self, compare: F)
        where F: FnMut(&T, &T) -> Ordering
    {
        merge_sort(self, compare)
    }

    /// Copies the elements from `src` into `self`.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// let mut dst = [0, 0, 0];
    /// let src = [1, 2, 3];
    ///
    /// dst.clone_from_slice(&src);
    /// assert!(dst == [1, 2, 3]);
    /// ```
    #[stable(feature = "clone_from_slice", since = "1.7.0")]
    pub fn clone_from_slice(&mut self, src: &[T]) where T: Clone {
        core_slice::SliceExt::clone_from_slice(self, src)
    }

    /// Copies all elements from `src` into `self`, using a memcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// let mut dst = [0, 0, 0];
    /// let src = [1, 2, 3];
    ///
    /// dst.copy_from_slice(&src);
    /// assert_eq!(src, dst);
    /// ```
    #[stable(feature = "copy_from_slice", since = "1.9.0")]
    pub fn copy_from_slice(&mut self, src: &[T]) where T: Copy {
        core_slice::SliceExt::copy_from_slice(self, src)
    }


    /// Copies `self` into a new `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = [10, 40, 30];
    /// let x = s.to_vec();
    /// // Here, `s` and `x` can be modified independently.
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_vec(&self) -> Vec<T>
        where T: Clone
    {
        // NB see hack module in this file
        hack::to_vec(self)
    }

    /// Converts `self` into a vector without clones or allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// let s: Box<[i32]> = Box::new([10, 40, 30]);
    /// let x = s.into_vec();
    /// // `s` cannot be used anymore because it has been converted into `x`.
    ///
    /// assert_eq!(x, vec![10, 40, 30]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn into_vec(self: Box<Self>) -> Vec<T> {
        // NB see hack module in this file
        hack::into_vec(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Extension traits for slices over specific kinds of data
////////////////////////////////////////////////////////////////////////////////
#[unstable(feature = "slice_concat_ext",
           reason = "trait should not have to exist",
           issue = "27747")]
/// An extension trait for concatenating slices
pub trait SliceConcatExt<T: ?Sized> {
    #[unstable(feature = "slice_concat_ext",
               reason = "trait should not have to exist",
               issue = "27747")]
    /// The resulting type after concatenation
    type Output;

    /// Flattens a slice of `T` into a single value `Self::Output`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(["hello", "world"].concat(), "helloworld");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn concat(&self) -> Self::Output;

    /// Flattens a slice of `T` into a single value `Self::Output`, placing a
    /// given separator between each.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(["hello", "world"].join(" "), "hello world");
    /// ```
    #[stable(feature = "rename_connect_to_join", since = "1.3.0")]
    fn join(&self, sep: &T) -> Self::Output;

    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.3.0", reason = "renamed to join")]
    fn connect(&self, sep: &T) -> Self::Output;
}

#[unstable(feature = "slice_concat_ext",
           reason = "trait should not have to exist",
           issue = "27747")]
impl<T: Clone, V: Borrow<[T]>> SliceConcatExt<T> for [V] {
    type Output = Vec<T>;

    fn concat(&self) -> Vec<T> {
        let size = self.iter().fold(0, |acc, v| acc + v.borrow().len());
        let mut result = Vec::with_capacity(size);
        for v in self {
            result.extend_from_slice(v.borrow())
        }
        result
    }

    fn join(&self, sep: &T) -> Vec<T> {
        let size = self.iter().fold(0, |acc, v| acc + v.borrow().len());
        let mut result = Vec::with_capacity(size + self.len());
        let mut first = true;
        for v in self {
            if first {
                first = false
            } else {
                result.push(sep.clone())
            }
            result.extend_from_slice(v.borrow())
        }
        result
    }

    fn connect(&self, sep: &T) -> Vec<T> {
        self.join(sep)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Standard trait implementations for slices
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Borrow<[T]> for Vec<T> {
    fn borrow(&self) -> &[T] {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> BorrowMut<[T]> for Vec<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> ToOwned for [T] {
    type Owned = Vec<T>;
    #[cfg(not(test))]
    fn to_owned(&self) -> Vec<T> {
        self.to_vec()
    }

    // HACK(japaric): with cfg(test) the inherent `[T]::to_vec`, which is required for this method
    // definition, is not available. Since we don't require this method for testing purposes, I'll
    // just stub it
    // NB see the slice::hack module in slice.rs for more information
    #[cfg(test)]
    fn to_owned(&self) -> Vec<T> {
        panic!("not available with cfg(test)")
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sorting
////////////////////////////////////////////////////////////////////////////////

/// Inserts `v[0]` into pre-sorted sequence `v[1..]` so that whole `v[..]` becomes sorted.
///
/// This is the integral subroutine of insertion sort.
fn insert_head<T, F>(v: &mut [T], compare: &mut F)
    where F: FnMut(&T, &T) -> Ordering
{
    if v.len() >= 2 && compare(&v[0], &v[1]) == Greater {
        unsafe {
            // There are three ways to implement insertion here:
            //
            // 1. Swap adjacent elements until the first one gets to its final destination.
            //    However, this way we copy data around more than is necessary. If elements are big
            //    structures (costly to copy), this method will be slow.
            //
            // 2. Iterate until the right place for the first element is found. Then shift the
            //    elements succeeding it to make room for it and finally place it into the
            //    remaining hole. This is a good method.
            //
            // 3. Copy the first element into a temporary variable. Iterate until the right place
            //    for it is found. As we go along, copy every traversed element into the slot
            //    preceding it. Finally, copy data from the temporary variable into the remaining
            //    hole. This method is very good. Benchmarks demonstrated slightly better
            //    performance than with the 2nd method.
            //
            // All methods were benchmarked, and the 3rd showed best results. So we chose that one.
            let mut tmp = NoDrop { value: ptr::read(&v[0]) };

            // Intermediate state of the insertion process is always tracked by `hole`, which
            // serves two purposes:
            // 1. Protects integrity of `v` from panics in `compare`.
            // 2. Fills the remaining hole in `v` in the end.
            //
            // Panic safety:
            //
            // If `compare` panics at any point during the process, `hole` will get dropped and
            // fill the hole in `v` with `tmp`, thus ensuring that `v` still holds every object it
            // initially held exactly once.
            let mut hole = InsertionHole {
                src: &mut tmp.value,
                dest: &mut v[1],
            };
            ptr::copy_nonoverlapping(&v[1], &mut v[0], 1);

            for i in 2..v.len() {
                if compare(&tmp.value, &v[i]) != Greater {
                    break;
                }
                ptr::copy_nonoverlapping(&v[i], &mut v[i - 1], 1);
                hole.dest = &mut v[i];
            }
            // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
        }
    }

    // Holds a value, but never drops it.
    #[allow(unions_with_drop_fields)]
    union NoDrop<T> {
        value: T
    }

    // When dropped, copies from `src` into `dest`.
    struct InsertionHole<T> {
        src: *mut T,
        dest: *mut T,
    }

    impl<T> Drop for InsertionHole<T> {
        fn drop(&mut self) {
            unsafe { ptr::copy_nonoverlapping(self.src, self.dest, 1); }
        }
    }
}

/// Merges non-decreasing runs `v[..mid]` and `v[mid..]` using `buf` as temporary storage, and
/// stores the result into `v[..]`.
///
/// # Safety
///
/// The two slices must be non-empty and `mid` must be in bounds. Buffer `buf` must be long enough
/// to hold a copy of the shorter slice. Also, `T` must not be a zero-sized type.
unsafe fn merge<T, F>(v: &mut [T], mid: usize, buf: *mut T, compare: &mut F)
    where F: FnMut(&T, &T) -> Ordering
{
    let len = v.len();
    let v = v.as_mut_ptr();
    let v_mid = v.offset(mid as isize);
    let v_end = v.offset(len as isize);

    // The merge process first copies the shorter run into `buf`. Then it traces the newly copied
    // run and the longer run forwards (or backwards), comparing their next unconsumed elements and
    // copying the lesser (or greater) one into `v`.
    //
    // As soon as the shorter run is fully consumed, the process is done. If the longer run gets
    // consumed first, then we must copy whatever is left of the shorter run into the remaining
    // hole in `v`.
    //
    // Intermediate state of the process is always tracked by `hole`, which serves two purposes:
    // 1. Protects integrity of `v` from panics in `compare`.
    // 2. Fills the remaining hole in `v` if the longer run gets consumed first.
    //
    // Panic safety:
    //
    // If `compare` panics at any point during the process, `hole` will get dropped and fill the
    // hole in `v` with the unconsumed range in `buf`, thus ensuring that `v` still holds every
    // object it initially held exactly once.
    let mut hole;

    if mid <= len - mid {
        // The left run is shorter.
        ptr::copy_nonoverlapping(v, buf, mid);
        hole = MergeHole {
            start: buf,
            end: buf.offset(mid as isize),
            dest: v,
        };

        // Initially, these pointers point to the beginnings of their arrays.
        let left = &mut hole.start;
        let mut right = v_mid;
        let out = &mut hole.dest;

        while *left < hole.end && right < v_end {
            // Consume the lesser side.
            // If equal, prefer the left run to maintain stability.
            let to_copy = if compare(&**left, &*right) == Greater {
                get_and_increment(&mut right)
            } else {
                get_and_increment(left)
            };
            ptr::copy_nonoverlapping(to_copy, get_and_increment(out), 1);
        }
    } else {
        // The right run is shorter.
        ptr::copy_nonoverlapping(v_mid, buf, len - mid);
        hole = MergeHole {
            start: buf,
            end: buf.offset((len - mid) as isize),
            dest: v_mid,
        };

        // Initially, these pointers point past the ends of their arrays.
        let left = &mut hole.dest;
        let right = &mut hole.end;
        let mut out = v_end;

        while v < *left && buf < *right {
            // Consume the greater side.
            // If equal, prefer the right run to maintain stability.
            let to_copy = if compare(&*left.offset(-1), &*right.offset(-1)) == Greater {
                decrement_and_get(left)
            } else {
                decrement_and_get(right)
            };
            ptr::copy_nonoverlapping(to_copy, decrement_and_get(&mut out), 1);
        }
    }
    // Finally, `hole` gets dropped. If the shorter run was not fully consumed, whatever remains of
    // it will now be copied into the hole in `v`.

    unsafe fn get_and_increment<T>(ptr: &mut *mut T) -> *mut T {
        let old = *ptr;
        *ptr = ptr.offset(1);
        old
    }

    unsafe fn decrement_and_get<T>(ptr: &mut *mut T) -> *mut T {
        *ptr = ptr.offset(-1);
        *ptr
    }

    // When dropped, copies the range `start..end` into `dest..`.
    struct MergeHole<T> {
        start: *mut T,
        end: *mut T,
        dest: *mut T,
    }

    impl<T> Drop for MergeHole<T> {
        fn drop(&mut self) {
            // `T` is not a zero-sized type, so it's okay to divide by it's size.
            let len = (self.end as usize - self.start as usize) / mem::size_of::<T>();
            unsafe { ptr::copy_nonoverlapping(self.start, self.dest, len); }
        }
    }
}

/// This merge sort borrows some (but not all) ideas from TimSort, which is described in detail
/// [here](http://svn.python.org/projects/python/trunk/Objects/listsort.txt).
///
/// The algorithm identifies strictly descending and non-descending subsequences, which are called
/// natural runs. There is a stack of pending runs yet to be merged. Each newly found run is pushed
/// onto the stack, and then some pairs of adjacent runs are merged until these two invariants are
/// satisfied:
///
/// 1. for every `i` in `1..runs.len()`: `runs[i - 1].len > runs[i].len`
/// 2. for every `i` in `2..runs.len()`: `runs[i - 2].len > runs[i - 1].len + runs[i].len`
///
/// The invariants ensure that the total running time is `O(n log n)` worst-case.
fn merge_sort<T, F>(v: &mut [T], mut compare: F)
    where F: FnMut(&T, &T) -> Ordering
{
    // Sorting has no meaningful behavior on zero-sized types.
    if size_of::<T>() == 0 {
        return;
    }

    // FIXME #12092: These numbers are platform-specific and need more extensive testing/tuning.
    //
    // If `v` has length up to `insertion_len`, simply switch to insertion sort because it is going
    // to perform better than merge sort. For bigger types `T`, the threshold is smaller.
    //
    // Short runs are extended using insertion sort to span at least `min_run` elements, in order
    // to improve performance.
    let (max_insertion, min_run) = if size_of::<T>() <= 16 {
        (64, 32)
    } else {
        (32, 16)
    };

    let len = v.len();

    // Short arrays get sorted in-place via insertion sort to avoid allocations.
    if len <= max_insertion {
        if len >= 2 {
            for i in (0..len-1).rev() {
                insert_head(&mut v[i..], &mut compare);
            }
        }
        return;
    }

    // Allocate a buffer to use as scratch memory. We keep the length 0 so we can keep in it
    // shallow copies of the contents of `v` without risking the dtors running on copies if
    // `compare` panics. When merging two sorted runs, this buffer holds a copy of the shorter run,
    // which will always have length at most `len / 2`.
    let mut buf = Vec::with_capacity(len / 2);

    // In order to identify natural runs in `v`, we traverse it backwards. That might seem like a
    // strange decision, but consider the fact that merges more often go in the opposite direction
    // (forwards). According to benchmarks, merging forwards is slightly faster than merging
    // backwards. To conclude, identifying runs by traversing backwards improves performance.
    let mut runs = vec![];
    let mut end = len;
    while end > 0 {
        // Find the next natural run, and reverse it if it's strictly descending.
        let mut start = end - 1;
        if start > 0 {
            start -= 1;
            if compare(&v[start], &v[start + 1]) == Greater {
                while start > 0 && compare(&v[start - 1], &v[start]) == Greater {
                    start -= 1;
                }
                v[start..end].reverse();
            } else {
                while start > 0 && compare(&v[start - 1], &v[start]) != Greater {
                    start -= 1;
                }
            }
        }

        // Insert some more elements into the run if it's too short. Insertion sort is faster than
        // merge sort on short sequences, so this significantly improves performance.
        while start > 0 && end - start < min_run {
            start -= 1;
            insert_head(&mut v[start..end], &mut compare);
        }

        // Push this run onto the stack.
        runs.push(Run {
            start: start,
            len: end - start,
        });
        end = start;

        // Merge some pairs of adjacent runs to satisfy the invariants.
        while let Some(r) = collapse(&runs) {
            let left = runs[r + 1];
            let right = runs[r];
            unsafe {
                merge(&mut v[left.start .. right.start + right.len], left.len, buf.as_mut_ptr(),
                      &mut compare);
            }
            runs[r] = Run {
                start: left.start,
                len: left.len + right.len,
            };
            runs.remove(r + 1);
        }
    }

    // Finally, exactly one run must remain in the stack.
    debug_assert!(runs.len() == 1 && runs[0].start == 0 && runs[0].len == len);

    // Examines the stack of runs and identifies the next pair of runs to merge. More specifically,
    // if `Some(r)` is returned, that means `runs[r]` and `runs[r + 1]` must be merged next. If the
    // algorithm should continue building a new run instead, `None` is returned.
    //
    // TimSort is infamous for it's buggy implementations, as described here:
    // http://envisage-project.eu/timsort-specification-and-verification/
    //
    // The gist of the story is: we must enforce the invariants on the top four runs on the stack.
    // Enforcing them on just top three is not sufficient to ensure that the invariants will still
    // hold for *all* runs in the stack.
    //
    // This function correctly checks invariants for the top four runs. Additionally, if the top
    // run starts at index 0, it will always demand a merge operation until the stack is fully
    // collapsed, in order to complete the sort.
    #[inline]
    fn collapse(runs: &[Run]) -> Option<usize> {
        let n = runs.len();
        if n >= 2 && (runs[n - 1].start == 0 ||
                      runs[n - 2].len <= runs[n - 1].len ||
                      (n >= 3 && runs[n - 3].len <= runs[n - 2].len + runs[n - 1].len) ||
                      (n >= 4 && runs[n - 4].len <= runs[n - 3].len + runs[n - 2].len)) {
            if n >= 3 && runs[n - 3].len < runs[n - 1].len {
                Some(n - 3)
            } else {
                Some(n - 2)
            }
        } else {
            None
        }
    }

    #[derive(Clone, Copy)]
    struct Run {
        start: usize,
        len: usize,
    }
}
