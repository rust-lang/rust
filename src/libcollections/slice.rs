// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities for slice manipulation
//!
//! The `slice` module contains useful code to help work with slice values.
//! Slices are a view into a block of memory represented as a pointer and a length.
//!
//! ```rust
//! // slicing a Vec
//! let vec = vec!(1, 2, 3);
//! let int_slice = vec.as_slice();
//! // coercing an array to a slice
//! let str_slice: &[&str] = &["one", "two", "three"];
//! ```
//!
//! Slices are either mutable or shared. The shared slice type is `&[T]`,
//! while the mutable slice type is `&mut[T]`. For example, you can mutate the
//! block of memory that a mutable slice points to:
//!
//! ```rust
//! let x: &mut[i32] = &mut [1, 2, 3];
//! x[1] = 7;
//! assert_eq!(x[0], 1);
//! assert_eq!(x[1], 7);
//! assert_eq!(x[2], 3);
//! ```
//!
//! Here are some of the things this module contains:
//!
//! ## Structs
//!
//! There are several structs that are useful for slices, such as `Iter`, which
//! represents iteration over a slice.
//!
//! ## Traits
//!
//! A number of traits add methods that allow you to accomplish tasks
//! with slices, the most important being `SliceExt`. Other traits
//! apply only to slices of elements satisfying certain bounds (like
//! `Ord`).
//!
//! An example is the `slice` method which enables slicing syntax `[a..b]` that
//! returns an immutable "view" into a `Vec` or another slice from the index
//! interval `[a, b)`:
//!
//! ```rust
//! fn main() {
//!     let numbers = [0, 1, 2];
//!     let last_numbers = &numbers[1..3];
//!     // last_numbers is now &[1, 2]
//! }
//! ```
//!
//! ## Implementations of other traits
//!
//! There are several implementations of common traits for slices. Some examples
//! include:
//!
//! * `Clone`
//! * `Eq`, `Ord` - for immutable slices whose element type are `Eq` or `Ord`.
//! * `Hash` - for slices whose element type is `Hash`
//!
//! ## Iteration
//!
//! The method `iter()` returns an iteration value for a slice. The iterator
//! yields references to the slice's elements, so if the element
//! type of the slice is `isize`, the element type of the iterator is `&isize`.
//!
//! ```rust
//! let numbers = [0, 1, 2];
//! for &x in numbers.iter() {
//!     println!("{} is a number!", x);
//! }
//! ```
//!
//! * `.iter_mut()` returns an iterator that allows modifying each value.
//! * Further iterators exist that split, chunk or permute the slice.

#![doc(primitive = "slice")]
#![stable(feature = "rust1", since = "1.0.0")]

use alloc::boxed::Box;
use core::clone::Clone;
use core::cmp::Ordering::{self, Greater, Less};
use core::cmp::{self, Ord, PartialEq};
use core::iter::{Iterator, IteratorExt};
use core::iter::{range_step, MultiplicativeIterator};
use core::marker::Sized;
use core::mem::size_of;
use core::mem;
use core::ops::FnMut;
use core::option::Option::{self, Some, None};
use core::ptr::PtrExt;
use core::ptr;
use core::result::Result;
use core::slice as core_slice;
use self::Direction::*;

use borrow::{Borrow, BorrowMut, ToOwned};
use vec::Vec;

pub use core::slice::{Chunks, AsSlice, Windows};
pub use core::slice::{Iter, IterMut};
pub use core::slice::{IntSliceExt, SplitMut, ChunksMut, Split};
pub use core::slice::{SplitN, RSplitN, SplitNMut, RSplitNMut};
pub use core::slice::{bytes, mut_ref_slice, ref_slice};
pub use core::slice::{from_raw_parts, from_raw_parts_mut};
pub use core::slice::{from_raw_buf, from_raw_mut_buf};

////////////////////////////////////////////////////////////////////////////////
// Basic slice extension methods
////////////////////////////////////////////////////////////////////////////////

/// Allocating extension methods for slices.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait SliceExt {
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Sorts the slice, in place, using `compare` to compare
    /// elements.
    ///
    /// This sort is `O(n log n)` worst-case and stable, but allocates
    /// approximately `2 * n`, where `n` is the length of `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut v = [5, 4, 1, 3, 2];
    /// v.sort_by(|a, b| a.cmp(b));
    /// assert!(v == [1, 2, 3, 4, 5]);
    ///
    /// // reverse sorting
    /// v.sort_by(|a, b| b.cmp(a));
    /// assert!(v == [5, 4, 3, 2, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sort_by<F>(&mut self, compare: F) where F: FnMut(&Self::Item, &Self::Item) -> Ordering;

    /// Consumes `src` and moves as many elements as it can into `self`
    /// from the range [start,end).
    ///
    /// Returns the number of elements copied (the shorter of `self.len()`
    /// and `end - start`).
    ///
    /// # Arguments
    ///
    /// * src - A mutable vector of `T`
    /// * start - The index into `src` to start copying from
    /// * end - The index into `src` to stop copying from
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut a = [1, 2, 3, 4, 5];
    /// let b = vec![6, 7, 8];
    /// let num_moved = a.move_from(b, 0, 3);
    /// assert_eq!(num_moved, 3);
    /// assert!(a == [6, 7, 8, 4, 5]);
    /// ```
    #[unstable(feature = "collections",
               reason = "uncertain about this API approach")]
    fn move_from(&mut self, src: Vec<Self::Item>, start: usize, end: usize) -> usize;

    /// Deprecated: use `&s[start .. end]` notation instead.
    #[unstable(feature = "collections",
               reason = "will be replaced by slice syntax")]
    #[deprecated(since = "1.0.0", reason = "use &s[start .. end] instead")]
    fn slice(&self, start: usize, end: usize) -> &[Self::Item];

    /// Deprecated: use `&s[start..]` notation instead.
    #[unstable(feature = "collections",
               reason = "will be replaced by slice syntax")]
    #[deprecated(since = "1.0.0", reason = "use &s[start..] instead")]
    fn slice_from(&self, start: usize) -> &[Self::Item];

    /// Deprecated: use `&s[..end]` notation instead.
    #[unstable(feature = "collections",
               reason = "will be replaced by slice syntax")]
    #[deprecated(since = "1.0.0", reason = "use &s[..end] instead")]
    fn slice_to(&self, end: usize) -> &[Self::Item];

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
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
    fn split_at(&self, mid: usize) -> (&[Self::Item], &[Self::Item]);

    /// Returns an iterator over the slice.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn iter(&self) -> Iter<Self::Item>;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`.  The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// Print the slice split by numbers divisible by 3 (i.e. `[10, 40]`,
    /// `[20]`, `[50]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    /// for group in v.split(|num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn split<F>(&self, pred: F) -> Split<Self::Item, F>
                where F: FnMut(&Self::Item) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to splitting at most `n` times.  The matched element is
    /// not contained in the subslices.
    ///
    /// # Examples
    ///
    /// Print the slice split once by numbers divisible by 3 (i.e. `[10, 40]`,
    /// `[20, 60, 50]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    /// for group in v.splitn(1, |num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn splitn<F>(&self, n: usize, pred: F) -> SplitN<Self::Item, F>
                 where F: FnMut(&Self::Item) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to splitting at most `n` times. This starts at the end of
    /// the slice and works backwards.  The matched element is not contained in
    /// the subslices.
    ///
    /// # Examples
    ///
    /// Print the slice split once, starting from the end, by numbers divisible
    /// by 3 (i.e. `[50]`, `[10, 40, 30, 20]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    /// for group in v.rsplitn(1, |num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rsplitn<F>(&self, n: usize, pred: F) -> RSplitN<Self::Item, F>
                  where F: FnMut(&Self::Item) -> bool;

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
    /// Print the adjacent pairs of a slice (i.e. `[1,2]`, `[2,3]`,
    /// `[3,4]`):
    ///
    /// ```rust
    /// let v = &[1, 2, 3, 4];
    /// for win in v.windows(2) {
    ///     println!("{:?}", win);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn windows(&self, size: usize) -> Windows<Self::Item>;

    /// Returns an iterator over `size` elements of the slice at a
    /// time. The chunks do not overlap. If `size` does not divide the
    /// length of the slice, then the last chunk will not have length
    /// `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Example
    ///
    /// Print the slice two elements at a time (i.e. `[1,2]`,
    /// `[3,4]`, `[5]`):
    ///
    /// ```rust
    /// let v = &[1, 2, 3, 4, 5];
    /// for win in v.chunks(2) {
    ///     println!("{:?}", win);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chunks(&self, size: usize) -> Chunks<Self::Item>;

    /// Returns the element of a slice at the given index, or `None` if the
    /// index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&40), v.get(1));
    /// assert_eq!(None, v.get(3));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn get(&self, index: usize) -> Option<&Self::Item>;

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
    fn first(&self) -> Option<&Self::Item>;

    /// Returns all but the first element of a slice.
    #[unstable(feature = "collections", reason = "likely to be renamed")]
    fn tail(&self) -> &[Self::Item];

    /// Returns all but the last element of a slice.
    #[unstable(feature = "collections", reason = "likely to be renamed")]
    fn init(&self) -> &[Self::Item];

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
    fn last(&self) -> Option<&Self::Item>;

    /// Returns a pointer to the element at the given index, without doing
    /// bounds checking.
    #[stable(feature = "rust1", since = "1.0.0")]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item;

    /// Returns an unsafe pointer to the slice's buffer
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the slice may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_ptr(&self) -> *const Self::Item;

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
    /// found; the fourth could match any position in `[1,4]`.
    ///
    /// ```rust
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    /// let s = s.as_slice();
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
    fn binary_search_by<F>(&self, f: F) -> Result<usize, usize> where
        F: FnMut(&Self::Item) -> Ordering;

    /// Return the number of elements in the slice
    ///
    /// # Example
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn len(&self) -> usize;

    /// Returns true if the slice has a length of 0
    ///
    /// # Example
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Returns a mutable reference to the element at the given index,
    /// or `None` if the index is out of bounds
    #[stable(feature = "rust1", since = "1.0.0")]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item>;

    /// Work with `self` as a mut slice.
    /// Primarily intended for getting a &mut [T] from a [T; N].
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_mut_slice(&mut self) -> &mut [Self::Item];

    /// Deprecated: use `&mut s[start .. end]` instead.
    #[unstable(feature = "collections",
               reason = "will be replaced by slice syntax")]
    #[deprecated(since = "1.0.0", reason = "use &mut s[start .. end] instead")]
    fn slice_mut(&mut self, start: usize, end: usize) -> &mut [Self::Item];

    /// Deprecated: use `&mut s[start ..]` instead.
    #[unstable(feature = "collections",
               reason = "will be replaced by slice syntax")]
    #[deprecated(since = "1.0.0", reason = "use &mut s[start ..] instead")]
    fn slice_from_mut(&mut self, start: usize) -> &mut [Self::Item];

    /// Deprecated: use `&mut s[.. end]` instead.
    #[unstable(feature = "collections",
               reason = "will be replaced by slice syntax")]
    #[deprecated(since = "1.0.0", reason = "use &mut s[.. end] instead")]
    fn slice_to_mut(&mut self, end: usize) -> &mut [Self::Item];

    /// Returns an iterator that allows modifying each value
    #[stable(feature = "rust1", since = "1.0.0")]
    fn iter_mut(&mut self) -> IterMut<Self::Item>;

    /// Returns a mutable pointer to the first element of a slice, or `None` if it is empty
    #[stable(feature = "rust1", since = "1.0.0")]
    fn first_mut(&mut self) -> Option<&mut Self::Item>;

    /// Returns all but the first element of a mutable slice
    #[unstable(feature = "collections",
               reason = "likely to be renamed or removed")]
    fn tail_mut(&mut self) -> &mut [Self::Item];

    /// Returns all but the last element of a mutable slice
    #[unstable(feature = "collections",
               reason = "likely to be renamed or removed")]
    fn init_mut(&mut self) -> &mut [Self::Item];

    /// Returns a mutable pointer to the last item in the slice.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn last_mut(&mut self) -> Option<&mut Self::Item>;

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`.  The matched element is not contained in the subslices.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn split_mut<F>(&mut self, pred: F) -> SplitMut<Self::Item, F>
                    where F: FnMut(&Self::Item) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to splitting at most `n` times.  The matched element is
    /// not contained in the subslices.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn splitn_mut<F>(&mut self, n: usize, pred: F) -> SplitNMut<Self::Item, F>
                     where F: FnMut(&Self::Item) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to splitting at most `n` times. This starts at the end of
    /// the slice and works backwards.  The matched element is not contained in
    /// the subslices.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rsplitn_mut<F>(&mut self,  n: usize, pred: F) -> RSplitNMut<Self::Item, F>
                      where F: FnMut(&Self::Item) -> bool;

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    /// The chunks are mutable and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last chunk will not
    /// have length `chunk_size`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<Self::Item>;

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
    /// # Example
    ///
    /// ```rust
    /// let mut v = ["a", "b", "c", "d"];
    /// v.swap(1, 3);
    /// assert!(v == ["a", "d", "c", "b"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn swap(&mut self, a: usize, b: usize);

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
    /// # Example
    ///
    /// ```rust
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
    fn split_at_mut(&mut self, mid: usize) -> (&mut [Self::Item], &mut [Self::Item]);

    /// Reverse the order of elements in a slice, in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [1, 2, 3];
    /// v.reverse();
    /// assert!(v == [3, 2, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn reverse(&mut self);

    /// Returns an unsafe mutable pointer to the element in index
    #[stable(feature = "rust1", since = "1.0.0")]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;

    /// Return an unsafe mutable pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the slice may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_mut_ptr(&mut self) -> *mut Self::Item;

    /// Copies `self` into a new `Vec`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_vec(&self) -> Vec<Self::Item> where Self::Item: Clone;

    /// Creates an iterator that yields every possible permutation of the
    /// vector in succession.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let v = [1, 2, 3];
    /// let mut perms = v.permutations();
    ///
    /// for p in perms {
    ///   println!("{:?}", p);
    /// }
    /// ```
    ///
    /// Iterating through permutations one by one.
    ///
    /// ```rust
    /// let v = [1, 2, 3];
    /// let mut perms = v.permutations();
    ///
    /// assert_eq!(Some(vec![1, 2, 3]), perms.next());
    /// assert_eq!(Some(vec![1, 3, 2]), perms.next());
    /// assert_eq!(Some(vec![3, 1, 2]), perms.next());
    /// ```
    #[unstable(feature = "collections")]
    fn permutations(&self) -> Permutations<Self::Item> where Self::Item: Clone;

    /// Copies as many elements from `src` as it can into `self` (the
    /// shorter of `self.len()` and `src.len()`). Returns the number
    /// of elements copied.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut dst = [0, 0, 0];
    /// let src = [1, 2];
    ///
    /// assert!(dst.clone_from_slice(&src) == 2);
    /// assert!(dst == [1, 2, 0]);
    ///
    /// let src2 = [3, 4, 5, 6];
    /// assert!(dst.clone_from_slice(&src2) == 3);
    /// assert!(dst == [3, 4, 5]);
    /// ```
    #[unstable(feature = "collections")]
    fn clone_from_slice(&mut self, &[Self::Item]) -> usize where Self::Item: Clone;

    /// Sorts the slice, in place.
    ///
    /// This is equivalent to `self.sort_by(|a, b| a.cmp(b))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut v = [-5, 4, 1, -3, 2];
    ///
    /// v.sort();
    /// assert!(v == [-5, -3, 1, 2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sort(&mut self) where Self::Item: Ord;

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
    /// found; the fourth could match any position in `[1,4]`.
    ///
    /// ```rust
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    /// let s = s.as_slice();
    ///
    /// assert_eq!(s.binary_search(&13),  Ok(9));
    /// assert_eq!(s.binary_search(&4),   Err(7));
    /// assert_eq!(s.binary_search(&100), Err(13));
    /// let r = s.binary_search(&1);
    /// assert!(match r { Ok(1...4) => true, _ => false, });
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn binary_search(&self, x: &Self::Item) -> Result<usize, usize> where Self::Item: Ord;

    /// Deprecated: use `binary_search` instead.
    #[unstable(feature = "collections")]
    #[deprecated(since = "1.0.0", reason = "use binary_search instead")]
    fn binary_search_elem(&self, x: &Self::Item) -> Result<usize, usize> where Self::Item: Ord {
        self.binary_search(x)
    }

    /// Mutates the slice to the next lexicographic permutation.
    ///
    /// Returns `true` if successful and `false` if the slice is at the
    /// last-ordered permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: &mut [_] = &mut [0, 1, 2];
    /// v.next_permutation();
    /// let b: &mut [_] = &mut [0, 2, 1];
    /// assert!(v == b);
    /// v.next_permutation();
    /// let b: &mut [_] = &mut [1, 0, 2];
    /// assert!(v == b);
    /// ```
    #[unstable(feature = "collections",
               reason = "uncertain if this merits inclusion in std")]
    fn next_permutation(&mut self) -> bool where Self::Item: Ord;

    /// Mutates the slice to the previous lexicographic permutation.
    ///
    /// Returns `true` if successful and `false` if the slice is at the
    /// first-ordered permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: &mut [_] = &mut [1, 0, 2];
    /// v.prev_permutation();
    /// let b: &mut [_] = &mut [0, 2, 1];
    /// assert!(v == b);
    /// v.prev_permutation();
    /// let b: &mut [_] = &mut [0, 1, 2];
    /// assert!(v == b);
    /// ```
    #[unstable(feature = "collections",
               reason = "uncertain if this merits inclusion in std")]
    fn prev_permutation(&mut self) -> bool where Self::Item: Ord;

    /// Find the first index containing a matching value.
    #[unstable(feature = "collections")]
    fn position_elem(&self, t: &Self::Item) -> Option<usize> where Self::Item: PartialEq;

    /// Find the last index containing a matching value.
    #[unstable(feature = "collections")]
    fn rposition_elem(&self, t: &Self::Item) -> Option<usize> where Self::Item: PartialEq;

    /// Returns true if the slice contains an element with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.contains(&30));
    /// assert!(!v.contains(&50));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn contains(&self, x: &Self::Item) -> bool where Self::Item: PartialEq;

    /// Returns true if `needle` is a prefix of the slice.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    fn starts_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    /// Returns true if `needle` is a suffix of the slice.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ends_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    /// Convert `self` into a vector without clones or allocation.
    #[unstable(feature = "collections")]
    fn into_vec(self: Box<Self>) -> Vec<Self::Item>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> SliceExt for [T] {
    type Item = T;

    #[inline]
    fn sort_by<F>(&mut self, compare: F) where F: FnMut(&T, &T) -> Ordering {
        merge_sort(self, compare)
    }

    #[inline]
    fn move_from(&mut self, mut src: Vec<T>, start: usize, end: usize) -> usize {
        for (a, b) in self.iter_mut().zip(src[start .. end].iter_mut()) {
            mem::swap(a, b);
        }
        cmp::min(self.len(), end-start)
    }

    #[inline]
    fn slice(&self, start: usize, end: usize) -> &[T] {
        &self[start .. end]
    }

    #[inline]
    fn slice_from(&self, start: usize) -> &[T] {
        &self[start ..]
    }

    #[inline]
    fn slice_to(&self, end: usize) -> &[T] {
        &self[.. end]
    }

    #[inline]
    fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        core_slice::SliceExt::split_at(self, mid)
    }

    #[inline]
    fn iter(&self) -> Iter<T> {
        core_slice::SliceExt::iter(self)
    }

    #[inline]
    fn split<F>(&self, pred: F) -> Split<T, F>
                where F: FnMut(&T) -> bool {
        core_slice::SliceExt::split(self, pred)
    }

    #[inline]
    fn splitn<F>(&self, n: usize, pred: F) -> SplitN<T, F>
                 where F: FnMut(&T) -> bool {
        core_slice::SliceExt::splitn(self, n, pred)
    }

    #[inline]
    fn rsplitn<F>(&self, n: usize, pred: F) -> RSplitN<T, F>
                  where F: FnMut(&T) -> bool {
        core_slice::SliceExt::rsplitn(self, n, pred)
    }

    #[inline]
    fn windows(&self, size: usize) -> Windows<T> {
        core_slice::SliceExt::windows(self, size)
    }

    #[inline]
    fn chunks(&self, size: usize) -> Chunks<T> {
        core_slice::SliceExt::chunks(self, size)
    }

    #[inline]
    fn get(&self, index: usize) -> Option<&T> {
        core_slice::SliceExt::get(self, index)
    }

    #[inline]
    fn first(&self) -> Option<&T> {
        core_slice::SliceExt::first(self)
    }

    #[inline]
    fn tail(&self) -> &[T] {
        core_slice::SliceExt::tail(self)
    }

    #[inline]
    fn init(&self) -> &[T] {
        core_slice::SliceExt::init(self)
    }

    #[inline]
    fn last(&self) -> Option<&T> {
        core_slice::SliceExt::last(self)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        core_slice::SliceExt::get_unchecked(self, index)
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        core_slice::SliceExt::as_ptr(self)
    }

    #[inline]
    fn binary_search_by<F>(&self, f: F) -> Result<usize, usize>
                        where F: FnMut(&T) -> Ordering {
        core_slice::SliceExt::binary_search_by(self, f)
    }

    #[inline]
    fn len(&self) -> usize {
        core_slice::SliceExt::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        core_slice::SliceExt::is_empty(self)
    }

    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        core_slice::SliceExt::get_mut(self, index)
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        core_slice::SliceExt::as_mut_slice(self)
    }

    #[inline]
    fn slice_mut(&mut self, start: usize, end: usize) -> &mut [T] {
        &mut self[start .. end]
    }

    #[inline]
    fn slice_from_mut(&mut self, start: usize) -> &mut [T] {
        &mut self[start ..]
    }

    #[inline]
    fn slice_to_mut(&mut self, end: usize) -> &mut [T] {
        &mut self[.. end]
    }

    #[inline]
    fn iter_mut(&mut self) -> IterMut<T> {
        core_slice::SliceExt::iter_mut(self)
    }

    #[inline]
    fn first_mut(&mut self) -> Option<&mut T> {
        core_slice::SliceExt::first_mut(self)
    }

    #[inline]
    fn tail_mut(&mut self) -> &mut [T] {
        core_slice::SliceExt::tail_mut(self)
    }

    #[inline]
    fn init_mut(&mut self) -> &mut [T] {
        core_slice::SliceExt::init_mut(self)
    }

    #[inline]
    fn last_mut(&mut self) -> Option<&mut T> {
        core_slice::SliceExt::last_mut(self)
    }

    #[inline]
    fn split_mut<F>(&mut self, pred: F) -> SplitMut<T, F>
                    where F: FnMut(&T) -> bool {
        core_slice::SliceExt::split_mut(self, pred)
    }

    #[inline]
    fn splitn_mut<F>(&mut self, n: usize, pred: F) -> SplitNMut<T, F>
                     where F: FnMut(&T) -> bool {
        core_slice::SliceExt::splitn_mut(self, n, pred)
    }

    #[inline]
    fn rsplitn_mut<F>(&mut self,  n: usize, pred: F) -> RSplitNMut<T, F>
                      where F: FnMut(&T) -> bool {
        core_slice::SliceExt::rsplitn_mut(self, n, pred)
    }

    #[inline]
    fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        core_slice::SliceExt::chunks_mut(self, chunk_size)
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        core_slice::SliceExt::swap(self, a, b)
    }

    #[inline]
    fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        core_slice::SliceExt::split_at_mut(self, mid)
    }

    #[inline]
    fn reverse(&mut self) {
        core_slice::SliceExt::reverse(self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        core_slice::SliceExt::get_unchecked_mut(self, index)
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        core_slice::SliceExt::as_mut_ptr(self)
    }

    /// Returns a copy of `v`.
    #[inline]
    fn to_vec(&self) -> Vec<T> where T: Clone {
        let mut vector = Vec::with_capacity(self.len());
        vector.push_all(self);
        vector
    }

    /// Returns an iterator over all permutations of a vector.
    fn permutations(&self) -> Permutations<T> where T: Clone {
        Permutations{
            swaps: ElementSwaps::new(self.len()),
            v: self.to_vec(),
        }
    }

    fn clone_from_slice(&mut self, src: &[T]) -> usize where T: Clone {
        core_slice::SliceExt::clone_from_slice(self, src)
    }

    #[inline]
    fn sort(&mut self) where T: Ord {
        self.sort_by(|a, b| a.cmp(b))
    }

    fn binary_search(&self, x: &T) -> Result<usize, usize> where T: Ord {
        core_slice::SliceExt::binary_search(self, x)
    }

    fn next_permutation(&mut self) -> bool where T: Ord {
        core_slice::SliceExt::next_permutation(self)
    }

    fn prev_permutation(&mut self) -> bool where T: Ord {
        core_slice::SliceExt::prev_permutation(self)
    }

    fn position_elem(&self, t: &T) -> Option<usize> where T: PartialEq {
        core_slice::SliceExt::position_elem(self, t)
    }

    fn rposition_elem(&self, t: &T) -> Option<usize> where T: PartialEq {
        core_slice::SliceExt::rposition_elem(self, t)
    }

    fn contains(&self, x: &T) -> bool where T: PartialEq {
        core_slice::SliceExt::contains(self, x)
    }

    fn starts_with(&self, needle: &[T]) -> bool where T: PartialEq {
        core_slice::SliceExt::starts_with(self, needle)
    }

    fn ends_with(&self, needle: &[T]) -> bool where T: PartialEq {
        core_slice::SliceExt::ends_with(self, needle)
    }

    fn into_vec(mut self: Box<Self>) -> Vec<T> {
        unsafe {
            let xs = Vec::from_raw_parts(self.as_mut_ptr(), self.len(), self.len());
            mem::forget(self);
            xs
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Extension traits for slices over specific kinds of data
////////////////////////////////////////////////////////////////////////////////
#[unstable(feature = "collections", reason = "U should be an associated type")]
/// An extension trait for concatenating slices
pub trait SliceConcatExt<T: ?Sized, U> {
    /// Flattens a slice of `T` into a single value `U`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec!["hello", "world"];
    ///
    /// let s: String = v.concat();
    ///
    /// println!("{}", s); // prints "helloworld"
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn concat(&self) -> U;

    /// Flattens a slice of `T` into a single value `U`, placing a given separator between each.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec!["hello", "world"];
    ///
    /// let s: String = v.connect(" ");
    ///
    /// println!("{}", s); // prints "hello world"
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn connect(&self, sep: &T) -> U;
}

impl<T: Clone, V: AsSlice<T>> SliceConcatExt<T, Vec<T>> for [V] {
    fn concat(&self) -> Vec<T> {
        let size = self.iter().fold(0, |acc, v| acc + v.as_slice().len());
        let mut result = Vec::with_capacity(size);
        for v in self {
            result.push_all(v.as_slice())
        }
        result
    }

    fn connect(&self, sep: &T) -> Vec<T> {
        let size = self.iter().fold(0, |acc, v| acc + v.as_slice().len());
        let mut result = Vec::with_capacity(size + self.len());
        let mut first = true;
        for v in self {
            if first { first = false } else { result.push(sep.clone()) }
            result.push_all(v.as_slice())
        }
        result
    }
}

/// An iterator that yields the element swaps needed to produce
/// a sequence of all possible permutations for an indexed sequence of
/// elements. Each permutation is only a single swap apart.
///
/// The Steinhaus-Johnson-Trotter algorithm is used.
///
/// Generates even and odd permutations alternately.
///
/// The last generated swap is always (0, 1), and it returns the
/// sequence to its initial order.
#[unstable(feature = "collections")]
#[derive(Clone)]
pub struct ElementSwaps {
    sdir: Vec<SizeDirection>,
    /// If `true`, emit the last swap that returns the sequence to initial
    /// state.
    emit_reset: bool,
    swaps_made : usize,
}

impl ElementSwaps {
    /// Creates an `ElementSwaps` iterator for a sequence of `length` elements.
    #[unstable(feature = "collections")]
    pub fn new(length: usize) -> ElementSwaps {
        // Initialize `sdir` with a direction that position should move in
        // (all negative at the beginning) and the `size` of the
        // element (equal to the original index).
        ElementSwaps{
            emit_reset: true,
            sdir: (0..length).map(|i| SizeDirection{ size: i, dir: Neg }).collect(),
            swaps_made: 0
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Standard trait implementations for slices
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Borrow<[T]> for Vec<T> {
    fn borrow(&self) -> &[T] { &self[..] }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> BorrowMut<[T]> for Vec<T> {
    fn borrow_mut(&mut self) -> &mut [T] { &mut self[..] }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> ToOwned for [T] {
    type Owned = Vec<T>;
    fn to_owned(&self) -> Vec<T> { self.to_vec() }
}

////////////////////////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
enum Direction { Pos, Neg }

/// An `Index` and `Direction` together.
#[derive(Copy, Clone)]
struct SizeDirection {
    size: usize,
    dir: Direction,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for ElementSwaps {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        fn new_pos(i: usize, s: Direction) -> usize {
            i + match s { Pos => 1, Neg => -1 }
        }

        // Find the index of the largest mobile element:
        // The direction should point into the vector, and the
        // swap should be with a smaller `size` element.
        let max = self.sdir.iter().cloned().enumerate()
                           .filter(|&(i, sd)|
                                new_pos(i, sd.dir) < self.sdir.len() &&
                                self.sdir[new_pos(i, sd.dir)].size < sd.size)
                           .max_by(|&(_, sd)| sd.size);
        match max {
            Some((i, sd)) => {
                let j = new_pos(i, sd.dir);
                self.sdir.swap(i, j);

                // Swap the direction of each larger SizeDirection
                for x in &mut self.sdir {
                    if x.size > sd.size {
                        x.dir = match x.dir { Pos => Neg, Neg => Pos };
                    }
                }
                self.swaps_made += 1;
                Some((i, j))
            },
            None => if self.emit_reset {
                self.emit_reset = false;
                if self.sdir.len() > 1 {
                    // The last swap
                    self.swaps_made += 1;
                    Some((0, 1))
                } else {
                    // Vector is of the form [] or [x], and the only permutation is itself
                    self.swaps_made += 1;
                    Some((0,0))
                }
            } else { None }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // For a vector of size n, there are exactly n! permutations.
        let n = (2..self.sdir.len() + 1).product();
        (n - self.swaps_made, Some(n - self.swaps_made))
    }
}

/// An iterator that uses `ElementSwaps` to iterate through
/// all possible permutations of a vector.
///
/// The first iteration yields a clone of the vector as it is,
/// then each successive element is the vector with one
/// swap applied.
///
/// Generates even and odd permutations alternately.
#[unstable(feature = "collections")]
pub struct Permutations<T> {
    swaps: ElementSwaps,
    v: Vec<T>,
}

#[unstable(feature = "collections", reason = "trait is unstable")]
impl<T: Clone> Iterator for Permutations<T> {
    type Item = Vec<T>;

    #[inline]
    fn next(&mut self) -> Option<Vec<T>> {
        match self.swaps.next() {
            None => None,
            Some((0,0)) => Some(self.v.clone()),
            Some((a, b)) => {
                let elt = self.v.clone();
                self.v.swap(a, b);
                Some(elt)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.swaps.size_hint()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sorting
////////////////////////////////////////////////////////////////////////////////

fn insertion_sort<T, F>(v: &mut [T], mut compare: F) where F: FnMut(&T, &T) -> Ordering {
    let len = v.len() as isize;
    let buf_v = v.as_mut_ptr();

    // 1 <= i < len;
    for i in 1..len {
        // j satisfies: 0 <= j <= i;
        let mut j = i;
        unsafe {
            // `i` is in bounds.
            let read_ptr = buf_v.offset(i) as *const T;

            // find where to insert, we need to do strict <,
            // rather than <=, to maintain stability.

            // 0 <= j - 1 < len, so .offset(j - 1) is in bounds.
            while j > 0 &&
                    compare(&*read_ptr, &*buf_v.offset(j - 1)) == Less {
                j -= 1;
            }

            // shift everything to the right, to make space to
            // insert this value.

            // j + 1 could be `len` (for the last `i`), but in
            // that case, `i == j` so we don't copy. The
            // `.offset(j)` is always in bounds.

            if i != j {
                let tmp = ptr::read(read_ptr);
                ptr::copy_memory(buf_v.offset(j + 1),
                                 &*buf_v.offset(j),
                                 (i - j) as usize);
                ptr::copy_nonoverlapping_memory(buf_v.offset(j),
                                                &tmp,
                                                1);
                mem::forget(tmp);
            }
        }
    }
}

fn merge_sort<T, F>(v: &mut [T], mut compare: F) where F: FnMut(&T, &T) -> Ordering {
    // warning: this wildly uses unsafe.
    static BASE_INSERTION: usize = 32;
    static LARGE_INSERTION: usize = 16;

    // FIXME #12092: smaller insertion runs seems to make sorting
    // vectors of large elements a little faster on some platforms,
    // but hasn't been tested/tuned extensively
    let insertion = if size_of::<T>() <= 16 {
        BASE_INSERTION
    } else {
        LARGE_INSERTION
    };

    let len = v.len();

    // short vectors get sorted in-place via insertion sort to avoid allocations
    if len <= insertion {
        insertion_sort(v, compare);
        return;
    }

    // allocate some memory to use as scratch memory, we keep the
    // length 0 so we can keep shallow copies of the contents of `v`
    // without risking the dtors running on an object twice if
    // `compare` panics.
    let mut working_space = Vec::with_capacity(2 * len);
    // these both are buffers of length `len`.
    let mut buf_dat = working_space.as_mut_ptr();
    let mut buf_tmp = unsafe {buf_dat.offset(len as isize)};

    // length `len`.
    let buf_v = v.as_ptr();

    // step 1. sort short runs with insertion sort. This takes the
    // values from `v` and sorts them into `buf_dat`, leaving that
    // with sorted runs of length INSERTION.

    // We could hardcode the sorting comparisons here, and we could
    // manipulate/step the pointers themselves, rather than repeatedly
    // .offset-ing.
    for start in range_step(0, len, insertion) {
        // start <= i < len;
        for i in start..cmp::min(start + insertion, len) {
            // j satisfies: start <= j <= i;
            let mut j = i as isize;
            unsafe {
                // `i` is in bounds.
                let read_ptr = buf_v.offset(i as isize);

                // find where to insert, we need to do strict <,
                // rather than <=, to maintain stability.

                // start <= j - 1 < len, so .offset(j - 1) is in
                // bounds.
                while j > start as isize &&
                        compare(&*read_ptr, &*buf_dat.offset(j - 1)) == Less {
                    j -= 1;
                }

                // shift everything to the right, to make space to
                // insert this value.

                // j + 1 could be `len` (for the last `i`), but in
                // that case, `i == j` so we don't copy. The
                // `.offset(j)` is always in bounds.
                ptr::copy_memory(buf_dat.offset(j + 1),
                                 &*buf_dat.offset(j),
                                 i - j as usize);
                ptr::copy_nonoverlapping_memory(buf_dat.offset(j), read_ptr, 1);
            }
        }
    }

    // step 2. merge the sorted runs.
    let mut width = insertion;
    while width < len {
        // merge the sorted runs of length `width` in `buf_dat` two at
        // a time, placing the result in `buf_tmp`.

        // 0 <= start <= len.
        for start in range_step(0, len, 2 * width) {
            // manipulate pointers directly for speed (rather than
            // using a `for` loop with `range` and `.offset` inside
            // that loop).
            unsafe {
                // the end of the first run & start of the
                // second. Offset of `len` is defined, since this is
                // precisely one byte past the end of the object.
                let right_start = buf_dat.offset(cmp::min(start + width, len) as isize);
                // end of the second. Similar reasoning to the above re safety.
                let right_end_idx = cmp::min(start + 2 * width, len);
                let right_end = buf_dat.offset(right_end_idx as isize);

                // the pointers to the elements under consideration
                // from the two runs.

                // both of these are in bounds.
                let mut left = buf_dat.offset(start as isize);
                let mut right = right_start;

                // where we're putting the results, it is a run of
                // length `2*width`, so we step it once for each step
                // of either `left` or `right`.  `buf_tmp` has length
                // `len`, so these are in bounds.
                let mut out = buf_tmp.offset(start as isize);
                let out_end = buf_tmp.offset(right_end_idx as isize);

                while out < out_end {
                    // Either the left or the right run are exhausted,
                    // so just copy the remainder from the other run
                    // and move on; this gives a huge speed-up (order
                    // of 25%) for mostly sorted vectors (the best
                    // case).
                    if left == right_start {
                        // the number remaining in this run.
                        let elems = (right_end as usize - right as usize) / mem::size_of::<T>();
                        ptr::copy_nonoverlapping_memory(out, &*right, elems);
                        break;
                    } else if right == right_end {
                        let elems = (right_start as usize - left as usize) / mem::size_of::<T>();
                        ptr::copy_nonoverlapping_memory(out, &*left, elems);
                        break;
                    }

                    // check which side is smaller, and that's the
                    // next element for the new run.

                    // `left < right_start` and `right < right_end`,
                    // so these are valid.
                    let to_copy = if compare(&*left, &*right) == Greater {
                        step(&mut right)
                    } else {
                        step(&mut left)
                    };
                    ptr::copy_nonoverlapping_memory(out, &*to_copy, 1);
                    step(&mut out);
                }
            }
        }

        mem::swap(&mut buf_dat, &mut buf_tmp);

        width *= 2;
    }

    // write the result to `v` in one go, so that there are never two copies
    // of the same object in `v`.
    unsafe {
        ptr::copy_nonoverlapping_memory(v.as_mut_ptr(), &*buf_dat, len);
    }

    // increment the pointer, returning the old pointer.
    #[inline(always)]
    unsafe fn step<T>(ptr: &mut *mut T) -> *mut T {
        let old = *ptr;
        *ptr = ptr.offset(1);
        old
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering::{Greater, Less, Equal};
    use core::prelude::{Some, None, Clone};
    use core::prelude::{Iterator, IteratorExt};
    use core::prelude::{AsSlice};
    use core::prelude::Ord;
    use core::default::Default;
    use core::mem;
    use std::iter::RandomAccessIterator;
    use std::rand::{Rng, thread_rng};
    use std::rc::Rc;
    use string::ToString;
    use vec::Vec;
    use super::{ElementSwaps, SliceConcatExt, SliceExt};

    fn square(n: usize) -> usize { n * n }

    fn is_odd(n: &usize) -> bool { *n % 2 == 1 }

    #[test]
    fn test_from_fn() {
        // Test on-stack from_fn.
        let mut v: Vec<_> = (0..3).map(square).collect();
        {
            let v = v;
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], 0);
            assert_eq!(v[1], 1);
            assert_eq!(v[2], 4);
        }

        // Test on-heap from_fn.
        v = (0..5).map(square).collect();
        {
            let v = v;
            assert_eq!(v.len(), 5);
            assert_eq!(v[0], 0);
            assert_eq!(v[1], 1);
            assert_eq!(v[2], 4);
            assert_eq!(v[3], 9);
            assert_eq!(v[4], 16);
        }
    }

    #[test]
    fn test_from_elem() {
        // Test on-stack from_elem.
        let mut v = vec![10, 10];
        {
            let v = v;
            assert_eq!(v.len(), 2);
            assert_eq!(v[0], 10);
            assert_eq!(v[1], 10);
        }

        // Test on-heap from_elem.
        v = vec![20; 6];
        {
            let v = v.as_slice();
            assert_eq!(v[0], 20);
            assert_eq!(v[1], 20);
            assert_eq!(v[2], 20);
            assert_eq!(v[3], 20);
            assert_eq!(v[4], 20);
            assert_eq!(v[5], 20);
        }
    }

    #[test]
    fn test_is_empty() {
        let xs: [i32; 0] = [];
        assert!(xs.is_empty());
        assert!(![0].is_empty());
    }

    #[test]
    fn test_len_divzero() {
        type Z = [i8; 0];
        let v0 : &[Z] = &[];
        let v1 : &[Z] = &[[]];
        let v2 : &[Z] = &[[], []];
        assert_eq!(mem::size_of::<Z>(), 0);
        assert_eq!(v0.len(), 0);
        assert_eq!(v1.len(), 1);
        assert_eq!(v2.len(), 2);
    }

    #[test]
    fn test_get() {
        let mut a = vec![11];
        assert_eq!(a.get(1), None);
        a = vec![11, 12];
        assert_eq!(a.get(1).unwrap(), &12);
        a = vec![11, 12, 13];
        assert_eq!(a.get(1).unwrap(), &12);
    }

    #[test]
    fn test_first() {
        let mut a = vec![];
        assert_eq!(a.first(), None);
        a = vec![11];
        assert_eq!(a.first().unwrap(), &11);
        a = vec![11, 12];
        assert_eq!(a.first().unwrap(), &11);
    }

    #[test]
    fn test_first_mut() {
        let mut a = vec![];
        assert_eq!(a.first_mut(), None);
        a = vec![11];
        assert_eq!(*a.first_mut().unwrap(), 11);
        a = vec![11, 12];
        assert_eq!(*a.first_mut().unwrap(), 11);
    }

    #[test]
    fn test_tail() {
        let mut a = vec![11];
        let b: &[i32] = &[];
        assert_eq!(a.tail(), b);
        a = vec![11, 12];
        let b: &[i32] = &[12];
        assert_eq!(a.tail(), b);
    }

    #[test]
    fn test_tail_mut() {
        let mut a = vec![11];
        let b: &mut [i32] = &mut [];
        assert!(a.tail_mut() == b);
        a = vec![11, 12];
        let b: &mut [_] = &mut [12];
        assert!(a.tail_mut() == b);
    }

    #[test]
    #[should_fail]
    fn test_tail_empty() {
        let a = Vec::<i32>::new();
        a.tail();
    }

    #[test]
    #[should_fail]
    fn test_tail_mut_empty() {
        let mut a = Vec::<i32>::new();
        a.tail_mut();
    }

    #[test]
    fn test_init() {
        let mut a = vec![11];
        let b: &[i32] = &[];
        assert_eq!(a.init(), b);
        a = vec![11, 12];
        let b: &[_] = &[11];
        assert_eq!(a.init(), b);
    }

    #[test]
    fn test_init_mut() {
        let mut a = vec![11];
        let b: &mut [i32] = &mut [];
        assert!(a.init_mut() == b);
        a = vec![11, 12];
        let b: &mut [_] = &mut [11];
        assert!(a.init_mut() == b);
    }

    #[test]
    #[should_fail]
    fn test_init_empty() {
        let a = Vec::<i32>::new();
        a.init();
    }

    #[test]
    #[should_fail]
    fn test_init_mut_empty() {
        let mut a = Vec::<i32>::new();
        a.init_mut();
    }

    #[test]
    fn test_last() {
        let mut a = vec![];
        assert_eq!(a.last(), None);
        a = vec![11];
        assert_eq!(a.last().unwrap(), &11);
        a = vec![11, 12];
        assert_eq!(a.last().unwrap(), &12);
    }

    #[test]
    fn test_last_mut() {
        let mut a = vec![];
        assert_eq!(a.last_mut(), None);
        a = vec![11];
        assert_eq!(*a.last_mut().unwrap(), 11);
        a = vec![11, 12];
        assert_eq!(*a.last_mut().unwrap(), 12);
    }

    #[test]
    fn test_slice() {
        // Test fixed length vector.
        let vec_fixed = [1, 2, 3, 4];
        let v_a = vec_fixed[1..vec_fixed.len()].to_vec();
        assert_eq!(v_a.len(), 3);

        assert_eq!(v_a[0], 2);
        assert_eq!(v_a[1], 3);
        assert_eq!(v_a[2], 4);

        // Test on stack.
        let vec_stack: &[_] = &[1, 2, 3];
        let v_b = vec_stack[1..3].to_vec();
        assert_eq!(v_b.len(), 2);

        assert_eq!(v_b[0], 2);
        assert_eq!(v_b[1], 3);

        // Test `Box<[T]>`
        let vec_unique = vec![1, 2, 3, 4, 5, 6];
        let v_d = vec_unique[1..6].to_vec();
        assert_eq!(v_d.len(), 5);

        assert_eq!(v_d[0], 2);
        assert_eq!(v_d[1], 3);
        assert_eq!(v_d[2], 4);
        assert_eq!(v_d[3], 5);
        assert_eq!(v_d[4], 6);
    }

    #[test]
    fn test_slice_from() {
        let vec: &[_] = &[1, 2, 3, 4];
        assert_eq!(&vec[..], vec);
        let b: &[_] = &[3, 4];
        assert_eq!(&vec[2..], b);
        let b: &[_] = &[];
        assert_eq!(&vec[4..], b);
    }

    #[test]
    fn test_slice_to() {
        let vec: &[_] = &[1, 2, 3, 4];
        assert_eq!(&vec[..4], vec);
        let b: &[_] = &[1, 2];
        assert_eq!(&vec[..2], b);
        let b: &[_] = &[];
        assert_eq!(&vec[..0], b);
    }


    #[test]
    fn test_pop() {
        let mut v = vec![5];
        let e = v.pop();
        assert_eq!(v.len(), 0);
        assert_eq!(e, Some(5));
        let f = v.pop();
        assert_eq!(f, None);
        let g = v.pop();
        assert_eq!(g, None);
    }

    #[test]
    fn test_swap_remove() {
        let mut v = vec![1, 2, 3, 4, 5];
        let mut e = v.swap_remove(0);
        assert_eq!(e, 1);
        assert_eq!(v, vec![5, 2, 3, 4]);
        e = v.swap_remove(3);
        assert_eq!(e, 4);
        assert_eq!(v, vec![5, 2, 3]);
    }

    #[test]
    #[should_fail]
    fn test_swap_remove_fail() {
        let mut v = vec![1];
        let _ = v.swap_remove(0);
        let _ = v.swap_remove(0);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = Vec::new();
        v.push(box 0u8);
        v.push(box 0u8);
        v.push(box 0u8);
        let mut _e = v.swap_remove(0);
        assert_eq!(v.len(), 2);
        _e = v.swap_remove(1);
        assert_eq!(v.len(), 1);
        _e = v.swap_remove(0);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_push() {
        // Test on-stack push().
        let mut v = vec![];
        v.push(1);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 1);

        // Test on-heap push().
        v.push(2);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
    }

    #[test]
    fn test_truncate() {
        let mut v = vec![box 6,box 5,box 4];
        v.truncate(1);
        let v = v;
        assert_eq!(v.len(), 1);
        assert_eq!(*(v[0]), 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_clear() {
        let mut v = vec![box 6,box 5,box 4];
        v.clear();
        assert_eq!(v.len(), 0);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_dedup() {
        fn case(a: Vec<i32>, b: Vec<i32>) {
            let mut v = a;
            v.dedup();
            assert_eq!(v, b);
        }
        case(vec![], vec![]);
        case(vec![1], vec![1]);
        case(vec![1,1], vec![1]);
        case(vec![1,2,3], vec![1,2,3]);
        case(vec![1,1,2,3], vec![1,2,3]);
        case(vec![1,2,2,3], vec![1,2,3]);
        case(vec![1,2,3,3], vec![1,2,3]);
        case(vec![1,1,2,2,2,3,3], vec![1,2,3]);
    }

    #[test]
    fn test_dedup_unique() {
        let mut v0 = vec![box 1, box 1, box 2, box 3];
        v0.dedup();
        let mut v1 = vec![box 1, box 2, box 2, box 3];
        v1.dedup();
        let mut v2 = vec![box 1, box 2, box 3, box 3];
        v2.dedup();
        /*
         * If the boxed pointers were leaked or otherwise misused, valgrind
         * and/or rt should raise errors.
         */
    }

    #[test]
    fn test_dedup_shared() {
        let mut v0 = vec![box 1, box 1, box 2, box 3];
        v0.dedup();
        let mut v1 = vec![box 1, box 2, box 2, box 3];
        v1.dedup();
        let mut v2 = vec![box 1, box 2, box 3, box 3];
        v2.dedup();
        /*
         * If the pointers were leaked or otherwise misused, valgrind and/or
         * rt should raise errors.
         */
    }

    #[test]
    fn test_retain() {
        let mut v = vec![1, 2, 3, 4, 5];
        v.retain(is_odd);
        assert_eq!(v, vec![1, 3, 5]);
    }

    #[test]
    fn test_element_swaps() {
        let mut v = [1, 2, 3];
        for (i, (a, b)) in ElementSwaps::new(v.len()).enumerate() {
            v.swap(a, b);
            match i {
                0 => assert!(v == [1, 3, 2]),
                1 => assert!(v == [3, 1, 2]),
                2 => assert!(v == [3, 2, 1]),
                3 => assert!(v == [2, 3, 1]),
                4 => assert!(v == [2, 1, 3]),
                5 => assert!(v == [1, 2, 3]),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn test_permutations() {
        {
            let v: [i32; 0] = [];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 1);
            assert_eq!(max_opt.unwrap(), 1);
            assert_eq!(it.next(), Some(v.to_vec()));
            assert_eq!(it.next(), None);
        }
        {
            let v = ["Hello".to_string()];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 1);
            assert_eq!(max_opt.unwrap(), 1);
            assert_eq!(it.next(), Some(v.to_vec()));
            assert_eq!(it.next(), None);
        }
        {
            let v = [1, 2, 3];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 3*2);
            assert_eq!(max_opt.unwrap(), 3*2);
            assert_eq!(it.next(), Some(vec![1,2,3]));
            assert_eq!(it.next(), Some(vec![1,3,2]));
            assert_eq!(it.next(), Some(vec![3,1,2]));
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 3);
            assert_eq!(max_opt.unwrap(), 3);
            assert_eq!(it.next(), Some(vec![3,2,1]));
            assert_eq!(it.next(), Some(vec![2,3,1]));
            assert_eq!(it.next(), Some(vec![2,1,3]));
            assert_eq!(it.next(), None);
        }
        {
            // check that we have N! permutations
            let v = ['A', 'B', 'C', 'D', 'E', 'F'];
            let mut amt = 0;
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            for _perm in it.by_ref() {
                amt += 1;
            }
            assert_eq!(amt, it.swaps.swaps_made);
            assert_eq!(amt, min_size);
            assert_eq!(amt, 2 * 3 * 4 * 5 * 6);
            assert_eq!(amt, max_opt.unwrap());
        }
    }

    #[test]
    fn test_lexicographic_permutations() {
        let v : &mut[_] = &mut[1, 2, 3, 4, 5];
        assert!(v.prev_permutation() == false);
        assert!(v.next_permutation());
        let b: &mut[_] = &mut[1, 2, 3, 5, 4];
        assert!(v == b);
        assert!(v.prev_permutation());
        let b: &mut[_] = &mut[1, 2, 3, 4, 5];
        assert!(v == b);
        assert!(v.next_permutation());
        assert!(v.next_permutation());
        let b: &mut[_] = &mut[1, 2, 4, 3, 5];
        assert!(v == b);
        assert!(v.next_permutation());
        let b: &mut[_] = &mut[1, 2, 4, 5, 3];
        assert!(v == b);

        let v : &mut[_] = &mut[1, 0, 0, 0];
        assert!(v.next_permutation() == false);
        assert!(v.prev_permutation());
        let b: &mut[_] = &mut[0, 1, 0, 0];
        assert!(v == b);
        assert!(v.prev_permutation());
        let b: &mut[_] = &mut[0, 0, 1, 0];
        assert!(v == b);
        assert!(v.prev_permutation());
        let b: &mut[_] = &mut[0, 0, 0, 1];
        assert!(v == b);
        assert!(v.prev_permutation() == false);
    }

    #[test]
    fn test_lexicographic_permutations_empty_and_short() {
        let empty : &mut[i32] = &mut[];
        assert!(empty.next_permutation() == false);
        let b: &mut[i32] = &mut[];
        assert!(empty == b);
        assert!(empty.prev_permutation() == false);
        assert!(empty == b);

        let one_elem : &mut[_] = &mut[4];
        assert!(one_elem.prev_permutation() == false);
        let b: &mut[_] = &mut[4];
        assert!(one_elem == b);
        assert!(one_elem.next_permutation() == false);
        assert!(one_elem == b);

        let two_elem : &mut[_] = &mut[1, 2];
        assert!(two_elem.prev_permutation() == false);
        let b : &mut[_] = &mut[1, 2];
        let c : &mut[_] = &mut[2, 1];
        assert!(two_elem == b);
        assert!(two_elem.next_permutation());
        assert!(two_elem == c);
        assert!(two_elem.next_permutation() == false);
        assert!(two_elem == c);
        assert!(two_elem.prev_permutation());
        assert!(two_elem == b);
        assert!(two_elem.prev_permutation() == false);
        assert!(two_elem == b);
    }

    #[test]
    fn test_position_elem() {
        assert!([].position_elem(&1).is_none());

        let v1 = vec![1, 2, 3, 3, 2, 5];
        assert_eq!(v1.position_elem(&1), Some(0));
        assert_eq!(v1.position_elem(&2), Some(1));
        assert_eq!(v1.position_elem(&5), Some(5));
        assert!(v1.position_elem(&4).is_none());
    }

    #[test]
    fn test_binary_search() {
        assert_eq!([1,2,3,4,5].binary_search(&5).ok(), Some(4));
        assert_eq!([1,2,3,4,5].binary_search(&4).ok(), Some(3));
        assert_eq!([1,2,3,4,5].binary_search(&3).ok(), Some(2));
        assert_eq!([1,2,3,4,5].binary_search(&2).ok(), Some(1));
        assert_eq!([1,2,3,4,5].binary_search(&1).ok(), Some(0));

        assert_eq!([2,4,6,8,10].binary_search(&1).ok(), None);
        assert_eq!([2,4,6,8,10].binary_search(&5).ok(), None);
        assert_eq!([2,4,6,8,10].binary_search(&4).ok(), Some(1));
        assert_eq!([2,4,6,8,10].binary_search(&10).ok(), Some(4));

        assert_eq!([2,4,6,8].binary_search(&1).ok(), None);
        assert_eq!([2,4,6,8].binary_search(&5).ok(), None);
        assert_eq!([2,4,6,8].binary_search(&4).ok(), Some(1));
        assert_eq!([2,4,6,8].binary_search(&8).ok(), Some(3));

        assert_eq!([2,4,6].binary_search(&1).ok(), None);
        assert_eq!([2,4,6].binary_search(&5).ok(), None);
        assert_eq!([2,4,6].binary_search(&4).ok(), Some(1));
        assert_eq!([2,4,6].binary_search(&6).ok(), Some(2));

        assert_eq!([2,4].binary_search(&1).ok(), None);
        assert_eq!([2,4].binary_search(&5).ok(), None);
        assert_eq!([2,4].binary_search(&2).ok(), Some(0));
        assert_eq!([2,4].binary_search(&4).ok(), Some(1));

        assert_eq!([2].binary_search(&1).ok(), None);
        assert_eq!([2].binary_search(&5).ok(), None);
        assert_eq!([2].binary_search(&2).ok(), Some(0));

        assert_eq!([].binary_search(&1).ok(), None);
        assert_eq!([].binary_search(&5).ok(), None);

        assert!([1,1,1,1,1].binary_search(&1).ok() != None);
        assert!([1,1,1,1,2].binary_search(&1).ok() != None);
        assert!([1,1,1,2,2].binary_search(&1).ok() != None);
        assert!([1,1,2,2,2].binary_search(&1).ok() != None);
        assert_eq!([1,2,2,2,2].binary_search(&1).ok(), Some(0));

        assert_eq!([1,2,3,4,5].binary_search(&6).ok(), None);
        assert_eq!([1,2,3,4,5].binary_search(&0).ok(), None);
    }

    #[test]
    fn test_reverse() {
        let mut v = vec![10, 20];
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        v.reverse();
        assert_eq!(v[0], 20);
        assert_eq!(v[1], 10);

        let mut v3 = Vec::<i32>::new();
        v3.reverse();
        assert!(v3.is_empty());
    }

    #[test]
    fn test_sort() {
        for len in 4..25 {
            for _ in 0..100 {
                let mut v: Vec<_> = thread_rng().gen_iter::<i32>().take(len).collect();
                let mut v1 = v.clone();

                v.sort();
                assert!(v.windows(2).all(|w| w[0] <= w[1]));

                v1.sort_by(|a, b| a.cmp(b));
                assert!(v1.windows(2).all(|w| w[0] <= w[1]));

                v1.sort_by(|a, b| b.cmp(a));
                assert!(v1.windows(2).all(|w| w[0] >= w[1]));
            }
        }

        // shouldn't panic
        let mut v: [i32; 0] = [];
        v.sort();

        let mut v = [0xDEADBEEFu64];
        v.sort();
        assert!(v == [0xDEADBEEF]);
    }

    #[test]
    fn test_sort_stability() {
        for len in 4..25 {
            for _ in 0..10 {
                let mut counts = [0; 10];

                // create a vector like [(6, 1), (5, 1), (6, 2), ...],
                // where the first item of each tuple is random, but
                // the second item represents which occurrence of that
                // number this element is, i.e. the second elements
                // will occur in sorted order.
                let mut v: Vec<_> = (0..len).map(|_| {
                        let n = thread_rng().gen::<usize>() % 10;
                        counts[n] += 1;
                        (n, counts[n])
                    }).collect();

                // only sort on the first element, so an unstable sort
                // may mix up the counts.
                v.sort_by(|&(a,_), &(b,_)| a.cmp(&b));

                // this comparison includes the count (the second item
                // of the tuple), so elements with equal first items
                // will need to be ordered with increasing
                // counts... i.e. exactly asserting that this sort is
                // stable.
                assert!(v.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }

    #[test]
    fn test_concat() {
        let v: [Vec<i32>; 0] = [];
        let c = v.concat();
        assert_eq!(c, []);
        let d = [vec![1], vec![2,3]].concat();
        assert_eq!(d, vec![1, 2, 3]);

        let v: &[&[_]] = &[&[1], &[2, 3]];
        assert_eq!(v.connect(&0), vec![1, 0, 2, 3]);
        let v: &[&[_]] = &[&[1], &[2], &[3]];
        assert_eq!(v.connect(&0), vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_connect() {
        let v: [Vec<i32>; 0] = [];
        assert_eq!(v.connect(&0), vec![]);
        assert_eq!([vec![1], vec![2, 3]].connect(&0), vec![1, 0, 2, 3]);
        assert_eq!([vec![1], vec![2], vec![3]].connect(&0), vec![1, 0, 2, 0, 3]);

        let v: [&[_]; 2] = [&[1], &[2, 3]];
        assert_eq!(v.connect(&0), vec![1, 0, 2, 3]);
        let v: [&[_]; 3] = [&[1], &[2], &[3]];
        assert_eq!(v.connect(&0), vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_insert() {
        let mut a = vec![1, 2, 4];
        a.insert(2, 3);
        assert_eq!(a, vec![1, 2, 3, 4]);

        let mut a = vec![1, 2, 3];
        a.insert(0, 0);
        assert_eq!(a, vec![0, 1, 2, 3]);

        let mut a = vec![1, 2, 3];
        a.insert(3, 4);
        assert_eq!(a, vec![1, 2, 3, 4]);

        let mut a = vec![];
        a.insert(0, 1);
        assert_eq!(a, vec![1]);
    }

    #[test]
    #[should_fail]
    fn test_insert_oob() {
        let mut a = vec![1, 2, 3];
        a.insert(4, 5);
    }

    #[test]
    fn test_remove() {
        let mut a = vec![1, 2, 3, 4];

        assert_eq!(a.remove(2), 3);
        assert_eq!(a, vec![1, 2, 4]);

        assert_eq!(a.remove(2), 4);
        assert_eq!(a, vec![1, 2]);

        assert_eq!(a.remove(0), 1);
        assert_eq!(a, vec![2]);

        assert_eq!(a.remove(0), 2);
        assert_eq!(a, vec![]);
    }

    #[test]
    #[should_fail]
    fn test_remove_fail() {
        let mut a = vec![1];
        let _ = a.remove(0);
        let _ = a.remove(0);
    }

    #[test]
    fn test_capacity() {
        let mut v = vec![0];
        v.reserve_exact(10);
        assert!(v.capacity() >= 11);
    }

    #[test]
    fn test_slice_2() {
        let v = vec![1, 2, 3, 4, 5];
        let v = v.slice(1, 3);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 3);
    }

    #[test]
    #[should_fail]
    fn test_permute_fail() {
        let v = [(box 0, Rc::new(0)), (box 0, Rc::new(0)),
                 (box 0, Rc::new(0)), (box 0, Rc::new(0))];
        let mut i = 0;
        for _ in v.permutations() {
            if i == 2 {
                panic!()
            }
            i += 1;
        }
    }

    #[test]
    fn test_total_ord() {
        let c = &[1, 2, 3];
        [1, 2, 3, 4][..].cmp(c) == Greater;
        let c = &[1, 2, 3, 4];
        [1, 2, 3][..].cmp(c) == Less;
        let c = &[1, 2, 3, 6];
        [1, 2, 3, 4][..].cmp(c) == Equal;
        let c = &[1, 2, 3, 4, 5, 6];
        [1, 2, 3, 4, 5, 5, 5, 5][..].cmp(c) == Less;
        let c = &[1, 2, 3, 4];
        [2, 2][..].cmp(c) == Greater;
    }

    #[test]
    fn test_iterator() {
        let xs = [1, 2, 5, 10, 11];
        let mut it = xs.iter();
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.next().unwrap(), &1);
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.next().unwrap(), &2);
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.next().unwrap(), &5);
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.next().unwrap(), &10);
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), &11);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_random_access_iterator() {
        let xs = [1, 2, 5, 10, 11];
        let mut it = xs.iter();

        assert_eq!(it.indexable(), 5);
        assert_eq!(it.idx(0).unwrap(), &1);
        assert_eq!(it.idx(2).unwrap(), &5);
        assert_eq!(it.idx(4).unwrap(), &11);
        assert!(it.idx(5).is_none());

        assert_eq!(it.next().unwrap(), &1);
        assert_eq!(it.indexable(), 4);
        assert_eq!(it.idx(0).unwrap(), &2);
        assert_eq!(it.idx(3).unwrap(), &11);
        assert!(it.idx(4).is_none());

        assert_eq!(it.next().unwrap(), &2);
        assert_eq!(it.indexable(), 3);
        assert_eq!(it.idx(1).unwrap(), &10);
        assert!(it.idx(3).is_none());

        assert_eq!(it.next().unwrap(), &5);
        assert_eq!(it.indexable(), 2);
        assert_eq!(it.idx(1).unwrap(), &11);

        assert_eq!(it.next().unwrap(), &10);
        assert_eq!(it.indexable(), 1);
        assert_eq!(it.idx(0).unwrap(), &11);
        assert!(it.idx(1).is_none());

        assert_eq!(it.next().unwrap(), &11);
        assert_eq!(it.indexable(), 0);
        assert!(it.idx(0).is_none());

        assert!(it.next().is_none());
    }

    #[test]
    fn test_iter_size_hints() {
        let mut xs = [1, 2, 5, 10, 11];
        assert_eq!(xs.iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.iter_mut().size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_iter_clone() {
        let xs = [1, 2, 5];
        let mut it = xs.iter();
        it.next();
        let mut jt = it.clone();
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next(), jt.next());
    }

    #[test]
    fn test_mut_iterator() {
        let mut xs = [1, 2, 3, 4, 5];
        for x in &mut xs {
            *x += 1;
        }
        assert!(xs == [2, 3, 4, 5, 6])
    }

    #[test]
    fn test_rev_iterator() {

        let xs = [1, 2, 5, 10, 11];
        let ys = [11, 10, 5, 2, 1];
        let mut i = 0;
        for &x in xs.iter().rev() {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, 5);
    }

    #[test]
    fn test_mut_rev_iterator() {
        let mut xs = [1, 2, 3, 4, 5];
        for (i,x) in xs.iter_mut().rev().enumerate() {
            *x += i;
        }
        assert!(xs == [5, 5, 5, 5, 5])
    }

    #[test]
    fn test_move_iterator() {
        let xs = vec![1,2,3,4,5];
        assert_eq!(xs.into_iter().fold(0, |a: usize, b: usize| 10*a + b), 12345);
    }

    #[test]
    fn test_move_rev_iterator() {
        let xs = vec![1,2,3,4,5];
        assert_eq!(xs.into_iter().rev().fold(0, |a: usize, b: usize| 10*a + b), 54321);
    }

    #[test]
    fn test_splitator() {
        let xs = &[1,2,3,4,5];

        let splits: &[&[_]] = &[&[1], &[3], &[5]];
        assert_eq!(xs.split(|x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[], &[2,3,4,5]];
        assert_eq!(xs.split(|x| *x == 1).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[1,2,3,4], &[]];
        assert_eq!(xs.split(|x| *x == 5).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.split(|x| *x == 10).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[], &[], &[], &[], &[], &[]];
        assert_eq!(xs.split(|_| true).collect::<Vec<&[i32]>>(),
                   splits);

        let xs: &[i32] = &[];
        let splits: &[&[i32]] = &[&[]];
        assert_eq!(xs.split(|x| *x == 5).collect::<Vec<&[i32]>>(), splits);
    }

    #[test]
    fn test_splitnator() {
        let xs = &[1,2,3,4,5];

        let splits: &[&[_]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.splitn(0, |x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[1], &[3,4,5]];
        assert_eq!(xs.splitn(1, |x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[], &[], &[], &[4,5]];
        assert_eq!(xs.splitn(3, |_| true).collect::<Vec<_>>(),
                   splits);

        let xs: &[i32] = &[];
        let splits: &[&[i32]] = &[&[]];
        assert_eq!(xs.splitn(1, |x| *x == 5).collect::<Vec<_>>(), splits);
    }

    #[test]
    fn test_splitnator_mut() {
        let xs = &mut [1,2,3,4,5];

        let splits: &[&mut[_]] = &[&mut [1,2,3,4,5]];
        assert_eq!(xs.splitn_mut(0, |x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&mut[_]] = &[&mut [1], &mut [3,4,5]];
        assert_eq!(xs.splitn_mut(1, |x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&mut[_]] = &[&mut [], &mut [], &mut [], &mut [4,5]];
        assert_eq!(xs.splitn_mut(3, |_| true).collect::<Vec<_>>(),
                   splits);

        let xs: &mut [i32] = &mut [];
        let splits: &[&mut[i32]] = &[&mut []];
        assert_eq!(xs.splitn_mut(1, |x| *x == 5).collect::<Vec<_>>(),
                   splits);
    }

    #[test]
    fn test_rsplitator() {
        let xs = &[1,2,3,4,5];

        let splits: &[&[_]] = &[&[5], &[3], &[1]];
        assert_eq!(xs.split(|x| *x % 2 == 0).rev().collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[2,3,4,5], &[]];
        assert_eq!(xs.split(|x| *x == 1).rev().collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[], &[1,2,3,4]];
        assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.split(|x| *x == 10).rev().collect::<Vec<_>>(),
                   splits);

        let xs: &[i32] = &[];
        let splits: &[&[i32]] = &[&[]];
        assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<&[i32]>>(), splits);
    }

    #[test]
    fn test_rsplitnator() {
        let xs = &[1,2,3,4,5];

        let splits: &[&[_]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.rsplitn(0, |x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[5], &[1,2,3]];
        assert_eq!(xs.rsplitn(1, |x| *x % 2 == 0).collect::<Vec<_>>(),
                   splits);
        let splits: &[&[_]] = &[&[], &[], &[], &[1,2]];
        assert_eq!(xs.rsplitn(3, |_| true).collect::<Vec<_>>(),
                   splits);

        let xs: &[i32]  = &[];
        let splits: &[&[i32]] = &[&[]];
        assert_eq!(xs.rsplitn(1, |x| *x == 5).collect::<Vec<&[i32]>>(), splits);
    }

    #[test]
    fn test_windowsator() {
        let v = &[1,2,3,4];

        let wins: &[&[_]] = &[&[1,2], &[2,3], &[3,4]];
        assert_eq!(v.windows(2).collect::<Vec<_>>(), wins);

        let wins: &[&[_]] = &[&[1,2,3], &[2,3,4]];
        assert_eq!(v.windows(3).collect::<Vec<_>>(), wins);
        assert!(v.windows(6).next().is_none());

        let wins: &[&[_]] = &[&[3,4], &[2,3], &[1,2]];
        assert_eq!(v.windows(2).rev().collect::<Vec<&[_]>>(), wins);
        let mut it = v.windows(2);
        assert_eq!(it.indexable(), 3);
        let win: &[_] = &[1,2];
        assert_eq!(it.idx(0).unwrap(), win);
        let win: &[_] = &[2,3];
        assert_eq!(it.idx(1).unwrap(), win);
        let win: &[_] = &[3,4];
        assert_eq!(it.idx(2).unwrap(), win);
        assert_eq!(it.idx(3), None);
    }

    #[test]
    #[should_fail]
    fn test_windowsator_0() {
        let v = &[1,2,3,4];
        let _it = v.windows(0);
    }

    #[test]
    fn test_chunksator() {
        use core::iter::ExactSizeIterator;

        let v = &[1,2,3,4,5];

        assert_eq!(v.chunks(2).len(), 3);

        let chunks: &[&[_]] = &[&[1,2], &[3,4], &[5]];
        assert_eq!(v.chunks(2).collect::<Vec<_>>(), chunks);
        let chunks: &[&[_]] = &[&[1,2,3], &[4,5]];
        assert_eq!(v.chunks(3).collect::<Vec<_>>(), chunks);
        let chunks: &[&[_]] = &[&[1,2,3,4,5]];
        assert_eq!(v.chunks(6).collect::<Vec<_>>(), chunks);

        let chunks: &[&[_]] = &[&[5], &[3,4], &[1,2]];
        assert_eq!(v.chunks(2).rev().collect::<Vec<_>>(), chunks);
        let mut it = v.chunks(2);
        assert_eq!(it.indexable(), 3);

        let chunk: &[_] = &[1,2];
        assert_eq!(it.idx(0).unwrap(), chunk);
        let chunk: &[_] = &[3,4];
        assert_eq!(it.idx(1).unwrap(), chunk);
        let chunk: &[_] = &[5];
        assert_eq!(it.idx(2).unwrap(), chunk);
        assert_eq!(it.idx(3), None);
    }

    #[test]
    #[should_fail]
    fn test_chunksator_0() {
        let v = &[1,2,3,4];
        let _it = v.chunks(0);
    }

    #[test]
    fn test_move_from() {
        let mut a = [1,2,3,4,5];
        let b = vec![6,7,8];
        assert_eq!(a.move_from(b, 0, 3), 3);
        assert!(a == [6,7,8,4,5]);
        let mut a = [7,2,8,1];
        let b = vec![3,1,4,1,5,9];
        assert_eq!(a.move_from(b, 0, 6), 4);
        assert!(a == [3,1,4,1]);
        let mut a = [1,2,3,4];
        let b = vec![5,6,7,8,9,0];
        assert_eq!(a.move_from(b, 2, 3), 1);
        assert!(a == [7,2,3,4]);
        let mut a = [1,2,3,4,5];
        let b = vec![5,6,7,8,9,0];
        assert_eq!(a[2..4].move_from(b,1,6), 2);
        assert!(a == [1,2,6,7,5]);
    }

    #[test]
    fn test_reverse_part() {
        let mut values = [1,2,3,4,5];
        values[1..4].reverse();
        assert!(values == [1,4,3,2,5]);
    }

    #[test]
    fn test_show() {
        macro_rules! test_show_vec {
            ($x:expr, $x_str:expr) => ({
                let (x, x_str) = ($x, $x_str);
                assert_eq!(format!("{:?}", x), x_str);
                assert_eq!(format!("{:?}", x), x_str);
            })
        }
        let empty = Vec::<i32>::new();
        test_show_vec!(empty, "[]");
        test_show_vec!(vec![1], "[1]");
        test_show_vec!(vec![1, 2, 3], "[1, 2, 3]");
        test_show_vec!(vec![vec![], vec![1], vec![1, 1]],
                       "[[], [1], [1, 1]]");

        let empty_mut: &mut [i32] = &mut[];
        test_show_vec!(empty_mut, "[]");
        let v = &mut[1];
        test_show_vec!(v, "[1]");
        let v = &mut[1, 2, 3];
        test_show_vec!(v, "[1, 2, 3]");
        let v: &mut[&mut[_]] = &mut[&mut[], &mut[1], &mut[1, 1]];
        test_show_vec!(v, "[[], [1], [1, 1]]");
    }

    #[test]
    fn test_vec_default() {
        macro_rules! t {
            ($ty:ty) => {{
                let v: $ty = Default::default();
                assert!(v.is_empty());
            }}
        }

        t!(&[i32]);
        t!(Vec<i32>);
    }

    #[test]
    fn test_bytes_set_memory() {
        use slice::bytes::MutableByteVector;
        let mut values = [1u8,2,3,4,5];
        values[0..5].set_memory(0xAB);
        assert!(values == [0xAB, 0xAB, 0xAB, 0xAB, 0xAB]);
        values[2..4].set_memory(0xFF);
        assert!(values == [0xAB, 0xAB, 0xFF, 0xFF, 0xAB]);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault() {
        let mut v = vec![];
        v.reserve_exact(-1);
        v.push(1);
        v.push(2);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault_managed() {
        let mut v = vec![Rc::new(1)];
        v.reserve_exact(-1);
        v.push(Rc::new(2));
    }

    #[test]
    fn test_mut_split_at() {
        let mut values = [1u8,2,3,4,5];
        {
            let (left, right) = values.split_at_mut(2);
            {
                let left: &[_] = left;
                assert!(left[..left.len()] == [1, 2][]);
            }
            for p in left {
                *p += 1;
            }

            {
                let right: &[_] = right;
                assert!(right[..right.len()] == [3, 4, 5][]);
            }
            for p in right {
                *p += 2;
            }
        }

        assert!(values == [2, 3, 5, 6, 7]);
    }

    #[derive(Clone, PartialEq)]
    struct Foo;

    #[test]
    fn test_iter_zero_sized() {
        let mut v = vec![Foo, Foo, Foo];
        assert_eq!(v.len(), 3);
        let mut cnt = 0;

        for f in &v {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 3);

        for f in &v[1..3] {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 5);

        for f in &mut v {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 8);

        for f in v {
            assert!(f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 11);

        let xs: [Foo; 3] = [Foo, Foo, Foo];
        cnt = 0;
        for f in &xs {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert!(cnt == 3);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut xs = vec![0, 1, 2, 3];
        for i in 4..100 {
            xs.push(i)
        }
        assert_eq!(xs.capacity(), 128);
        xs.shrink_to_fit();
        assert_eq!(xs.capacity(), 100);
        assert_eq!(xs, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_starts_with() {
        assert!(b"foobar".starts_with(b"foo"));
        assert!(!b"foobar".starts_with(b"oob"));
        assert!(!b"foobar".starts_with(b"bar"));
        assert!(!b"foo".starts_with(b"foobar"));
        assert!(!b"bar".starts_with(b"foobar"));
        assert!(b"foobar".starts_with(b"foobar"));
        let empty: &[u8] = &[];
        assert!(empty.starts_with(empty));
        assert!(!empty.starts_with(b"foo"));
        assert!(b"foobar".starts_with(empty));
    }

    #[test]
    fn test_ends_with() {
        assert!(b"foobar".ends_with(b"bar"));
        assert!(!b"foobar".ends_with(b"oba"));
        assert!(!b"foobar".ends_with(b"foo"));
        assert!(!b"foo".ends_with(b"foobar"));
        assert!(!b"bar".ends_with(b"foobar"));
        assert!(b"foobar".ends_with(b"foobar"));
        let empty: &[u8] = &[];
        assert!(empty.ends_with(empty));
        assert!(!empty.ends_with(b"foo"));
        assert!(b"foobar".ends_with(empty));
    }

    #[test]
    fn test_mut_splitator() {
        let mut xs = [0,1,0,2,3,0,0,4,5,0];
        assert_eq!(xs.split_mut(|x| *x == 0).count(), 6);
        for slice in xs.split_mut(|x| *x == 0) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0]);

        let mut xs = [0,1,0,2,3,0,0,4,5,0,6,7];
        for slice in xs.split_mut(|x| *x == 0).take(5) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0,6,7]);
    }

    #[test]
    fn test_mut_splitator_rev() {
        let mut xs = [1,2,0,3,4,0,0,5,6,0];
        for slice in xs.split_mut(|x| *x == 0).rev().take(4) {
            slice.reverse();
        }
        assert!(xs == [1,2,0,4,3,0,0,6,5,0]);
    }

    #[test]
    fn test_get_mut() {
        let mut v = [0,1,2];
        assert_eq!(v.get_mut(3), None);
        v.get_mut(1).map(|e| *e = 7);
        assert_eq!(v[1], 7);
        let mut x = 2;
        assert_eq!(v.get_mut(2), Some(&mut x));
    }

    #[test]
    fn test_mut_chunks() {
        use core::iter::ExactSizeIterator;

        let mut v = [0u8, 1, 2, 3, 4, 5, 6];
        assert_eq!(v.chunks_mut(2).len(), 4);
        for (i, chunk) in v.chunks_mut(3).enumerate() {
            for x in chunk {
                *x = i as u8;
            }
        }
        let result = [0u8, 0, 0, 1, 1, 1, 2];
        assert!(v == result);
    }

    #[test]
    fn test_mut_chunks_rev() {
        let mut v = [0u8, 1, 2, 3, 4, 5, 6];
        for (i, chunk) in v.chunks_mut(3).rev().enumerate() {
            for x in chunk {
                *x = i as u8;
            }
        }
        let result = [2u8, 2, 2, 1, 1, 1, 0];
        assert!(v == result);
    }

    #[test]
    #[should_fail]
    fn test_mut_chunks_0() {
        let mut v = [1, 2, 3, 4];
        let _it = v.chunks_mut(0);
    }

    #[test]
    fn test_mut_last() {
        let mut x = [1, 2, 3, 4, 5];
        let h = x.last_mut();
        assert_eq!(*h.unwrap(), 5);

        let y: &mut [i32] = &mut [];
        assert!(y.last_mut().is_none());
    }

    #[test]
    fn test_to_vec() {
        let xs = box [1, 2, 3];
        let ys = xs.to_vec();
        assert_eq!(ys, [1, 2, 3]);
    }
}

#[cfg(test)]
mod bench {
    use prelude::*;
    use core::mem;
    use core::ptr;
    use core::iter::repeat;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    #[bench]
    fn iterator(b: &mut Bencher) {
        // peculiar numbers to stop LLVM from optimising the summation
        // out.
        let v: Vec<_> = (0..100).map(|i| i ^ (i << 1) ^ (i >> 1)).collect();

        b.iter(|| {
            let mut sum = 0;
            for x in &v {
                sum += *x;
            }
            // sum == 11806, to stop dead code elimination.
            if sum == 0 {panic!()}
        })
    }

    #[bench]
    fn mut_iterator(b: &mut Bencher) {
        let mut v: Vec<_> = repeat(0).take(100).collect();

        b.iter(|| {
            let mut i = 0;
            for x in &mut v {
                *x = i;
                i += 1;
            }
        })
    }

    #[bench]
    fn concat(b: &mut Bencher) {
        let xss: Vec<Vec<i32>> =
            (0..100).map(|i| (0..i).collect()).collect();
        b.iter(|| {
            xss.concat();
        });
    }

    #[bench]
    fn connect(b: &mut Bencher) {
        let xss: Vec<Vec<i32>> =
            (0..100).map(|i| (0..i).collect()).collect();
        b.iter(|| {
            xss.connect(&0)
        });
    }

    #[bench]
    fn push(b: &mut Bencher) {
        let mut vec = Vec::<i32>::new();
        b.iter(|| {
            vec.push(0);
            black_box(&vec);
        });
    }

    #[bench]
    fn starts_with_same_vector(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        b.iter(|| {
            vec.starts_with(&vec)
        })
    }

    #[bench]
    fn starts_with_single_element(b: &mut Bencher) {
        let vec: Vec<_> = vec![0];
        b.iter(|| {
            vec.starts_with(&vec)
        })
    }

    #[bench]
    fn starts_with_diff_one_element_at_end(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        let mut match_vec: Vec<_> = (0..99).collect();
        match_vec.push(0);
        b.iter(|| {
            vec.starts_with(&match_vec)
        })
    }

    #[bench]
    fn ends_with_same_vector(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        b.iter(|| {
            vec.ends_with(&vec)
        })
    }

    #[bench]
    fn ends_with_single_element(b: &mut Bencher) {
        let vec: Vec<_> = vec![0];
        b.iter(|| {
            vec.ends_with(&vec)
        })
    }

    #[bench]
    fn ends_with_diff_one_element_at_beginning(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        let mut match_vec: Vec<_> = (0..100).collect();
        match_vec[0] = 200;
        b.iter(|| {
            vec.starts_with(&match_vec)
        })
    }

    #[bench]
    fn contains_last_element(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        b.iter(|| {
            vec.contains(&99)
        })
    }

    #[bench]
    fn zero_1kb_from_elem(b: &mut Bencher) {
        b.iter(|| {
            repeat(0u8).take(1024).collect::<Vec<_>>()
        });
    }

    #[bench]
    fn zero_1kb_set_memory(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::<u8>::with_capacity(1024);
            unsafe {
                let vp = v.as_mut_ptr();
                ptr::set_memory(vp, 0, 1024);
                v.set_len(1024);
            }
            v
        });
    }

    #[bench]
    fn zero_1kb_loop_set(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::<u8>::with_capacity(1024);
            unsafe {
                v.set_len(1024);
            }
            for i in 0..1024 {
                v[i] = 0;
            }
        });
    }

    #[bench]
    fn zero_1kb_mut_iter(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::<u8>::with_capacity(1024);
            unsafe {
                v.set_len(1024);
            }
            for x in &mut v {
                *x = 0;
            }
            v
        });
    }

    #[bench]
    fn random_inserts(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v: Vec<_> = repeat((0, 0)).take(30).collect();
            for _ in 0..100 {
                let l = v.len();
                v.insert(rng.gen::<usize>() % (l + 1),
                         (1, 1));
            }
        })
    }
    #[bench]
    fn random_removes(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v: Vec<_> = repeat((0, 0)).take(130).collect();
            for _ in 0..100 {
                let l = v.len();
                v.remove(rng.gen::<usize>() % l);
            }
        })
    }

    #[bench]
    fn sort_random_small(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v: Vec<_> = rng.gen_iter::<u64>().take(5).collect();
            v.sort();
        });
        b.bytes = 5 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_random_medium(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v: Vec<_> = rng.gen_iter::<u64>().take(100).collect();
            v.sort();
        });
        b.bytes = 100 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_random_large(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v: Vec<_> = rng.gen_iter::<u64>().take(10000).collect();
            v.sort();
        });
        b.bytes = 10000 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_sorted(b: &mut Bencher) {
        let mut v: Vec<_> = (0..10000).collect();
        b.iter(|| {
            v.sort();
        });
        b.bytes = (v.len() * mem::size_of_val(&v[0])) as u64;
    }

    type BigSortable = (u64, u64, u64, u64);

    #[bench]
    fn sort_big_random_small(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<BigSortable>().take(5)
                           .collect::<Vec<BigSortable>>();
            v.sort();
        });
        b.bytes = 5 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_random_medium(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<BigSortable>().take(100)
                           .collect::<Vec<BigSortable>>();
            v.sort();
        });
        b.bytes = 100 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_random_large(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<BigSortable>().take(10000)
                           .collect::<Vec<BigSortable>>();
            v.sort();
        });
        b.bytes = 10000 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_sorted(b: &mut Bencher) {
        let mut v: Vec<BigSortable> = (0..10000).map(|i| (i, i, i, i)).collect();
        b.iter(|| {
            v.sort();
        });
        b.bytes = (v.len() * mem::size_of_val(&v[0])) as u64;
    }
}
