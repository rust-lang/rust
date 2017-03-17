// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Slice management and manipulation
//!
//! For more details see [`std::slice`].
//!
//! [`std::slice`]: ../../std/slice/index.html

#![stable(feature = "rust1", since = "1.0.0")]

// How this module is organized.
//
// The library infrastructure for slices is fairly messy. There's
// a lot of stuff defined here. Let's keep it clean.
//
// Since slices don't support inherent methods; all operations
// on them are defined on traits, which are then reexported from
// the prelude for convenience. So there are a lot of traits here.
//
// The layout of this file is thus:
//
// * Slice-specific 'extension' traits and their implementations. This
//   is where most of the slice API resides.
// * Implementations of a few common traits with important slice ops.
// * Definitions of a bunch of iterators.
// * Free functions.
// * The `raw` and `bytes` submodules.
// * Boilerplate trait implementations.

use borrow::Borrow;
use cmp::Ordering::{self, Less, Equal, Greater};
use cmp;
use fmt;
use intrinsics::assume;
use iter::*;
use ops::{FnMut, self};
use option::Option;
use option::Option::{None, Some};
use result::Result;
use result::Result::{Ok, Err};
use ptr;
use mem;
use marker::{Copy, Send, Sync, Sized, self};
use iter_private::TrustedRandomAccess;

#[repr(C)]
struct Repr<T> {
    pub data: *const T,
    pub len: usize,
}

//
// Extension traits
//

/// Extension methods for slices.
#[unstable(feature = "core_slice_ext",
           reason = "stable interface provided by `impl [T]` in later crates",
           issue = "32110")]
#[allow(missing_docs)] // documented elsewhere
pub trait SliceExt {
    type Item;

    #[stable(feature = "core", since = "1.6.0")]
    fn split_at(&self, mid: usize) -> (&[Self::Item], &[Self::Item]);
    #[stable(feature = "core", since = "1.6.0")]
    fn iter(&self) -> Iter<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split<P>(&self, pred: P) -> Split<Self::Item, P>
                    where P: FnMut(&Self::Item) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn splitn<P>(&self, n: usize, pred: P) -> SplitN<Self::Item, P>
                     where P: FnMut(&Self::Item) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn rsplitn<P>(&self,  n: usize, pred: P) -> RSplitN<Self::Item, P>
                      where P: FnMut(&Self::Item) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn windows(&self, size: usize) -> Windows<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn chunks(&self, size: usize) -> Chunks<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn get<I>(&self, index: I) -> Option<&I::Output>
        where I: SliceIndex<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn first(&self) -> Option<&Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_first(&self) -> Option<(&Self::Item, &[Self::Item])>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_last(&self) -> Option<(&Self::Item, &[Self::Item])>;
    #[stable(feature = "core", since = "1.6.0")]
    fn last(&self) -> Option<&Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_ptr(&self) -> *const Self::Item;
    #[stable(feature = "core", since = "1.6.0")]
    fn binary_search<Q: ?Sized>(&self, x: &Q) -> Result<usize, usize>
        where Self::Item: Borrow<Q>,
              Q: Ord;
    #[stable(feature = "core", since = "1.6.0")]
    fn binary_search_by<'a, F>(&'a self, f: F) -> Result<usize, usize>
        where F: FnMut(&'a Self::Item) -> Ordering;
    #[stable(feature = "slice_binary_search_by_key", since = "1.10.0")]
    fn binary_search_by_key<'a, B, F, Q: ?Sized>(&'a self, b: &Q, f: F) -> Result<usize, usize>
        where F: FnMut(&'a Self::Item) -> B,
              B: Borrow<Q>,
              Q: Ord;
    #[stable(feature = "core", since = "1.6.0")]
    fn len(&self) -> usize;
    #[stable(feature = "core", since = "1.6.0")]
    fn is_empty(&self) -> bool { self.len() == 0 }
    #[stable(feature = "core", since = "1.6.0")]
    fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
        where I: SliceIndex<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn iter_mut(&mut self) -> IterMut<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn first_mut(&mut self) -> Option<&mut Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_first_mut(&mut self) -> Option<(&mut Self::Item, &mut [Self::Item])>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_last_mut(&mut self) -> Option<(&mut Self::Item, &mut [Self::Item])>;
    #[stable(feature = "core", since = "1.6.0")]
    fn last_mut(&mut self) -> Option<&mut Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_mut<P>(&mut self, pred: P) -> SplitMut<Self::Item, P>
                        where P: FnMut(&Self::Item) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn splitn_mut<P>(&mut self, n: usize, pred: P) -> SplitNMut<Self::Item, P>
                     where P: FnMut(&Self::Item) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn rsplitn_mut<P>(&mut self,  n: usize, pred: P) -> RSplitNMut<Self::Item, P>
                      where P: FnMut(&Self::Item) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn swap(&mut self, a: usize, b: usize);
    #[stable(feature = "core", since = "1.6.0")]
    fn split_at_mut(&mut self, mid: usize) -> (&mut [Self::Item], &mut [Self::Item]);
    #[stable(feature = "core", since = "1.6.0")]
    fn reverse(&mut self);
    #[stable(feature = "core", since = "1.6.0")]
    unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
        where I: SliceIndex<Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_mut_ptr(&mut self) -> *mut Self::Item;

    #[stable(feature = "core", since = "1.6.0")]
    fn contains(&self, x: &Self::Item) -> bool where Self::Item: PartialEq;

    #[stable(feature = "core", since = "1.6.0")]
    fn starts_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    #[stable(feature = "core", since = "1.6.0")]
    fn ends_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    #[stable(feature = "clone_from_slice", since = "1.7.0")]
    fn clone_from_slice(&mut self, src: &[Self::Item]) where Self::Item: Clone;
    #[stable(feature = "copy_from_slice", since = "1.9.0")]
    fn copy_from_slice(&mut self, src: &[Self::Item]) where Self::Item: Copy;
}

// Use macros to be generic over const/mut
macro_rules! slice_offset {
    ($ptr:expr, $by:expr) => {{
        let ptr = $ptr;
        if size_from_ptr(ptr) == 0 {
            (ptr as *mut i8).wrapping_offset($by) as _
        } else {
            ptr.offset($by)
        }
    }};
}

// make a &T from a *const T
macro_rules! make_ref {
    ($ptr:expr) => {{
        let ptr = $ptr;
        if size_from_ptr(ptr) == 0 {
            // Use a non-null pointer value
            &*(1 as *mut _)
        } else {
            &*ptr
        }
    }};
}

// make a &mut T from a *mut T
macro_rules! make_ref_mut {
    ($ptr:expr) => {{
        let ptr = $ptr;
        if size_from_ptr(ptr) == 0 {
            // Use a non-null pointer value
            &mut *(1 as *mut _)
        } else {
            &mut *ptr
        }
    }};
}

#[unstable(feature = "core_slice_ext",
           reason = "stable interface provided by `impl [T]` in later crates",
           issue = "32110")]
impl<T> SliceExt for [T] {
    type Item = T;

    #[inline]
    fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        (&self[..mid], &self[mid..])
    }

    #[inline]
    fn iter(&self) -> Iter<T> {
        unsafe {
            let p = if mem::size_of::<T>() == 0 {
                1 as *const _
            } else {
                let p = self.as_ptr();
                assume(!p.is_null());
                p
            };

            Iter {
                ptr: p,
                end: slice_offset!(p, self.len() as isize),
                _marker: marker::PhantomData
            }
        }
    }

    #[inline]
    fn split<P>(&self, pred: P) -> Split<T, P> where P: FnMut(&T) -> bool {
        Split {
            v: self,
            pred: pred,
            finished: false
        }
    }

    #[inline]
    fn splitn<P>(&self, n: usize, pred: P) -> SplitN<T, P> where
        P: FnMut(&T) -> bool,
    {
        SplitN {
            inner: GenericSplitN {
                iter: self.split(pred),
                count: n,
                invert: false
            }
        }
    }

    #[inline]
    fn rsplitn<P>(&self, n: usize, pred: P) -> RSplitN<T, P> where
        P: FnMut(&T) -> bool,
    {
        RSplitN {
            inner: GenericSplitN {
                iter: self.split(pred),
                count: n,
                invert: true
            }
        }
    }

    #[inline]
    fn windows(&self, size: usize) -> Windows<T> {
        assert!(size != 0);
        Windows { v: self, size: size }
    }

    #[inline]
    fn chunks(&self, size: usize) -> Chunks<T> {
        assert!(size != 0);
        Chunks { v: self, size: size }
    }

    #[inline]
    fn get<I>(&self, index: I) -> Option<&I::Output>
        where I: SliceIndex<T>
    {
        index.get(self)
    }

    #[inline]
    fn first(&self) -> Option<&T> {
        if self.is_empty() { None } else { Some(&self[0]) }
    }

    #[inline]
    fn split_first(&self) -> Option<(&T, &[T])> {
        if self.is_empty() { None } else { Some((&self[0], &self[1..])) }
    }

    #[inline]
    fn split_last(&self) -> Option<(&T, &[T])> {
        let len = self.len();
        if len == 0 { None } else { Some((&self[len - 1], &self[..(len - 1)])) }
    }

    #[inline]
    fn last(&self) -> Option<&T> {
        if self.is_empty() { None } else { Some(&self[self.len() - 1]) }
    }

    #[inline]
    unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<T>
    {
        index.get_unchecked(self)
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self as *const [T] as *const T
    }

    fn binary_search_by<'a, F>(&'a self, mut f: F) -> Result<usize, usize>
        where F: FnMut(&'a T) -> Ordering
    {
        let mut base = 0usize;
        let mut s = self;

        loop {
            let (head, tail) = s.split_at(s.len() >> 1);
            if tail.is_empty() {
                return Err(base)
            }
            match f(&tail[0]) {
                Less => {
                    base += head.len() + 1;
                    s = &tail[1..];
                }
                Greater => s = head,
                Equal => return Ok(base + head.len()),
            }
        }
    }

    #[inline]
    fn len(&self) -> usize {
        unsafe {
            mem::transmute::<&[T], Repr<T>>(self).len
        }
    }

    #[inline]
    fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
        where I: SliceIndex<T>
    {
        index.get_mut(self)
    }

    #[inline]
    fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        let len = self.len();
        let ptr = self.as_mut_ptr();

        unsafe {
            assert!(mid <= len);

            (from_raw_parts_mut(ptr, mid),
             from_raw_parts_mut(ptr.offset(mid as isize), len - mid))
        }
    }

    #[inline]
    fn iter_mut(&mut self) -> IterMut<T> {
        unsafe {
            let p = if mem::size_of::<T>() == 0 {
                1 as *mut _
            } else {
                let p = self.as_mut_ptr();
                assume(!p.is_null());
                p
            };

            IterMut {
                ptr: p,
                end: slice_offset!(p, self.len() as isize),
                _marker: marker::PhantomData
            }
        }
    }

    #[inline]
    fn last_mut(&mut self) -> Option<&mut T> {
        let len = self.len();
        if len == 0 { return None; }
        Some(&mut self[len - 1])
    }

    #[inline]
    fn first_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() { None } else { Some(&mut self[0]) }
    }

    #[inline]
    fn split_first_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        if self.is_empty() { None } else {
            let split = self.split_at_mut(1);
            Some((&mut split.0[0], split.1))
        }
    }

    #[inline]
    fn split_last_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        let len = self.len();
        if len == 0 { None } else {
            let split = self.split_at_mut(len - 1);
            Some((&mut split.1[0], split.0))
        }
    }

    #[inline]
    fn split_mut<P>(&mut self, pred: P) -> SplitMut<T, P> where P: FnMut(&T) -> bool {
        SplitMut { v: self, pred: pred, finished: false }
    }

    #[inline]
    fn splitn_mut<P>(&mut self, n: usize, pred: P) -> SplitNMut<T, P> where
        P: FnMut(&T) -> bool
    {
        SplitNMut {
            inner: GenericSplitN {
                iter: self.split_mut(pred),
                count: n,
                invert: false
            }
        }
    }

    #[inline]
    fn rsplitn_mut<P>(&mut self, n: usize, pred: P) -> RSplitNMut<T, P> where
        P: FnMut(&T) -> bool,
    {
        RSplitNMut {
            inner: GenericSplitN {
                iter: self.split_mut(pred),
                count: n,
                invert: true
            }
        }
   }

    #[inline]
    fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        assert!(chunk_size > 0);
        ChunksMut { v: self, chunk_size: chunk_size }
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        unsafe {
            // Can't take two mutable loans from one vector, so instead just cast
            // them to their raw pointers to do the swap
            let pa: *mut T = &mut self[a];
            let pb: *mut T = &mut self[b];
            ptr::swap(pa, pb);
        }
    }

    fn reverse(&mut self) {
        let mut i: usize = 0;
        let ln = self.len();
        while i < ln / 2 {
            // Unsafe swap to avoid the bounds check in safe swap.
            unsafe {
                let pa: *mut T = self.get_unchecked_mut(i);
                let pb: *mut T = self.get_unchecked_mut(ln - i - 1);
                ptr::swap(pa, pb);
            }
            i += 1;
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
        where I: SliceIndex<T>
    {
        index.get_unchecked_mut(self)
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut [T] as *mut T
    }

    #[inline]
    fn contains(&self, x: &T) -> bool where T: PartialEq {
        self.iter().any(|elt| *x == *elt)
    }

    #[inline]
    fn starts_with(&self, needle: &[T]) -> bool where T: PartialEq {
        let n = needle.len();
        self.len() >= n && needle == &self[..n]
    }

    #[inline]
    fn ends_with(&self, needle: &[T]) -> bool where T: PartialEq {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle == &self[m-n..]
    }

    fn binary_search<Q: ?Sized>(&self, x: &Q) -> Result<usize, usize> where T: Borrow<Q>, Q: Ord {
        self.binary_search_by(|p| p.borrow().cmp(x))
    }

    #[inline]
    fn clone_from_slice(&mut self, src: &[T]) where T: Clone {
        assert!(self.len() == src.len(),
                "destination and source slices have different lengths");
        // NOTE: We need to explicitly slice them to the same length
        // for bounds checking to be elided, and the optimizer will
        // generate memcpy for simple cases (for example T = u8).
        let len = self.len();
        let src = &src[..len];
        for i in 0..len {
            self[i].clone_from(&src[i]);
        }
    }

    #[inline]
    fn copy_from_slice(&mut self, src: &[T]) where T: Copy {
        assert!(self.len() == src.len(),
                "destination and source slices have different lengths");
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr(), self.as_mut_ptr(), self.len());
        }
    }

    #[inline]
    fn binary_search_by_key<'a, B, F, Q: ?Sized>(&'a self, b: &Q, mut f: F) -> Result<usize, usize>
        where F: FnMut(&'a Self::Item) -> B,
              B: Borrow<Q>,
              Q: Ord
    {
        self.binary_search_by(|k| f(k).borrow().cmp(b))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
impl<T, I> ops::Index<I> for [T]
    where I: SliceIndex<T>
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
impl<T, I> ops::IndexMut<I> for [T]
    where I: SliceIndex<T>
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

#[inline(never)]
#[cold]
fn slice_index_len_fail(index: usize, len: usize) -> ! {
    panic!("index {} out of range for slice of length {}", index, len);
}

#[inline(never)]
#[cold]
fn slice_index_order_fail(index: usize, end: usize) -> ! {
    panic!("slice index starts at {} but ends at {}", index, end);
}

/// A helper trait used for indexing operations.
#[unstable(feature = "slice_get_slice", issue = "35729")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
pub trait SliceIndex<T> {
    /// The output type returned by methods.
    type Output: ?Sized;

    /// Returns a shared reference to the output at this location, if in
    /// bounds.
    fn get(self, slice: &[T]) -> Option<&Self::Output>;

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output>;

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    unsafe fn get_unchecked(self, slice: &[T]) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut Self::Output;

    /// Returns a shared reference to the output at this location, panicking
    /// if out of bounds.
    fn index(self, slice: &[T]) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, panicking
    /// if out of bounds.
    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output;
}

#[stable(feature = "slice-get-slice-impls", since = "1.15.0")]
impl<T> SliceIndex<T> for usize {
    type Output = T;

    #[inline]
    fn get(self, slice: &[T]) -> Option<&T> {
        if self < slice.len() {
            unsafe {
                Some(self.get_unchecked(slice))
            }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut T> {
        if self < slice.len() {
            unsafe {
                Some(self.get_unchecked_mut(slice))
            }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &T {
        &*slice.as_ptr().offset(self as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut T {
        &mut *slice.as_mut_ptr().offset(self as isize)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &T {
        // NB: use intrinsic indexing
        &(*slice)[self]
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut T {
        // NB: use intrinsic indexing
        &mut (*slice)[self]
    }
}

#[stable(feature = "slice-get-slice-impls", since = "1.15.0")]
impl<T> SliceIndex<T> for  ops::Range<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        if self.start > self.end || self.end > slice.len() {
            None
        } else {
            unsafe {
                Some(self.get_unchecked(slice))
            }
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        if self.start > self.end || self.end > slice.len() {
            None
        } else {
            unsafe {
                Some(self.get_unchecked_mut(slice))
            }
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        from_raw_parts(slice.as_ptr().offset(self.start as isize), self.end - self.start)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        from_raw_parts_mut(slice.as_mut_ptr().offset(self.start as isize), self.end - self.start)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_index_len_fail(self.end, slice.len());
        }
        unsafe {
            self.get_unchecked(slice)
        }
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_index_len_fail(self.end, slice.len());
        }
        unsafe {
            self.get_unchecked_mut(slice)
        }
    }
}

#[stable(feature = "slice-get-slice-impls", since = "1.15.0")]
impl<T> SliceIndex<T> for ops::RangeTo<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0..self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0..self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (0..self.end).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..self.end).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0..self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..self.end).index_mut(slice)
    }
}

#[stable(feature = "slice-get-slice-impls", since = "1.15.0")]
impl<T> SliceIndex<T> for ops::RangeFrom<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (self.start..slice.len()).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (self.start..slice.len()).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (self.start..slice.len()).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (self.start..slice.len()).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (self.start..slice.len()).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (self.start..slice.len()).index_mut(slice)
    }
}

#[stable(feature = "slice-get-slice-impls", since = "1.15.0")]
impl<T> SliceIndex<T> for ops::RangeFull {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        Some(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        Some(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        slice
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        slice
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        slice
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        slice
    }
}


#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<T> SliceIndex<T> for ops::RangeInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        match self {
            ops::RangeInclusive::Empty { .. } => Some(&[]),
            ops::RangeInclusive::NonEmpty { end, .. } if end == usize::max_value() => None,
            ops::RangeInclusive::NonEmpty { start, end } => (start..end + 1).get(slice),
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        match self {
            ops::RangeInclusive::Empty { .. } => Some(&mut []),
            ops::RangeInclusive::NonEmpty { end, .. } if end == usize::max_value() => None,
            ops::RangeInclusive::NonEmpty { start, end } => (start..end + 1).get_mut(slice),
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        match self {
            ops::RangeInclusive::Empty { .. } => &[],
            ops::RangeInclusive::NonEmpty { start, end } => (start..end + 1).get_unchecked(slice),
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        match self {
            ops::RangeInclusive::Empty { .. } => &mut [],
            ops::RangeInclusive::NonEmpty { start, end } => {
                (start..end + 1).get_unchecked_mut(slice)
            }
        }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        match self {
            ops::RangeInclusive::Empty { .. } => &[],
            ops::RangeInclusive::NonEmpty { end, .. } if end == usize::max_value() => {
                panic!("attempted to index slice up to maximum usize");
            },
            ops::RangeInclusive::NonEmpty { start, end } => (start..end + 1).index(slice),
        }
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        match self {
            ops::RangeInclusive::Empty { .. } => &mut [],
            ops::RangeInclusive::NonEmpty { end, .. } if end == usize::max_value() => {
                panic!("attempted to index slice up to maximum usize");
            },
            ops::RangeInclusive::NonEmpty { start, end } => (start..end + 1).index_mut(slice),
        }
    }
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<T> SliceIndex<T> for ops::RangeToInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0...self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0...self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (0...self.end).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (0...self.end).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0...self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0...self.end).index_mut(slice)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common traits
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Default for &'a [T] {
    /// Creates an empty slice.
    fn default() -> &'a [T] { &[] }
}

#[stable(feature = "mut_slice_default", since = "1.5.0")]
impl<'a, T> Default for &'a mut [T] {
    /// Creates a mutable empty slice.
    fn default() -> &'a mut [T] { &mut [] }
}

//
// Iterators
//

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a [T] {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut [T] {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[inline(always)]
fn size_from_ptr<T>(_: *const T) -> usize {
    mem::size_of::<T>()
}

// The shared definition of the `Iter` and `IterMut` iterators
macro_rules! iterator {
    (struct $name:ident -> $ptr:ty, $elem:ty, $mkref:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> Iterator for $name<'a, T> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    if mem::size_of::<T>() != 0 {
                        assume(!self.ptr.is_null());
                        assume(!self.end.is_null());
                    }
                    if self.ptr == self.end {
                        None
                    } else {
                        Some($mkref!(self.ptr.post_inc()))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let exact = ptrdistance(self.ptr, self.end);
                (exact, Some(exact))
            }

            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                // Call helper method. Can't put the definition here because mut versus const.
                self.iter_nth(n)
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            fn all<F>(&mut self, mut predicate: F) -> bool
                where F: FnMut(Self::Item) -> bool,
            {
                self.search_while(true, move |elt| {
                    if predicate(elt) {
                        SearchWhile::Continue
                    } else {
                        SearchWhile::Done(false)
                    }
                })
            }

            fn any<F>(&mut self, mut predicate: F) -> bool
                where F: FnMut(Self::Item) -> bool,
            {
                !self.all(move |elt| !predicate(elt))
            }

            fn find<F>(&mut self, mut predicate: F) -> Option<Self::Item>
                where F: FnMut(&Self::Item) -> bool,
            {
                self.search_while(None, move |elt| {
                    if predicate(&elt) {
                        SearchWhile::Done(Some(elt))
                    } else {
                        SearchWhile::Continue
                    }
                })
            }

            fn position<F>(&mut self, mut predicate: F) -> Option<usize>
                where F: FnMut(Self::Item) -> bool,
            {
                let mut index = 0;
                self.search_while(None, move |elt| {
                    if predicate(elt) {
                        SearchWhile::Done(Some(index))
                    } else {
                        index += 1;
                        SearchWhile::Continue
                    }
                })
            }

            fn rposition<F>(&mut self, mut predicate: F) -> Option<usize>
                where F: FnMut(Self::Item) -> bool,
            {
                let mut index = self.len();
                self.rsearch_while(None, move |elt| {
                    index -= 1;
                    if predicate(elt) {
                        SearchWhile::Done(Some(index))
                    } else {
                        SearchWhile::Continue
                    }
                })
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    if mem::size_of::<T>() != 0 {
                        assume(!self.ptr.is_null());
                        assume(!self.end.is_null());
                    }
                    if self.end == self.ptr {
                        None
                    } else {
                        Some($mkref!(self.end.pre_dec()))
                    }
                }
            }
        }

        // search_while is a generalization of the internal iteration methods.
        impl<'a, T> $name<'a, T> {
            // search through the iterator's element using the closure `g`.
            // if no element was found, return `default`.
            fn search_while<Acc, G>(&mut self, default: Acc, mut g: G) -> Acc
                where Self: Sized,
                      G: FnMut($elem) -> SearchWhile<Acc>
            {
                // manual unrolling is needed when there are conditional exits from the loop
                unsafe {
                    while ptrdistance(self.ptr, self.end) >= 4 {
                        search_while!(g($mkref!(self.ptr.post_inc())));
                        search_while!(g($mkref!(self.ptr.post_inc())));
                        search_while!(g($mkref!(self.ptr.post_inc())));
                        search_while!(g($mkref!(self.ptr.post_inc())));
                    }
                    while self.ptr != self.end {
                        search_while!(g($mkref!(self.ptr.post_inc())));
                    }
                }
                default
            }

            fn rsearch_while<Acc, G>(&mut self, default: Acc, mut g: G) -> Acc
                where Self: Sized,
                      G: FnMut($elem) -> SearchWhile<Acc>
            {
                unsafe {
                    while ptrdistance(self.ptr, self.end) >= 4 {
                        search_while!(g($mkref!(self.end.pre_dec())));
                        search_while!(g($mkref!(self.end.pre_dec())));
                        search_while!(g($mkref!(self.end.pre_dec())));
                        search_while!(g($mkref!(self.end.pre_dec())));
                    }
                    while self.ptr != self.end {
                        search_while!(g($mkref!(self.end.pre_dec())));
                    }
                }
                default
            }
        }
    }
}

macro_rules! make_slice {
    ($start: expr, $end: expr) => {{
        let start = $start;
        let diff = ($end as usize).wrapping_sub(start as usize);
        if size_from_ptr(start) == 0 {
            // use a non-null pointer value
            unsafe { from_raw_parts(1 as *const _, diff) }
        } else {
            let len = diff / size_from_ptr(start);
            unsafe { from_raw_parts(start, len) }
        }
    }}
}

macro_rules! make_mut_slice {
    ($start: expr, $end: expr) => {{
        let start = $start;
        let diff = ($end as usize).wrapping_sub(start as usize);
        if size_from_ptr(start) == 0 {
            // use a non-null pointer value
            unsafe { from_raw_parts_mut(1 as *mut _, diff) }
        } else {
            let len = diff / size_from_ptr(start);
            unsafe { from_raw_parts_mut(start, len) }
        }
    }}
}

// An enum used for controlling the execution of `.search_while()`.
enum SearchWhile<T> {
    // Continue searching
    Continue,
    // Fold is complete and will return this value
    Done(T),
}

// helper macro for search while's control flow
macro_rules! search_while {
    ($e:expr) => {
        match $e {
            SearchWhile::Continue => { }
            SearchWhile::Done(done) => return done,
        }
    }
}

/// Immutable slice iterator
///
/// This struct is created by the [`iter`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter` method to get the `Iter` struct (&[usize here]):
/// let slice = &[1, 2, 3];
///
/// // Then, we iterate over it:
/// for element in slice.iter() {
///     println!("{}", element);
/// }
/// ```
///
/// [`iter`]: ../../std/primitive.slice.html#method.iter
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _marker: marker::PhantomData<&'a T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug> fmt::Debug for Iter<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Iter")
            .field(&self.as_slice())
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Sync> Sync for Iter<'a, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Sync> Send for Iter<'a, T> {}

impl<'a, T> Iter<'a, T> {
    /// View the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has the `iter` method to get the `Iter`
    /// // struct (&[usize here]):
    /// let slice = &[1, 2, 3];
    ///
    /// // Then, we get the iterator:
    /// let mut iter = slice.iter();
    /// // So if we print what `as_slice` method returns here, we have "[1, 2, 3]":
    /// println!("{:?}", iter.as_slice());
    ///
    /// // Next, we move to the second element of the slice:
    /// iter.next();
    /// // Now `as_slice` returns "[2, 3]":
    /// println!("{:?}", iter.as_slice());
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn as_slice(&self) -> &'a [T] {
        make_slice!(self.ptr, self.end)
    }

    // Helper function for Iter::nth
    fn iter_nth(&mut self, n: usize) -> Option<&'a T> {
        match self.as_slice().get(n) {
            Some(elem_ref) => unsafe {
                self.ptr = slice_offset!(self.ptr, (n as isize).wrapping_add(1));
                Some(elem_ref)
            },
            None => {
                self.ptr = self.end;
                None
            }
        }
    }
}

iterator!{struct Iter -> *const T, &'a T, make_ref}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn is_empty(&self) -> bool {
        self.ptr == self.end
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for Iter<'a, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for Iter<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> { Iter { ptr: self.ptr, end: self.end, _marker: self._marker } }
}

#[stable(feature = "slice_iter_as_ref", since = "1.12.0")]
impl<'a, T> AsRef<[T]> for Iter<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// Mutable slice iterator.
///
/// This struct is created by the [`iter_mut`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter_mut` method to get the `IterMut`
/// // struct (&[usize here]):
/// let mut slice = &mut [1, 2, 3];
///
/// // Then, we iterate over it and increment each element value:
/// for element in slice.iter_mut() {
///     *element += 1;
/// }
///
/// // We now have "[2, 3, 4]":
/// println!("{:?}", slice);
/// ```
///
/// [`iter_mut`]: ../../std/primitive.slice.html#method.iter_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    ptr: *mut T,
    end: *mut T,
    _marker: marker::PhantomData<&'a mut T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug> fmt::Debug for IterMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IterMut")
            .field(&make_slice!(self.ptr, self.end))
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Sync> Sync for IterMut<'a, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Send> Send for IterMut<'a, T> {}

impl<'a, T> IterMut<'a, T> {
    /// View the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut` references that alias, this is forced
    /// to consume the iterator. Consider using the `Slice` and
    /// `SliceMut` implementations for obtaining slices with more
    /// restricted lifetimes that do not consume the iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has `iter_mut` method to get the `IterMut`
    /// // struct (&[usize here]):
    /// let mut slice = &mut [1, 2, 3];
    ///
    /// {
    ///     // Then, we get the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We move to next element:
    ///     iter.next();
    ///     // So if we print what `into_slice` method returns here, we have "[2, 3]":
    ///     println!("{:?}", iter.into_slice());
    /// }
    ///
    /// // Now let's modify a value of the slice:
    /// {
    ///     // First we get back the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We change the value of the first element of the slice returned by the `next` method:
    ///     *iter.next().unwrap() += 1;
    /// }
    /// // Now slice is "[2, 2, 3]":
    /// println!("{:?}", slice);
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn into_slice(self) -> &'a mut [T] {
        make_mut_slice!(self.ptr, self.end)
    }

    // Helper function for IterMut::nth
    fn iter_nth(&mut self, n: usize) -> Option<&'a mut T> {
        match make_mut_slice!(self.ptr, self.end).get_mut(n) {
            Some(elem_ref) => unsafe {
                self.ptr = slice_offset!(self.ptr, (n as isize).wrapping_add(1));
                Some(elem_ref)
            },
            None => {
                self.ptr = self.end;
                None
            }
        }
    }
}

iterator!{struct IterMut -> *mut T, &'a mut T, make_ref_mut}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn is_empty(&self) -> bool {
        self.ptr == self.end
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for IterMut<'a, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for IterMut<'a, T> {}


// Return the number of elements of `T` from `start` to `end`.
// Return the arithmetic difference if `T` is zero size.
#[inline(always)]
fn ptrdistance<T>(start: *const T, end: *const T) -> usize {
    let diff = (end as usize).wrapping_sub(start as usize);
    let size = mem::size_of::<T>();
    diff / (if size == 0 { 1 } else { size })
}

// Extension methods for raw pointers, used by the iterators
trait PointerExt : Copy {
    unsafe fn slice_offset(self, i: isize) -> Self;

    /// Increment self by 1, but return the old value
    #[inline(always)]
    unsafe fn post_inc(&mut self) -> Self {
        let current = *self;
        *self = self.slice_offset(1);
        current
    }

    /// Decrement self by 1, and return the new value
    #[inline(always)]
    unsafe fn pre_dec(&mut self) -> Self {
        *self = self.slice_offset(-1);
        *self
    }
}

impl<T> PointerExt for *const T {
    #[inline(always)]
    unsafe fn slice_offset(self, i: isize) -> Self {
        slice_offset!(self, i)
    }
}

impl<T> PointerExt for *mut T {
    #[inline(always)]
    unsafe fn slice_offset(self, i: isize) -> Self {
        slice_offset!(self, i)
    }
}

/// An internal abstraction over the splitting iterators, so that
/// splitn, splitn_mut etc can be implemented once.
#[doc(hidden)]
trait SplitIter: DoubleEndedIterator {
    /// Mark the underlying iterator as complete, extracting the remaining
    /// portion of the slice.
    fn finish(&mut self) -> Option<Self::Item>;
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
///
/// This struct is created by the [`split`] method on [slices].
///
/// [`split`]: ../../std/primitive.slice.html#method.split
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Split<'a, T:'a, P> where P: FnMut(&T) -> bool {
    v: &'a [T],
    pred: P,
    finished: bool
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for Split<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Split")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Clone for Split<'a, T, P> where P: Clone + FnMut(&T) -> bool {
    fn clone(&self) -> Split<'a, T, P> {
        Split {
            v: self.v,
            pred: self.pred.clone(),
            finished: self.finished,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for Split<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished { return None; }

        match self.v.iter().position(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[..idx]);
                self.v = &self.v[idx + 1..];
                ret
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for Split<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.finished { return None; }

        match self.v.iter().rposition(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[idx + 1..]);
                self.v = &self.v[..idx];
                ret
            }
        }
    }
}

impl<'a, T, P> SplitIter for Split<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        if self.finished { None } else { self.finished = true; Some(self.v) }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T, P> FusedIterator for Split<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`.
///
/// This struct is created by the [`split_mut`] method on [slices].
///
/// [`split_mut`]: ../../std/primitive.slice.html#method.split_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitMut<'a, T:'a, P> where P: FnMut(&T) -> bool {
    v: &'a mut [T],
    pred: P,
    finished: bool
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitMut")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, T, P> SplitIter for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(mem::replace(&mut self.v, &mut []))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished { return None; }

        let idx_opt = { // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().position(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = &mut tail[1..];
                Some(head)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // if the predicate doesn't match anything, we yield one slice
            // if it matches every element, we yield len+1 empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for SplitMut<'a, T, P> where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished { return None; }

        let idx_opt = { // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().rposition(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = head;
                Some(&mut tail[1..])
            }
        }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T, P> FusedIterator for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {}

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
#[derive(Debug)]
struct GenericSplitN<I> {
    iter: I,
    count: usize,
    invert: bool
}

impl<T, I: SplitIter<Item=T>> Iterator for GenericSplitN<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.count {
            0 => None,
            1 => { self.count -= 1; self.iter.finish() }
            _ => {
                self.count -= 1;
                if self.invert {self.iter.next_back()} else {self.iter.next()}
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper_opt) = self.iter.size_hint();
        (lower, upper_opt.map(|upper| cmp::min(self.count, upper)))
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn`] method on [slices].
///
/// [`splitn`]: ../../std/primitive.slice.html#method.splitn
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitN<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<Split<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for SplitN<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitN")
            .field("inner", &self.inner)
            .finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn`] method on [slices].
///
/// [`rsplitn`]: ../../std/primitive.slice.html#method.rsplitn
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitN<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<Split<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for RSplitN<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplitN")
            .field("inner", &self.inner)
            .finish()
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn_mut`] method on [slices].
///
/// [`splitn_mut`]: ../../std/primitive.slice.html#method.splitn_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitNMut<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<SplitMut<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for SplitNMut<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitNMut")
            .field("inner", &self.inner)
            .finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn_mut`] method on [slices].
///
/// [`rsplitn_mut`]: ../../std/primitive.slice.html#method.rsplitn_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitNMut<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<SplitMut<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for RSplitNMut<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplitNMut")
            .field("inner", &self.inner)
            .finish()
    }
}

macro_rules! forward_iterator {
    ($name:ident: $elem:ident, $iter_of:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, $elem, P> Iterator for $name<'a, $elem, P> where
            P: FnMut(&T) -> bool
        {
            type Item = $iter_of;

            #[inline]
            fn next(&mut self) -> Option<$iter_of> {
                self.inner.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        #[unstable(feature = "fused", issue = "35602")]
        impl<'a, $elem, P> FusedIterator for $name<'a, $elem, P>
            where P: FnMut(&T) -> bool {}
    }
}

forward_iterator! { SplitN: T, &'a [T] }
forward_iterator! { RSplitN: T, &'a [T] }
forward_iterator! { SplitNMut: T, &'a mut [T] }
forward_iterator! { RSplitNMut: T, &'a mut [T] }

/// An iterator over overlapping subslices of length `size`.
///
/// This struct is created by the [`windows`] method on [slices].
///
/// [`windows`]: ../../std/primitive.slice.html#method.windows
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Windows<'a, T:'a> {
    v: &'a [T],
    size: usize
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Windows<'a, T> {
    fn clone(&self) -> Windows<'a, T> {
        Windows {
            v: self.v,
            size: self.size,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Windows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[..self.size]);
            self.v = &self.v[1..];
            ret
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size > self.v.len() {
            (0, Some(0))
        } else {
            let size = self.v.len() - self.size + 1;
            (size, Some(size))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = self.size.overflowing_add(n);
        if end > self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let nth = &self.v[n..end];
            self.v = &self.v[n+1..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.size > self.v.len() {
            None
        } else {
            let start = self.v.len() - self.size;
            Some(&self.v[start..])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Windows<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[self.v.len()-self.size..]);
            self.v = &self.v[..self.v.len()-1];
            ret
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Windows<'a, T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for Windows<'a, T> {}

/// An iterator over a slice in (non-overlapping) chunks (`size` elements at a
/// time).
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks`] method on [slices].
///
/// [`chunks`]: ../../std/primitive.slice.html#method.chunks
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chunks<'a, T:'a> {
    v: &'a [T],
    size: usize
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Chunks<'a, T> {
    fn clone(&self) -> Chunks<'a, T> {
        Chunks {
            v: self.v,
            size: self.size,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.size);
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.size;
            let rem = self.v.len() % self.size;
            let n = if rem > 0 { n+1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let end = match start.checked_add(self.size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let nth = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.size * self.size;
            Some(&self.v[start..])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.size;
            let chunksz = if remainder != 0 { remainder } else { self.size };
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Chunks<'a, T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for Chunks<'a, T> {}

/// An iterator over a slice in (non-overlapping) mutable chunks (`size`
/// elements at a time). When the slice len is not evenly divided by the chunk
/// size, the last slice of the iteration will be the remainder.
///
/// This struct is created by the [`chunks_mut`] method on [slices].
///
/// [`chunks_mut`]: ../../std/primitive.slice.html#method.chunks_mut
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ChunksMut<'a, T:'a> {
    v: &'a mut [T],
    chunk_size: usize
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(end);
            let (_, nth) =  head.split_at_mut(start);
            self.v = tail;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&mut self.v[start..])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for ChunksMut<'a, T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for ChunksMut<'a, T> {}

//
// Free functions
//

/// Forms a slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `len` elements, nor whether the lifetime inferred is a suitable
/// lifetime for the returned slice.
///
/// `p` must be non-null, even for zero-length slices.
///
/// # Caveat
///
/// The lifetime for the returned slice is inferred from its usage. To
/// prevent accidental misuse, it's suggested to tie the lifetime to whichever
/// source lifetime is safe in the context, such as by providing a helper
/// function taking the lifetime of a host value for the slice, or by explicit
/// annotation.
///
/// # Examples
///
/// ```
/// use std::slice;
///
/// // manifest a slice out of thin air!
/// let ptr = 0x1234 as *const usize;
/// let amt = 10;
/// unsafe {
///     let slice = slice::from_raw_parts(ptr, amt);
/// }
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_raw_parts<'a, T>(p: *const T, len: usize) -> &'a [T] {
    mem::transmute(Repr { data: p, len: len })
}

/// Performs the same functionality as `from_raw_parts`, except that a mutable
/// slice is returned.
///
/// This function is unsafe for the same reasons as `from_raw_parts`, as well
/// as not being able to provide a non-aliasing guarantee of the returned
/// mutable slice.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_raw_parts_mut<'a, T>(p: *mut T, len: usize) -> &'a mut [T] {
    mem::transmute(Repr { data: p, len: len })
}

//
// Comparison traits
//

extern {
    /// Call implementation provided memcmp
    ///
    /// Interprets the data as u8.
    ///
    /// Return 0 for equal, < 0 for less than and > 0 for greater
    /// than.
    // FIXME(#32610): Return type should be c_int
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> PartialEq<[B]> for [A] where A: PartialEq<B> {
    fn eq(&self, other: &[B]) -> bool {
        SlicePartialEq::equal(self, other)
    }

    fn ne(&self, other: &[B]) -> bool {
        SlicePartialEq::not_equal(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for [T] {}

/// Implements comparison of vectors lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for [T] {
    fn cmp(&self, other: &[T]) -> Ordering {
        SliceOrd::compare(self, other)
    }
}

/// Implements comparison of vectors lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for [T] {
    fn partial_cmp(&self, other: &[T]) -> Option<Ordering> {
        SlicePartialOrd::partial_compare(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialEq
trait SlicePartialEq<B> {
    fn equal(&self, other: &[B]) -> bool;

    fn not_equal(&self, other: &[B]) -> bool { !self.equal(other) }
}

// Generic slice equality
impl<A, B> SlicePartialEq<B> for [A]
    where A: PartialEq<B>
{
    default fn equal(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for i in 0..self.len() {
            if !self[i].eq(&other[i]) {
                return false;
            }
        }

        true
    }
}

// Use memcmp for bytewise equality when the types allow
impl<A> SlicePartialEq<A> for [A]
    where A: PartialEq<A> + BytewiseEquality
{
    fn equal(&self, other: &[A]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.as_ptr() == other.as_ptr() {
            return true;
        }
        unsafe {
            let size = mem::size_of_val(self);
            memcmp(self.as_ptr() as *const u8,
                   other.as_ptr() as *const u8, size) == 0
        }
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialOrd
trait SlicePartialOrd<B> {
    fn partial_compare(&self, other: &[B]) -> Option<Ordering>;
}

impl<A> SlicePartialOrd<A> for [A]
    where A: PartialOrd
{
    default fn partial_compare(&self, other: &[A]) -> Option<Ordering> {
        let l = cmp::min(self.len(), other.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &self[..l];
        let rhs = &other[..l];

        for i in 0..l {
            match lhs[i].partial_cmp(&rhs[i]) {
                Some(Ordering::Equal) => (),
                non_eq => return non_eq,
            }
        }

        self.len().partial_cmp(&other.len())
    }
}

impl<A> SlicePartialOrd<A> for [A]
    where A: Ord
{
    default fn partial_compare(&self, other: &[A]) -> Option<Ordering> {
        Some(SliceOrd::compare(self, other))
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's Ord
trait SliceOrd<B> {
    fn compare(&self, other: &[B]) -> Ordering;
}

impl<A> SliceOrd<A> for [A]
    where A: Ord
{
    default fn compare(&self, other: &[A]) -> Ordering {
        let l = cmp::min(self.len(), other.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &self[..l];
        let rhs = &other[..l];

        for i in 0..l {
            match lhs[i].cmp(&rhs[i]) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            }
        }

        self.len().cmp(&other.len())
    }
}

// memcmp compares a sequence of unsigned bytes lexicographically.
// this matches the order we want for [u8], but no others (not even [i8]).
impl SliceOrd<u8> for [u8] {
    #[inline]
    fn compare(&self, other: &[u8]) -> Ordering {
        let order = unsafe {
            memcmp(self.as_ptr(), other.as_ptr(),
                   cmp::min(self.len(), other.len()))
        };
        if order == 0 {
            self.len().cmp(&other.len())
        } else if order < 0 {
            Less
        } else {
            Greater
        }
    }
}

#[doc(hidden)]
/// Trait implemented for types that can be compared for equality using
/// their bytewise representation
trait BytewiseEquality { }

macro_rules! impl_marker_for {
    ($traitname:ident, $($ty:ty)*) => {
        $(
            impl $traitname for $ty { }
        )*
    }
}

impl_marker_for!(BytewiseEquality,
                 u8 i8 u16 i16 u32 i32 u64 i64 usize isize char bool);

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Iter<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a T {
        &*self.ptr.offset(i as isize)
    }
    fn may_have_side_effect() -> bool { false }
}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for IterMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut T {
        &mut *self.ptr.offset(i as isize)
    }
    fn may_have_side_effect() -> bool { false }
}
