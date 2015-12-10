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
//! For more details `std::slice`.

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

use clone::Clone;
use cmp::{Ordering, PartialEq, PartialOrd, Eq, Ord};
use cmp::Ordering::{Less, Equal, Greater};
use cmp;
use default::Default;
use intrinsics::assume;
use iter::*;
use ops::{FnMut, self, Index};
use ops::RangeFull;
use option::Option;
use option::Option::{None, Some};
use result::Result;
use result::Result::{Ok, Err};
use ptr;
use mem;
use mem::size_of;
use marker::{Send, Sync, self};
use num::wrapping::OverflowingOps;
use raw::Repr;
// Avoid conflicts with *both* the Slice trait (buggy) and the `slice::raw` module.
use raw::Slice as RawSlice;


//
// Extension traits
//

/// Extension methods for slices.
#[unstable(feature = "core_slice_ext",
           reason = "stable interface provided by `impl [T]` in later crates",
           issue = "27701")]
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
    fn get(&self, index: usize) -> Option<&Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn first(&self) -> Option<&Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_first(&self) -> Option<(&Self::Item, &[Self::Item])>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_last(&self) -> Option<(&Self::Item, &[Self::Item])>;
    #[stable(feature = "core", since = "1.6.0")]
    fn last(&self) -> Option<&Self::Item>;
    #[stable(feature = "core", since = "1.6.0")]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_ptr(&self) -> *const Self::Item;
    #[stable(feature = "core", since = "1.6.0")]
    fn binary_search(&self, x: &Self::Item) -> Result<usize, usize>
        where Self::Item: Ord;
    #[stable(feature = "core", since = "1.6.0")]
    fn binary_search_by<F>(&self, f: F) -> Result<usize, usize>
        where F: FnMut(&Self::Item) -> Ordering;
    #[stable(feature = "core", since = "1.6.0")]
    fn len(&self) -> usize;
    #[stable(feature = "core", since = "1.6.0")]
    fn is_empty(&self) -> bool { self.len() == 0 }
    #[stable(feature = "core", since = "1.6.0")]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item>;
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
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_mut_ptr(&mut self) -> *mut Self::Item;

    #[stable(feature = "core", since = "1.6.0")]
    fn contains(&self, x: &Self::Item) -> bool where Self::Item: PartialEq;

    #[stable(feature = "core", since = "1.6.0")]
    fn starts_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    #[stable(feature = "core", since = "1.6.0")]
    fn ends_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    #[unstable(feature = "clone_from_slice", issue= "27750")]
    fn clone_from_slice(&mut self, &[Self::Item]) -> usize where Self::Item: Clone;
}

// Use macros to be generic over const/mut
macro_rules! slice_offset {
    ($ptr:expr, $by:expr) => {{
        let ptr = $ptr;
        if size_from_ptr(ptr) == 0 {
            ::intrinsics::arith_offset(ptr as *mut i8, $by) as *mut _
        } else {
            ptr.offset($by)
        }
    }};
}

macro_rules! slice_ref {
    ($ptr:expr) => {{
        let ptr = $ptr;
        if size_from_ptr(ptr) == 0 {
            // Use a non-null pointer value
            &mut *(1 as *mut _)
        } else {
            mem::transmute(ptr)
        }
    }};
}

#[unstable(feature = "core_slice_ext",
           reason = "stable interface provided by `impl [T]` in later crates",
           issue = "27701")]
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
    fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() { Some(&self[index]) } else { None }
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
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*(self.repr().data.offset(index as isize))
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.repr().data
    }

    fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize> where
        F: FnMut(&T) -> Ordering
    {
        let mut base : usize = 0;
        let mut lim : usize = self.len();

        while lim != 0 {
            let ix = base + (lim >> 1);
            match f(&self[ix]) {
                Equal => return Ok(ix),
                Less => {
                    base = ix + 1;
                    lim -= 1;
                }
                Greater => ()
            }
            lim >>= 1;
        }
        Err(base)
    }

    #[inline]
    fn len(&self) -> usize { self.repr().len }

    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() { Some(&mut self[index]) } else { None }
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
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *(self.repr().data as *mut T).offset(index as isize)
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.repr().data as *mut T
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

    fn binary_search(&self, x: &T) -> Result<usize, usize> where T: Ord {
        self.binary_search_by(|p| p.cmp(x))
    }

    #[inline]
    fn clone_from_slice(&mut self, src: &[T]) -> usize where T: Clone {
        let min = cmp::min(self.len(), src.len());
        let dst = &mut self[.. min];
        let src = &src[.. min];
        for i in 0..min {
            dst[i].clone_from(&src[i]);
        }
        min
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<usize> for [T] {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        assert!(index < self.len());
        unsafe { self.get_unchecked(index) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<usize> for [T] {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len());
        unsafe { self.get_unchecked_mut(index) }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::Range<usize>> for [T] {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &[T] {
        if index.start > index.end {
            slice_index_order_fail(index.start, index.end);
        } else if index.end > self.len() {
            slice_index_len_fail(index.end, self.len());
        }
        unsafe {
            from_raw_parts (
                self.as_ptr().offset(index.start as isize),
                index.end - index.start
            )
        }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeTo<usize>> for [T] {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &[T] {
        self.index(ops::Range{ start: 0, end: index.end })
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeFrom<usize>> for [T] {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &[T] {
        self.index(ops::Range{ start: index.start, end: self.len() })
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<RangeFull> for [T] {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::Range<usize>> for [T] {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [T] {
        if index.start > index.end {
            slice_index_order_fail(index.start, index.end);
        } else if index.end > self.len() {
            slice_index_len_fail(index.end, self.len());
        }
        unsafe {
            from_raw_parts_mut(
                self.as_mut_ptr().offset(index.start as isize),
                index.end - index.start
            )
        }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeTo<usize>> for [T] {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [T] {
        self.index_mut(ops::Range{ start: 0, end: index.end })
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeFrom<usize>> for [T] {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [T] {
        let len = self.len();
        self.index_mut(ops::Range{ start: index.start, end: len })
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<RangeFull> for [T] {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        self
    }
}


////////////////////////////////////////////////////////////////////////////////
// Common traits
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Default for &'a [T] {
    fn default() -> &'a [T] { &[] }
}

#[stable(feature = "mut_slice_default", since = "1.5.0")]
impl<'a, T> Default for &'a mut [T] {
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
    (struct $name:ident -> $ptr:ty, $elem:ty) => {
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
                        let old = self.ptr;
                        self.ptr = slice_offset!(self.ptr, 1);
                        Some(slice_ref!(old))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let diff = (self.end as usize).wrapping_sub(self.ptr as usize);
                let size = mem::size_of::<T>();
                let exact = diff / (if size == 0 {1} else {size});
                (exact, Some(exact))
            }

            #[inline]
            fn count(self) -> usize {
                self.size_hint().0
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
                        self.end = slice_offset!(self.end, -1);
                        Some(slice_ref!(self.end))
                    }
                }
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

/// Immutable slice iterator
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _marker: marker::PhantomData<&'a T>,
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

iterator!{struct Iter -> *const T, &'a T}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> { Iter { ptr: self.ptr, end: self.end, _marker: self._marker } }
}

/// Mutable slice iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    ptr: *mut T,
    end: *mut T,
    _marker: marker::PhantomData<&'a mut T>,
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

iterator!{struct IterMut -> *mut T, &'a mut T}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}

/// An internal abstraction over the splitting iterators, so that
/// splitn, splitn_mut etc can be implemented once.
trait SplitIter: DoubleEndedIterator {
    /// Mark the underlying iterator as complete, extracting the remaining
    /// portion of the slice.
    fn finish(&mut self) -> Option<Self::Item>;
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Split<'a, T:'a, P> where P: FnMut(&T) -> bool {
    v: &'a [T],
    pred: P,
    finished: bool
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

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitMut<'a, T:'a, P> where P: FnMut(&T) -> bool {
    v: &'a mut [T],
    pred: P,
    finished: bool
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

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
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
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitN<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<Split<'a, T, P>>
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitN<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<Split<'a, T, P>>
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitNMut<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<SplitMut<'a, T, P>>
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitNMut<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<SplitMut<'a, T, P>>
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
    }
}

forward_iterator! { SplitN: T, &'a [T] }
forward_iterator! { RSplitN: T, &'a [T] }
forward_iterator! { SplitNMut: T, &'a mut [T] }
forward_iterator! { RSplitNMut: T, &'a mut [T] }

/// An iterator over overlapping subslices of length `size`.
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
        self.size_hint().0
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

/// An iterator over a slice in (non-overlapping) chunks (`size` elements at a
/// time).
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
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
        self.size_hint().0
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

/// An iterator over a slice in (non-overlapping) mutable chunks (`size`
/// elements at a time). When the slice len is not evenly divided by the chunk
/// size, the last slice of the iteration will be the remainder.
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
        self.size_hint().0
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
    mem::transmute(RawSlice { data: p, len: len })
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
    mem::transmute(RawSlice { data: p, len: len })
}

//
// Submodules
//

/// Operations on `[u8]`.
#[unstable(feature = "slice_bytes", reason = "needs review",
           issue = "27740")]
#[rustc_deprecated(reason = "unidiomatic functions not pulling their weight",
                   since = "1.6.0")]
#[allow(deprecated)]
pub mod bytes {
    use ptr;
    use slice::SliceExt;

    /// A trait for operations on mutable `[u8]`s.
    pub trait MutableByteVector {
        /// Sets all bytes of the receiver to the given value.
        fn set_memory(&mut self, value: u8);
    }

    impl MutableByteVector for [u8] {
        #[inline]
        fn set_memory(&mut self, value: u8) {
            unsafe { ptr::write_bytes(self.as_mut_ptr(), value, self.len()) };
        }
    }

    /// Copies data from `src` to `dst`
    ///
    /// Panics if the length of `dst` is less than the length of `src`.
    #[inline]
    pub fn copy_memory(src: &[u8], dst: &mut [u8]) {
        let len_src = src.len();
        assert!(dst.len() >= len_src);
        // `dst` is unaliasable, so we know statically it doesn't overlap
        // with `src`.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(),
                                     dst.as_mut_ptr(),
                                     len_src);
        }
    }
}



//
// Boilerplate traits
//

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> PartialEq<[B]> for [A] where A: PartialEq<B> {
    fn eq(&self, other: &[B]) -> bool {
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
    fn ne(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return true;
        }

        for i in 0..self.len() {
            if self[i].ne(&other[i]) {
                return true;
            }
        }

        false
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for [T] {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for [T] {
    fn cmp(&self, other: &[T]) -> Ordering {
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for [T] {
    fn partial_cmp(&self, other: &[T]) -> Option<Ordering> {
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
