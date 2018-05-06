// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
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
// on them are defined on traits, which are then re-exported from
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

use cmp::Ordering::{self, Less, Equal, Greater};
use cmp;
use fmt;
use intrinsics::assume;
use iter::*;
use ops::{FnMut, Try, self};
use option::Option;
use option::Option::{None, Some};
use result::Result;
use result::Result::{Ok, Err};
use ptr;
use mem;
use marker::{Copy, Send, Sync, Sized, self};
use iter_private::TrustedRandomAccess;

#[unstable(feature = "slice_internals", issue = "0",
           reason = "exposed from core to be reused in std; use the memchr crate")]
/// Pure rust memchr implementation, taken from rust-memchr
pub mod memchr;

mod rotate;
mod sort;

#[repr(C)]
struct Repr<T> {
    pub data: *const T,
    pub len: usize,
}

//
// Extension traits
//

public_in_stage0! {
{
/// Extension methods for slices.
#[unstable(feature = "core_slice_ext",
           reason = "stable interface provided by `impl [T]` in later crates",
           issue = "32110")]
#[allow(missing_docs)] // documented elsewhere
}
trait SliceExt {
    type Item;

    #[stable(feature = "core", since = "1.6.0")]
    fn split_at(&self, mid: usize) -> (&[Self::Item], &[Self::Item]);

    #[stable(feature = "core", since = "1.6.0")]
    fn iter(&self) -> Iter<Self::Item>;

    #[stable(feature = "core", since = "1.6.0")]
    fn split<P>(&self, pred: P) -> Split<Self::Item, P>
        where P: FnMut(&Self::Item) -> bool;

    #[stable(feature = "slice_rsplit", since = "1.27.0")]
    fn rsplit<P>(&self, pred: P) -> RSplit<Self::Item, P>
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

    #[unstable(feature = "exact_chunks", issue = "47115")]
    fn exact_chunks(&self, size: usize) -> ExactChunks<Self::Item>;

    #[stable(feature = "core", since = "1.6.0")]
    fn get<I>(&self, index: I) -> Option<&I::Output>
        where I: SliceIndex<Self>;
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
        where I: SliceIndex<Self>;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_ptr(&self) -> *const Self::Item;

    #[stable(feature = "core", since = "1.6.0")]
    fn binary_search(&self, x: &Self::Item) -> Result<usize, usize>
        where Self::Item: Ord;

    #[stable(feature = "core", since = "1.6.0")]
    fn binary_search_by<'a, F>(&'a self, f: F) -> Result<usize, usize>
        where F: FnMut(&'a Self::Item) -> Ordering;

    #[stable(feature = "slice_binary_search_by_key", since = "1.10.0")]
    fn binary_search_by_key<'a, B, F>(&'a self, b: &B, f: F) -> Result<usize, usize>
        where F: FnMut(&'a Self::Item) -> B,
              B: Ord;

    #[stable(feature = "core", since = "1.6.0")]
    fn len(&self) -> usize;

    #[stable(feature = "core", since = "1.6.0")]
    fn is_empty(&self) -> bool { self.len() == 0 }

    #[stable(feature = "core", since = "1.6.0")]
    fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
        where I: SliceIndex<Self>;
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

    #[stable(feature = "slice_rsplit", since = "1.27.0")]
    fn rsplit_mut<P>(&mut self, pred: P) -> RSplitMut<Self::Item, P>
        where P: FnMut(&Self::Item) -> bool;

    #[stable(feature = "core", since = "1.6.0")]
    fn splitn_mut<P>(&mut self, n: usize, pred: P) -> SplitNMut<Self::Item, P>
        where P: FnMut(&Self::Item) -> bool;

    #[stable(feature = "core", since = "1.6.0")]
    fn rsplitn_mut<P>(&mut self,  n: usize, pred: P) -> RSplitNMut<Self::Item, P>
        where P: FnMut(&Self::Item) -> bool;

    #[stable(feature = "core", since = "1.6.0")]
    fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<Self::Item>;

    #[unstable(feature = "exact_chunks", issue = "47115")]
    fn exact_chunks_mut(&mut self, size: usize) -> ExactChunksMut<Self::Item>;

    #[stable(feature = "core", since = "1.6.0")]
    fn swap(&mut self, a: usize, b: usize);

    #[stable(feature = "core", since = "1.6.0")]
    fn split_at_mut(&mut self, mid: usize) -> (&mut [Self::Item], &mut [Self::Item]);

    #[stable(feature = "core", since = "1.6.0")]
    fn reverse(&mut self);

    #[stable(feature = "core", since = "1.6.0")]
    unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
        where I: SliceIndex<Self>;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_mut_ptr(&mut self) -> *mut Self::Item;

    #[stable(feature = "core", since = "1.6.0")]
    fn contains(&self, x: &Self::Item) -> bool where Self::Item: PartialEq;

    #[stable(feature = "core", since = "1.6.0")]
    fn starts_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    #[stable(feature = "core", since = "1.6.0")]
    fn ends_with(&self, needle: &[Self::Item]) -> bool where Self::Item: PartialEq;

    #[stable(feature = "slice_rotate", since = "1.26.0")]
    fn rotate_left(&mut self, mid: usize);

    #[stable(feature = "slice_rotate", since = "1.26.0")]
    fn rotate_right(&mut self, k: usize);

    #[stable(feature = "clone_from_slice", since = "1.7.0")]
    fn clone_from_slice(&mut self, src: &[Self::Item]) where Self::Item: Clone;

    #[stable(feature = "copy_from_slice", since = "1.9.0")]
    fn copy_from_slice(&mut self, src: &[Self::Item]) where Self::Item: Copy;

    #[stable(feature = "swap_with_slice", since = "1.27.0")]
    fn swap_with_slice(&mut self, src: &mut [Self::Item]);

    #[stable(feature = "sort_unstable", since = "1.20.0")]
    fn sort_unstable(&mut self)
        where Self::Item: Ord;

    #[stable(feature = "sort_unstable", since = "1.20.0")]
    fn sort_unstable_by<F>(&mut self, compare: F)
        where F: FnMut(&Self::Item, &Self::Item) -> Ordering;

    #[stable(feature = "sort_unstable", since = "1.20.0")]
    fn sort_unstable_by_key<B, F>(&mut self, f: F)
        where F: FnMut(&Self::Item) -> B,
              B: Ord;
}}

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
    fn split<P>(&self, pred: P) -> Split<T, P>
        where P: FnMut(&T) -> bool
    {
        Split {
            v: self,
            pred,
            finished: false
        }
    }

    #[inline]
    fn rsplit<P>(&self, pred: P) -> RSplit<T, P>
        where P: FnMut(&T) -> bool
    {
        RSplit { inner: self.split(pred) }
    }

    #[inline]
    fn splitn<P>(&self, n: usize, pred: P) -> SplitN<T, P>
        where P: FnMut(&T) -> bool
    {
        SplitN {
            inner: GenericSplitN {
                iter: self.split(pred),
                count: n
            }
        }
    }

    #[inline]
    fn rsplitn<P>(&self, n: usize, pred: P) -> RSplitN<T, P>
        where P: FnMut(&T) -> bool
    {
        RSplitN {
            inner: GenericSplitN {
                iter: self.rsplit(pred),
                count: n
            }
        }
    }

    #[inline]
    fn windows(&self, size: usize) -> Windows<T> {
        assert!(size != 0);
        Windows { v: self, size: size }
    }

    #[inline]
    fn chunks(&self, chunk_size: usize) -> Chunks<T> {
        assert!(chunk_size != 0);
        Chunks { v: self, chunk_size: chunk_size }
    }

    #[inline]
    fn exact_chunks(&self, chunk_size: usize) -> ExactChunks<T> {
        assert!(chunk_size != 0);
        let rem = self.len() % chunk_size;
        let len = self.len() - rem;
        ExactChunks { v: &self[..len], chunk_size: chunk_size}
    }

    #[inline]
    fn get<I>(&self, index: I) -> Option<&I::Output>
        where I: SliceIndex<[T]>
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
        where I: SliceIndex<[T]>
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
        let s = self;
        let mut size = s.len();
        if size == 0 {
            return Err(0);
        }
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // mid is always in [0, size), that means mid is >= 0 and < size.
            // mid >= 0: by definition
            // mid < size: mid = size / 2 + size / 4 + size / 8 ...
            let cmp = f(unsafe { s.get_unchecked(mid) });
            base = if cmp == Greater { base } else { mid };
            size -= half;
        }
        // base is always in [0, size) because base <= mid.
        let cmp = f(unsafe { s.get_unchecked(base) });
        if cmp == Equal { Ok(base) } else { Err(base + (cmp == Less) as usize) }
    }

    #[inline]
    fn len(&self) -> usize {
        unsafe {
            mem::transmute::<&[T], Repr<T>>(self).len
        }
    }

    #[inline]
    fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
        where I: SliceIndex<[T]>
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
    fn split_mut<P>(&mut self, pred: P) -> SplitMut<T, P>
        where P: FnMut(&T) -> bool
    {
        SplitMut { v: self, pred: pred, finished: false }
    }

    #[inline]
    fn rsplit_mut<P>(&mut self, pred: P) -> RSplitMut<T, P>
        where P: FnMut(&T) -> bool
    {
        RSplitMut { inner: self.split_mut(pred) }
    }

    #[inline]
    fn splitn_mut<P>(&mut self, n: usize, pred: P) -> SplitNMut<T, P>
        where P: FnMut(&T) -> bool
    {
        SplitNMut {
            inner: GenericSplitN {
                iter: self.split_mut(pred),
                count: n
            }
        }
    }

    #[inline]
    fn rsplitn_mut<P>(&mut self, n: usize, pred: P) -> RSplitNMut<T, P> where
        P: FnMut(&T) -> bool,
    {
        RSplitNMut {
            inner: GenericSplitN {
                iter: self.rsplit_mut(pred),
                count: n
            }
        }
    }

    #[inline]
    fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        assert!(chunk_size != 0);
        ChunksMut { v: self, chunk_size: chunk_size }
    }

    #[inline]
    fn exact_chunks_mut(&mut self, chunk_size: usize) -> ExactChunksMut<T> {
        assert!(chunk_size != 0);
        let rem = self.len() % chunk_size;
        let len = self.len() - rem;
        ExactChunksMut { v: &mut self[..len], chunk_size: chunk_size}
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

        // For very small types, all the individual reads in the normal
        // path perform poorly.  We can do better, given efficient unaligned
        // load/store, by loading a larger chunk and reversing a register.

        // Ideally LLVM would do this for us, as it knows better than we do
        // whether unaligned reads are efficient (since that changes between
        // different ARM versions, for example) and what the best chunk size
        // would be.  Unfortunately, as of LLVM 4.0 (2017-05) it only unrolls
        // the loop, so we need to do this ourselves.  (Hypothesis: reverse
        // is troublesome because the sides can be aligned differently --
        // will be, when the length is odd -- so there's no way of emitting
        // pre- and postludes to use fully-aligned SIMD in the middle.)

        let fast_unaligned =
            cfg!(any(target_arch = "x86", target_arch = "x86_64"));

        if fast_unaligned && mem::size_of::<T>() == 1 {
            // Use the llvm.bswap intrinsic to reverse u8s in a usize
            let chunk = mem::size_of::<usize>();
            while i + chunk - 1 < ln / 2 {
                unsafe {
                    let pa: *mut T = self.get_unchecked_mut(i);
                    let pb: *mut T = self.get_unchecked_mut(ln - i - chunk);
                    let va = ptr::read_unaligned(pa as *mut usize);
                    let vb = ptr::read_unaligned(pb as *mut usize);
                    ptr::write_unaligned(pa as *mut usize, vb.swap_bytes());
                    ptr::write_unaligned(pb as *mut usize, va.swap_bytes());
                }
                i += chunk;
            }
        }

        if fast_unaligned && mem::size_of::<T>() == 2 {
            // Use rotate-by-16 to reverse u16s in a u32
            let chunk = mem::size_of::<u32>() / 2;
            while i + chunk - 1 < ln / 2 {
                unsafe {
                    let pa: *mut T = self.get_unchecked_mut(i);
                    let pb: *mut T = self.get_unchecked_mut(ln - i - chunk);
                    let va = ptr::read_unaligned(pa as *mut u32);
                    let vb = ptr::read_unaligned(pb as *mut u32);
                    ptr::write_unaligned(pa as *mut u32, vb.rotate_left(16));
                    ptr::write_unaligned(pb as *mut u32, va.rotate_left(16));
                }
                i += chunk;
            }
        }

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
        where I: SliceIndex<[T]>
    {
        index.get_unchecked_mut(self)
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut [T] as *mut T
    }

    #[inline]
    fn contains(&self, x: &T) -> bool where T: PartialEq {
        x.slice_contains(self)
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

    fn binary_search(&self, x: &T) -> Result<usize, usize>
        where T: Ord
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    fn rotate_left(&mut self, mid: usize) {
        assert!(mid <= self.len());
        let k = self.len() - mid;

        unsafe {
            let p = self.as_mut_ptr();
            rotate::ptr_rotate(mid, p.offset(mid as isize), k);
        }
    }

    fn rotate_right(&mut self, k: usize) {
        assert!(k <= self.len());
        let mid = self.len() - k;

        unsafe {
            let p = self.as_mut_ptr();
            rotate::ptr_rotate(mid, p.offset(mid as isize), k);
        }
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
    fn swap_with_slice(&mut self, src: &mut [T]) {
        assert!(self.len() == src.len(),
                "destination and source slices have different lengths");
        unsafe {
            ptr::swap_nonoverlapping(
                self.as_mut_ptr(), src.as_mut_ptr(), self.len());
        }
    }

    #[inline]
    fn binary_search_by_key<'a, B, F>(&'a self, b: &B, mut f: F) -> Result<usize, usize>
        where F: FnMut(&'a Self::Item) -> B,
              B: Ord
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    #[inline]
    fn sort_unstable(&mut self)
        where Self::Item: Ord
    {
        sort::quicksort(self, |a, b| a.lt(b));
    }

    #[inline]
    fn sort_unstable_by<F>(&mut self, mut compare: F)
        where F: FnMut(&Self::Item, &Self::Item) -> Ordering
    {
        sort::quicksort(self, |a, b| compare(a, b) == Ordering::Less);
    }

    #[inline]
    fn sort_unstable_by_key<B, F>(&mut self, mut f: F)
        where F: FnMut(&Self::Item) -> B,
              B: Ord
    {
        sort::quicksort(self, |a, b| f(a).lt(&f(b)));
    }
}

// FIXME: remove (inline) this macro and the SliceExt trait
// when updating to a bootstrap compiler that has the new lang items.
#[cfg_attr(stage0, macro_export)]
#[unstable(feature = "core_slice_ext", issue = "32110")]
macro_rules! slice_core_methods { () => {
    /// Returns the number of elements in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len(&self) -> usize {
        SliceExt::len(self)
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert!(!a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_empty(&self) -> bool {
        SliceExt::is_empty(self)
    }

    /// Returns the first element of the slice, or `None` if it is empty.
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
        SliceExt::first(self)
    }

    /// Returns a mutable pointer to the first element of the slice, or `None` if it is empty.
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
        SliceExt::first_mut(self)
    }

    /// Returns the first and all the rest of the elements of the slice, or `None` if it is empty.
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
        SliceExt::split_first(self)
    }

    /// Returns the first and all the rest of the elements of the slice, or `None` if it is empty.
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
        SliceExt::split_first_mut(self)
    }

    /// Returns the last and all the rest of the elements of the slice, or `None` if it is empty.
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
        SliceExt::split_last(self)
    }

    /// Returns the last and all the rest of the elements of the slice, or `None` if it is empty.
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
        SliceExt::split_last_mut(self)
    }

    /// Returns the last element of the slice, or `None` if it is empty.
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
        SliceExt::last(self)
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
        SliceExt::last_mut(self)
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
        where I: SliceIndex<Self>
    {
        SliceExt::get(self, index)
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index (see [`get`]) or `None` if the index is out of bounds.
    ///
    /// [`get`]: #method.get
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
        where I: SliceIndex<Self>
    {
        SliceExt::get_mut(self, index)
    }

    /// Returns a reference to an element or subslice, without doing bounds
    /// checking.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`get`].
    ///
    /// [`get`]: #method.get
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
        where I: SliceIndex<Self>
    {
        SliceExt::get_unchecked(self, index)
    }

    /// Returns a mutable reference to an element or subslice, without doing
    /// bounds checking.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`get_mut`].
    ///
    /// [`get_mut`]: #method.get_mut
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
        where I: SliceIndex<Self>
    {
        SliceExt::get_unchecked_mut(self, index)
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the container referenced by this slice may cause its buffer
    /// to be reallocated, which would also make any pointers to it invalid.
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
        SliceExt::as_ptr(self)
    }

    /// Returns an unsafe mutable pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the container referenced by this slice may cause its buffer
    /// to be reallocated, which would also make any pointers to it invalid.
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
        SliceExt::as_mut_ptr(self)
    }

    /// Swaps two elements in the slice.
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
        SliceExt::swap(self, a, b)
    }

    /// Reverses the order of elements in the slice, in place.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 2, 3];
    /// v.reverse();
    /// assert!(v == [3, 2, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn reverse(&mut self) {
        SliceExt::reverse(self)
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
        SliceExt::iter(self)
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
        SliceExt::iter_mut(self)
    }

    /// Returns an iterator over all contiguous windows of length
    /// `size`. The windows overlap. If the slice is shorter than
    /// `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Examples
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
        SliceExt::windows(self, size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a
    /// time. The chunks are slices and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last chunk will
    /// not have length `chunk_size`.
    ///
    /// See [`exact_chunks`] for a variant of this iterator that returns chunks
    /// of always exactly `chunk_size` elements.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.chunks(2);
    /// assert_eq!(iter.next().unwrap(), &['l', 'o']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e']);
    /// assert_eq!(iter.next().unwrap(), &['m']);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// [`exact_chunks`]: #method.exact_chunks
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<T> {
        SliceExt::chunks(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a
    /// time. The chunks are slices and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last up to `chunk_size-1`
    /// elements will be omitted.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler
    /// can often optimize the resulting code better than in the case of
    /// [`chunks`].
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(exact_chunks)]
    ///
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.exact_chunks(2);
    /// assert_eq!(iter.next().unwrap(), &['l', 'o']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e']);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// [`chunks`]: #method.chunks
    #[unstable(feature = "exact_chunks", issue = "47115")]
    #[inline]
    pub fn exact_chunks(&self, chunk_size: usize) -> ExactChunks<T> {
        SliceExt::exact_chunks(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last chunk will not
    /// have length `chunk_size`.
    ///
    /// See [`exact_chunks_mut`] for a variant of this iterator that returns chunks
    /// of always exactly `chunk_size` elements.
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
    ///
    /// [`exact_chunks_mut`]: #method.exact_chunks_mut
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        SliceExt::chunks_mut(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last up to `chunk_size-1`
    /// elements will be omitted.
    ///
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler
    /// can often optimize the resulting code better than in the case of
    /// [`chunks_mut`].
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(exact_chunks)]
    ///
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.exact_chunks_mut(2) {
    ///     for elem in chunk.iter_mut() {
    ///         *elem += count;
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[1, 1, 2, 2, 0]);
    /// ```
    ///
    /// [`chunks_mut`]: #method.chunks_mut
    #[unstable(feature = "exact_chunks", issue = "47115")]
    #[inline]
    pub fn exact_chunks_mut(&mut self, chunk_size: usize) -> ExactChunksMut<T> {
        SliceExt::exact_chunks_mut(self, chunk_size)
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
    /// let v = [1, 2, 3, 4, 5, 6];
    ///
    /// {
    ///    let (left, right) = v.split_at(0);
    ///    assert!(left == []);
    ///    assert!(right == [1, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at(2);
    ///     assert!(left == [1, 2]);
    ///     assert!(right == [3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at(6);
    ///     assert!(left == [1, 2, 3, 4, 5, 6]);
    ///     assert!(right == []);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        SliceExt::split_at(self, mid)
    }

    /// Divides one mutable slice into two at an index.
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
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// // scoped to restrict the lifetime of the borrows
    /// {
    ///     let (left, right) = v.split_at_mut(2);
    ///     assert!(left == [1, 0]);
    ///     assert!(right == [3, 0, 5, 6]);
    ///     left[1] = 2;
    ///     right[1] = 4;
    /// }
    /// assert!(v == [1, 2, 3, 4, 5, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        SliceExt::split_at_mut(self, mid)
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
        SliceExt::split(self, pred)
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
        SliceExt::split_mut(self, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, starting at the end of the slice and working backwards.
    /// The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = [11, 22, 33, 0, 44, 55];
    /// let mut iter = slice.rsplit(|num| *num == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[44, 55]);
    /// assert_eq!(iter.next().unwrap(), &[11, 22, 33]);
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// As with `split()`, if the first or last element is matched, an empty
    /// slice will be the first (or last) item returned by the iterator.
    ///
    /// ```
    /// let v = &[0, 1, 1, 2, 3, 5, 8];
    /// let mut it = v.rsplit(|n| *n % 2 == 0);
    /// assert_eq!(it.next().unwrap(), &[]);
    /// assert_eq!(it.next().unwrap(), &[3, 5]);
    /// assert_eq!(it.next().unwrap(), &[1, 1]);
    /// assert_eq!(it.next().unwrap(), &[]);
    /// assert_eq!(it.next(), None);
    /// ```
    #[stable(feature = "slice_rsplit", since = "1.27.0")]
    #[inline]
    pub fn rsplit<F>(&self, pred: F) -> RSplit<T, F>
        where F: FnMut(&T) -> bool
    {
        SliceExt::rsplit(self, pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`, starting at the end of the slice and working
    /// backwards. The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [100, 400, 300, 200, 600, 500];
    ///
    /// let mut count = 0;
    /// for group in v.rsplit_mut(|num| *num % 3 == 0) {
    ///     count += 1;
    ///     group[0] = count;
    /// }
    /// assert_eq!(v, [3, 400, 300, 2, 600, 1]);
    /// ```
    ///
    #[stable(feature = "slice_rsplit", since = "1.27.0")]
    #[inline]
    pub fn rsplit_mut<F>(&mut self, pred: F) -> RSplitMut<T, F>
        where F: FnMut(&T) -> bool
    {
        SliceExt::rsplit_mut(self, pred)
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
        SliceExt::splitn(self, n, pred)
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
        SliceExt::splitn_mut(self, n, pred)
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
        SliceExt::rsplitn(self, n, pred)
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
        SliceExt::rsplitn_mut(self, n, pred)
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
        SliceExt::contains(self, x)
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
        SliceExt::starts_with(self, needle)
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
        SliceExt::ends_with(self, needle)
    }

    /// Binary searches this sorted slice for a given element.
    ///
    /// If the value is found then `Ok` is returned, containing the
    /// index of the matching element; if the value is not found then
    /// `Err` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// # Examples
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
        SliceExt::binary_search(self, x)
    }

    /// Binary searches this sorted slice with a comparator function.
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
    /// # Examples
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
        SliceExt::binary_search_by(self, f)
    }

    /// Binary searches this sorted slice with a key extraction function.
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
        SliceExt::binary_search_by_key(self, b, f)
    }

    /// Sorts the slice, but may not preserve the order of equal elements.
    ///
    /// This sort is unstable (i.e. may reorder equal elements), in-place (i.e. does not allocate),
    /// and `O(n log n)` worst-case.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g. when the
    /// slice consists of several concatenated sorted sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5, 4, 1, -3, 2];
    ///
    /// v.sort_unstable();
    /// assert!(v == [-5, -3, 1, 2, 4]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable(&mut self)
        where T: Ord
    {
        SliceExt::sort_unstable(self);
    }

    /// Sorts the slice with a comparator function, but may not preserve the order of equal
    /// elements.
    ///
    /// This sort is unstable (i.e. may reorder equal elements), in-place (i.e. does not allocate),
    /// and `O(n log n)` worst-case.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g. when the
    /// slice consists of several concatenated sorted sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [5, 4, 1, 3, 2];
    /// v.sort_unstable_by(|a, b| a.cmp(b));
    /// assert!(v == [1, 2, 3, 4, 5]);
    ///
    /// // reverse sorting
    /// v.sort_unstable_by(|a, b| b.cmp(a));
    /// assert!(v == [5, 4, 3, 2, 1]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, compare: F)
        where F: FnMut(&T, &T) -> Ordering
    {
        SliceExt::sort_unstable_by(self, compare);
    }

    /// Sorts the slice with a key extraction function, but may not preserve the order of equal
    /// elements.
    ///
    /// This sort is unstable (i.e. may reorder equal elements), in-place (i.e. does not allocate),
    /// and `O(m n log(m n))` worst-case, where the key function is `O(m)`.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 1, -3, 2];
    ///
    /// v.sort_unstable_by_key(|k| k.abs());
    /// assert!(v == [1, 2, -3, 4, -5]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, f: F)
        where F: FnMut(&T) -> K, K: Ord
    {
        SliceExt::sort_unstable_by_key(self, f);
    }

    /// Rotates the slice in-place such that the first `mid` elements of the
    /// slice move to the end while the last `self.len() - mid` elements move to
    /// the front. After calling `rotate_left`, the element previously at index
    /// `mid` will become the first element in the slice.
    ///
    /// # Panics
    ///
    /// This function will panic if `mid` is greater than the length of the
    /// slice. Note that `mid == self.len()` does _not_ panic and is a no-op
    /// rotation.
    ///
    /// # Complexity
    ///
    /// Takes linear (in `self.len()`) time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a.rotate_left(2);
    /// assert_eq!(a, ['c', 'd', 'e', 'f', 'a', 'b']);
    /// ```
    ///
    /// Rotating a subslice:
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a[1..5].rotate_left(1);
    /// assert_eq!(a, ['a', 'c', 'd', 'e', 'b', 'f']);
   /// ```
    #[stable(feature = "slice_rotate", since = "1.26.0")]
    pub fn rotate_left(&mut self, mid: usize) {
        SliceExt::rotate_left(self, mid);
    }

    /// Rotates the slice in-place such that the first `self.len() - k`
    /// elements of the slice move to the end while the last `k` elements move
    /// to the front. After calling `rotate_right`, the element previously at
    /// index `self.len() - k` will become the first element in the slice.
    ///
    /// # Panics
    ///
    /// This function will panic if `k` is greater than the length of the
    /// slice. Note that `k == self.len()` does _not_ panic and is a no-op
    /// rotation.
    ///
    /// # Complexity
    ///
    /// Takes linear (in `self.len()`) time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a.rotate_right(2);
    /// assert_eq!(a, ['e', 'f', 'a', 'b', 'c', 'd']);
    /// ```
    ///
    /// Rotate a subslice:
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a[1..5].rotate_right(1);
    /// assert_eq!(a, ['a', 'e', 'b', 'c', 'd', 'f']);
    /// ```
    #[stable(feature = "slice_rotate", since = "1.26.0")]
    pub fn rotate_right(&mut self, k: usize) {
        SliceExt::rotate_right(self, k);
    }

    /// Copies the elements from `src` into `self`.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// If `src` implements `Copy`, it can be more performant to use
    /// [`copy_from_slice`].
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// Cloning two elements from a slice into another:
    ///
    /// ```
    /// let src = [1, 2, 3, 4];
    /// let mut dst = [0, 0];
    ///
    /// dst.clone_from_slice(&src[2..]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(dst, [3, 4]);
    /// ```
    ///
    /// Rust enforces that there can only be one mutable reference with no
    /// immutable references to a particular piece of data in a particular
    /// scope. Because of this, attempting to use `clone_from_slice` on a
    /// single slice will result in a compile failure:
    ///
    /// ```compile_fail
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// slice[..2].clone_from_slice(&slice[3..]); // compile fail!
    /// ```
    ///
    /// To work around this, we can use [`split_at_mut`] to create two distinct
    /// sub-slices from a slice:
    ///
    /// ```
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// {
    ///     let (left, right) = slice.split_at_mut(2);
    ///     left.clone_from_slice(&right[1..]);
    /// }
    ///
    /// assert_eq!(slice, [4, 5, 3, 4, 5]);
    /// ```
    ///
    /// [`copy_from_slice`]: #method.copy_from_slice
    /// [`split_at_mut`]: #method.split_at_mut
    #[stable(feature = "clone_from_slice", since = "1.7.0")]
    pub fn clone_from_slice(&mut self, src: &[T]) where T: Clone {
        SliceExt::clone_from_slice(self, src)
    }

    /// Copies all elements from `src` into `self`, using a memcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// If `src` does not implement `Copy`, use [`clone_from_slice`].
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// Copying two elements from a slice into another:
    ///
    /// ```
    /// let src = [1, 2, 3, 4];
    /// let mut dst = [0, 0];
    ///
    /// dst.copy_from_slice(&src[2..]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(dst, [3, 4]);
    /// ```
    ///
    /// Rust enforces that there can only be one mutable reference with no
    /// immutable references to a particular piece of data in a particular
    /// scope. Because of this, attempting to use `copy_from_slice` on a
    /// single slice will result in a compile failure:
    ///
    /// ```compile_fail
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// slice[..2].copy_from_slice(&slice[3..]); // compile fail!
    /// ```
    ///
    /// To work around this, we can use [`split_at_mut`] to create two distinct
    /// sub-slices from a slice:
    ///
    /// ```
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// {
    ///     let (left, right) = slice.split_at_mut(2);
    ///     left.copy_from_slice(&right[1..]);
    /// }
    ///
    /// assert_eq!(slice, [4, 5, 3, 4, 5]);
    /// ```
    ///
    /// [`clone_from_slice`]: #method.clone_from_slice
    /// [`split_at_mut`]: #method.split_at_mut
    #[stable(feature = "copy_from_slice", since = "1.9.0")]
    pub fn copy_from_slice(&mut self, src: &[T]) where T: Copy {
        SliceExt::copy_from_slice(self, src)
    }

    /// Swaps all elements in `self` with those in `other`.
    ///
    /// The length of `other` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Example
    ///
    /// Swapping two elements across slices:
    ///
    /// ```
    /// let mut slice1 = [0, 0];
    /// let mut slice2 = [1, 2, 3, 4];
    ///
    /// slice1.swap_with_slice(&mut slice2[2..]);
    ///
    /// assert_eq!(slice1, [3, 4]);
    /// assert_eq!(slice2, [1, 2, 0, 0]);
    /// ```
    ///
    /// Rust enforces that there can only be one mutable reference to a
    /// particular piece of data in a particular scope. Because of this,
    /// attempting to use `swap_with_slice` on a single slice will result in
    /// a compile failure:
    ///
    /// ```compile_fail
    /// let mut slice = [1, 2, 3, 4, 5];
    /// slice[..2].swap_with_slice(&mut slice[3..]); // compile fail!
    /// ```
    ///
    /// To work around this, we can use [`split_at_mut`] to create two distinct
    /// mutable sub-slices from a slice:
    ///
    /// ```
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// {
    ///     let (left, right) = slice.split_at_mut(2);
    ///     left.swap_with_slice(&mut right[1..]);
    /// }
    ///
    /// assert_eq!(slice, [4, 5, 3, 1, 2]);
    /// ```
    ///
    /// [`split_at_mut`]: #method.split_at_mut
    #[stable(feature = "swap_with_slice", since = "1.27.0")]
    pub fn swap_with_slice(&mut self, other: &mut [T]) {
        SliceExt::swap_with_slice(self, other)
    }
}}

#[lang = "slice"]
#[cfg(not(test))]
#[cfg(not(stage0))]
impl<T> [T] {
    slice_core_methods!();
}

// FIXME: remove (inline) this macro
// when updating to a bootstrap compiler that has the new lang items.
#[cfg_attr(stage0, macro_export)]
#[unstable(feature = "core_slice_ext", issue = "32110")]
macro_rules! slice_u8_core_methods { () => {
    /// Checks if all bytes in this slice are within the ASCII range.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.iter().all(|b| b.is_ascii())
    }

    /// Checks that two slices are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other).all(|(a, b)| {
                a.eq_ignore_ascii_case(b)
            })
    }

    /// Converts this slice to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        for byte in self {
            byte.make_ascii_uppercase();
        }
    }

    /// Converts this slice to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        for byte in self {
            byte.make_ascii_lowercase();
        }
    }
}}

#[lang = "slice_u8"]
#[cfg(not(test))]
#[cfg(not(stage0))]
impl [u8] {
    slice_u8_core_methods!();
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
impl<T, I> ops::Index<I> for [T]
    where I: SliceIndex<[T]>
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
    where I: SliceIndex<[T]>
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
pub trait SliceIndex<T: ?Sized> {
    /// The output type returned by methods.
    type Output: ?Sized;

    /// Returns a shared reference to the output at this location, if in
    /// bounds.
    fn get(self, slice: &T) -> Option<&Self::Output>;

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    fn get_mut(self, slice: &mut T) -> Option<&mut Self::Output>;

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    unsafe fn get_unchecked(self, slice: &T) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    unsafe fn get_unchecked_mut(self, slice: &mut T) -> &mut Self::Output;

    /// Returns a shared reference to the output at this location, panicking
    /// if out of bounds.
    fn index(self, slice: &T) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, panicking
    /// if out of bounds.
    fn index_mut(self, slice: &mut T) -> &mut Self::Output;
}

#[stable(feature = "slice-get-slice-impls", since = "1.15.0")]
impl<T> SliceIndex<[T]> for usize {
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
impl<T> SliceIndex<[T]> for  ops::Range<usize> {
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
impl<T> SliceIndex<[T]> for ops::RangeTo<usize> {
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
impl<T> SliceIndex<[T]> for ops::RangeFrom<usize> {
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
impl<T> SliceIndex<[T]> for ops::RangeFull {
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


#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<T> SliceIndex<[T]> for ops::RangeInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        if self.end == usize::max_value() { None }
        else { (self.start..self.end + 1).get(slice) }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        if self.end == usize::max_value() { None }
        else { (self.start..self.end + 1).get_mut(slice) }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (self.start..self.end + 1).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (self.start..self.end + 1).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        assert!(self.end != usize::max_value(),
            "attempted to index slice up to maximum usize");
        (self.start..self.end + 1).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        assert!(self.end != usize::max_value(),
            "attempted to index slice up to maximum usize");
        (self.start..self.end + 1).index_mut(slice)
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<T> SliceIndex<[T]> for ops::RangeToInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0..=self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0..=self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (0..=self.end).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..=self.end).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0..=self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..=self.end).index_mut(slice)
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

#[inline]
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
                let exact = unsafe { ptrdistance(self.ptr, self.end) };
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

            #[inline]
            fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R where
                Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
            {
                // manual unrolling is needed when there are conditional exits from the loop
                let mut accum = init;
                unsafe {
                    while ptrdistance(self.ptr, self.end) >= 4 {
                        accum = f(accum, $mkref!(self.ptr.post_inc()))?;
                        accum = f(accum, $mkref!(self.ptr.post_inc()))?;
                        accum = f(accum, $mkref!(self.ptr.post_inc()))?;
                        accum = f(accum, $mkref!(self.ptr.post_inc()))?;
                    }
                    while self.ptr != self.end {
                        accum = f(accum, $mkref!(self.ptr.post_inc()))?;
                    }
                }
                Try::from_ok(accum)
            }

            #[inline]
            fn fold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
                where Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                // Let LLVM unroll this, rather than using the default
                // impl that would force the manual unrolling above
                let mut accum = init;
                while let Some(x) = self.next() {
                    accum = f(accum, x);
                }
                accum
            }

            #[inline]
            #[rustc_inherit_overflow_checks]
            fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
                Self: Sized,
                P: FnMut(Self::Item) -> bool,
            {
                // The addition might panic on overflow
                // Use the len of the slice to hint optimizer to remove result index bounds check.
                let n = make_slice!(self.ptr, self.end).len();
                self.try_fold(0, move |i, x| {
                    if predicate(x) { Err(i) }
                    else { Ok(i + 1) }
                }).err()
                    .map(|i| {
                        unsafe { assume(i < n) };
                        i
                    })
            }

            #[inline]
            fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
                P: FnMut(Self::Item) -> bool,
                Self: Sized + ExactSizeIterator + DoubleEndedIterator
            {
                // No need for an overflow check here, because `ExactSizeIterator`
                // implies that the number of elements fits into a `usize`.
                // Use the len of the slice to hint optimizer to remove result index bounds check.
                let n = make_slice!(self.ptr, self.end).len();
                self.try_rfold(n, move |i, x| {
                    let i = i - 1;
                    if predicate(x) { Err(i) }
                    else { Ok(i) }
                }).err()
                    .map(|i| {
                        unsafe { assume(i < n) };
                        i
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

            #[inline]
            fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R where
                Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
            {
                // manual unrolling is needed when there are conditional exits from the loop
                let mut accum = init;
                unsafe {
                    while ptrdistance(self.ptr, self.end) >= 4 {
                        accum = f(accum, $mkref!(self.end.pre_dec()))?;
                        accum = f(accum, $mkref!(self.end.pre_dec()))?;
                        accum = f(accum, $mkref!(self.end.pre_dec()))?;
                        accum = f(accum, $mkref!(self.end.pre_dec()))?;
                    }
                    while self.ptr != self.end {
                        accum = f(accum, $mkref!(self.end.pre_dec()))?;
                    }
                }
                Try::from_ok(accum)
            }

            #[inline]
            fn rfold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
                where Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                // Let LLVM unroll this, rather than using the default
                // impl that would force the manual unrolling above
                let mut accum = init;
                while let Some(x) = self.next_back() {
                    accum = f(accum, x);
                }
                accum
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

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, T> FusedIterator for Iter<'a, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for Iter<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> { Iter { ptr: self.ptr, end: self.end, _marker: self._marker } }
}

#[stable(feature = "slice_iter_as_ref", since = "1.13.0")]
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

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, T> FusedIterator for IterMut<'a, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for IterMut<'a, T> {}


// Return the number of elements of `T` from `start` to `end`.
// Return the arithmetic difference if `T` is zero size.
#[inline(always)]
unsafe fn ptrdistance<T>(start: *const T, end: *const T) -> usize {
    if mem::size_of::<T>() == 0 {
        (end as usize).wrapping_sub(start as usize)
    } else {
        end.offset_from(start) as usize
    }
}

// Extension methods for raw pointers, used by the iterators
trait PointerExt : Copy {
    unsafe fn slice_offset(self, i: isize) -> Self;

    /// Increments `self` by 1, but returns the old value.
    #[inline(always)]
    unsafe fn post_inc(&mut self) -> Self {
        let current = *self;
        *self = self.slice_offset(1);
        current
    }

    /// Decrements `self` by 1, and returns the new value.
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
    /// Marks the underlying iterator as complete, extracting the remaining
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

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
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

#[stable(feature = "fused", since = "1.26.0")]
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

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, T, P> FusedIterator for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [slices].
///
/// [`rsplit`]: ../../std/primitive.slice.html#method.rsplit
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "slice_rsplit", since = "1.27.0")]
#[derive(Clone)] // Is this correct, or does it incorrectly require `T: Clone`?
pub struct RSplit<'a, T:'a, P> where P: FnMut(&T) -> bool {
    inner: Split<'a, T, P>
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplit")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> FusedIterator for RSplit<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`, starting from the end of the slice.
///
/// This struct is created by the [`rsplit_mut`] method on [slices].
///
/// [`rsplit_mut`]: ../../std/primitive.slice.html#method.rsplit_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub struct RSplitMut<'a, T:'a, P> where P: FnMut(&T) -> bool {
    inner: SplitMut<'a, T, P>
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplitMut")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplitMut<'a, T, P> where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> FusedIterator for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {}

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
#[derive(Debug)]
struct GenericSplitN<I> {
    iter: I,
    count: usize,
}

impl<T, I: SplitIter<Item=T>> Iterator for GenericSplitN<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.count {
            0 => None,
            1 => { self.count -= 1; self.iter.finish() }
            _ => { self.count -= 1; self.iter.next() }
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
    inner: GenericSplitN<RSplit<'a, T, P>>
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
    inner: GenericSplitN<RSplitMut<'a, T, P>>
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

        #[stable(feature = "fused", since = "1.26.0")]
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

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
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

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, T> FusedIterator for Windows<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Windows<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        from_raw_parts(self.v.as_ptr().offset(i as isize), self.size)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
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
    chunk_size: usize
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Chunks<'a, T> {
    fn clone(&self) -> Chunks<'a, T> {
        Chunks {
            v: self.v,
            chunk_size: self.chunk_size,
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
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
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
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
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
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
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
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
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
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Chunks<'a, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, T> FusedIterator for Chunks<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Chunks<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        let start = i * self.chunk_size;
        let end = match start.checked_add(self.chunk_size) {
            None => self.v.len(),
            Some(end) => cmp::min(end, self.v.len()),
        };
        from_raw_parts(self.v.as_ptr().offset(start as isize), end - start)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
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

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, T> FusedIterator for ChunksMut<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for ChunksMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut [T] {
        let start = i * self.chunk_size;
        let end = match start.checked_add(self.chunk_size) {
            None => self.v.len(),
            Some(end) => cmp::min(end, self.v.len()),
        };
        from_raw_parts_mut(self.v.as_mut_ptr().offset(start as isize), end - start)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time).
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted.
///
/// This struct is created by the [`exact_chunks`] method on [slices].
///
/// [`exact_chunks`]: ../../std/primitive.slice.html#method.exact_chunks
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[unstable(feature = "exact_chunks", issue = "47115")]
pub struct ExactChunks<'a, T:'a> {
    v: &'a [T],
    chunk_size: usize
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> Clone for ExactChunks<'a, T> {
    fn clone(&self) -> ExactChunks<'a, T> {
        ExactChunks {
            v: self.v,
            chunk_size: self.chunk_size,
        }
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> Iterator for ExactChunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (_, snd) = self.v.split_at(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> DoubleEndedIterator for ExactChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> ExactSizeIterator for ExactChunks<'a, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> FusedIterator for ExactChunks<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for ExactChunks<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        let start = i * self.chunk_size;
        from_raw_parts(self.v.as_ptr().offset(start as isize), self.chunk_size)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time). When the slice len is not evenly divided by the chunk
/// size, the last up to `chunk_size-1` elements will be omitted.
///
/// This struct is created by the [`exact_chunks_mut`] method on [slices].
///
/// [`exact_chunks_mut`]: ../../std/primitive.slice.html#method.exact_chunks_mut
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[unstable(feature = "exact_chunks", issue = "47115")]
pub struct ExactChunksMut<'a, T:'a> {
    v: &'a mut [T],
    chunk_size: usize
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> Iterator for ExactChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
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
            let tmp = mem::replace(&mut self.v, &mut []);
            let (_, snd) = tmp.split_at_mut(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> DoubleEndedIterator for ExactChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> ExactSizeIterator for ExactChunksMut<'a, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "exact_chunks", issue = "47115")]
impl<'a, T> FusedIterator for ExactChunksMut<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for ExactChunksMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut [T] {
        let start = i * self.chunk_size;
        from_raw_parts_mut(self.v.as_mut_ptr().offset(start as isize), self.chunk_size)
    }
    fn may_have_side_effect() -> bool { false }
}

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
/// `p` must be non-null, even for zero-length slices, because non-zero bits
/// are required to distinguish between a zero-length slice within `Some()`
/// from `None`. `p` can be a bogus non-dereferencable pointer, such as `0x1`,
/// for zero-length slices, though.
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
/// mutable slice. `p` must be non-null even for zero-length slices as with
/// `from_raw_parts`.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_raw_parts_mut<'a, T>(p: *mut T, len: usize) -> &'a mut [T] {
    mem::transmute(Repr { data: p, len: len })
}

/// Converts a reference to T into a slice of length 1 (without copying).
#[unstable(feature = "from_ref", issue = "45703")]
pub fn from_ref<T>(s: &T) -> &[T] {
    unsafe {
        from_raw_parts(s, 1)
    }
}

/// Converts a reference to T into a slice of length 1 (without copying).
#[unstable(feature = "from_ref", issue = "45703")]
pub fn from_ref_mut<T>(s: &mut T) -> &mut [T] {
    unsafe {
        from_raw_parts_mut(s, 1)
    }
}

// This function is public only because there is no other way to unit test heapsort.
#[unstable(feature = "sort_internals", reason = "internal to sort module", issue = "0")]
#[doc(hidden)]
pub fn heapsort<T, F>(v: &mut [T], mut is_less: F)
    where F: FnMut(&T, &T) -> bool
{
    sort::heapsort(v, &mut is_less);
}

//
// Comparison traits
//

extern {
    /// Calls implementation provided memcmp.
    ///
    /// Interprets the data as u8.
    ///
    /// Returns 0 for equal, < 0 for less than and > 0 for greater
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

trait SliceContains: Sized {
    fn slice_contains(&self, x: &[Self]) -> bool;
}

impl<T> SliceContains for T where T: PartialEq {
    default fn slice_contains(&self, x: &[Self]) -> bool {
        x.iter().any(|y| *y == *self)
    }
}

impl SliceContains for u8 {
    fn slice_contains(&self, x: &[Self]) -> bool {
        memchr::memchr(*self, x).is_some()
    }
}

impl SliceContains for i8 {
    fn slice_contains(&self, x: &[Self]) -> bool {
        let byte = *self as u8;
        let bytes: &[u8] = unsafe { from_raw_parts(x.as_ptr() as *const u8, x.len()) };
        memchr::memchr(byte, bytes).is_some()
    }
}
