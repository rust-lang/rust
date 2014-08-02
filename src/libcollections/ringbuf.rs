// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A double-ended queue implemented as a circular buffer
//!
//! RingBuf implements the trait Deque. It should be imported with `use
//! collections::Deque`.

use core::prelude::*;

use core::cmp;
use core::fmt;
use core::mem;
use core::num;
use core::ptr;
use core::slice;
use core::default::Default;
use core::iter::Chain;
use core::iter::FromIterator;
use core::raw::Slice;

use std::hash::{Writer, Hash};

use {Deque, Collection, Mutable, MutableSeq};
use vec::Vec;

/// RingBuf is a circular buffer that implements Deque.
///
/// # Examples
///
/// ```rust
/// # use std::collections::{RingBuf, Deque};
/// let mut ringbuf = RingBuf::new();
/// ringbuf.push_front(1i);
/// ringbuf.push(2);
///
/// assert_eq!(ringbuf.len(), 2);
/// assert_eq!(ringbuf.get(0), &1);
/// assert_eq!(ringbuf.front(), Some(&1));
/// assert_eq!(ringbuf.back(), Some(&2));
///
/// assert_eq!(ringbuf.pop_back(), Some(2));
/// assert_eq!(ringbuf.len(), 1);
/// ```
#[unsafe_no_drop_flag]
pub struct RingBuf<T> {

    /// The index of the 0th element
    /// invariant: `0 <= lo < cap`
    lo: uint,

    /// The number of elements currently in the ring.
    /// invariant: `0 <= len <= cap`
    len: uint,

    /// Capacity of the buffer.
    cap: uint,

    /// Pointer to the start of the buffer
    ptr: *mut T
}

impl<T> Collection for RingBuf<T> {
    /// Return the number of elements in the `RingBuf`.
    #[inline]
    fn len(&self) -> uint {
        self.len
    }
}

impl<T> Mutable for RingBuf<T> {
    /// Clear the `RingBuf`, removing all values.
    #[inline]
    fn clear(&mut self) {
        self.truncate(0)
    }
}

impl<T> Deque<T> for RingBuf<T> {
    /// Return a reference to the first element in the `RingBuf`.
    fn front<'a>(&'a self) -> Option<&'a T> {
        if self.len > 0 { Some(self.get(0)) } else { None }
    }

    /// Return a mutable reference to the first element in the `RingBuf`.
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        if self.len > 0 { Some(self.get_mut(0)) } else { None }
    }

    /// Return a reference to the last element in the `RingBuf`.
    fn back<'a>(&'a self) -> Option<&'a T> {
        if self.len > 0 { Some(self.get(self.len - 1)) } else { None }
    }

    /// Return a mutable reference to the last element in the `RingBuf`.
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        let len = self.len;
        if len > 0 { Some(self.get_mut(len - 1)) } else { None }
    }

    /// Remove the first element from a ring buffer and return it, or `None` if
    /// it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut ringbuf = RingBuf::new();
    /// ringbuf.push(1i);
    /// assert_eq!(Some(1), ringbuf.pop_front());
    /// assert_eq!(None, ringbuf.pop_front());
    /// ```
    #[inline]
    fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let offset = self.get_offset(0) as int;
                self.lo = self.get_offset(1);
                self.len -= 1;
                Some(ptr::read(self.ptr.offset(offset) as *const T))
            }
        }
    }

    /// Prepend an element to a ring buffer.
    ///
    /// # Failure
    ///
    /// Fails if the number of elements in the ring buffer overflows a `uint`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut ringbuf = RingBuf::new();
    /// ringbuf.push(1i);
    /// assert_eq!(Some(&1), ringbuf.front());
    /// ```
    #[inline]
    fn push_front(&mut self, value: T) {
        if mem::size_of::<T>() == 0 {
            // zero-size types consume no memory,
            // so we can't rely on the address space running out
            self.len = self.len.checked_add(&1).expect("length overflow");
            unsafe { mem::forget(value); }
            return;
        }
        if self.len == self.cap {
            let capacity = cmp::max(self.len, 1) * 2;
            self.resize(capacity);
        }
        unsafe {
            let offset = self.get_front_offset();
            let slot = self.ptr.offset(offset as int);
            ptr::write(slot, value);
            self.len += 1;
            self.lo = offset;
        }
    }
}

impl<T> MutableSeq<T> for RingBuf<T> {
    /// Append an element to a ring buffer.
    ///
    /// # Failure
    ///
    /// Fails if the number of elements in the ring buffer overflows a `uint`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut ringbuf = RingBuf::new();
    /// ringbuf.push(1i);
    /// assert_eq!(Some(&1), ringbuf.back());
    /// ```
    #[inline]
    fn push(&mut self, value: T) {
        if mem::size_of::<T>() == 0 {
            // zero-size types consume no memory, so we can't rely on the
            // address space running out
            self.len = self.len.checked_add(&1).expect("length overflow");
            unsafe { mem::forget(value); }
            return
        }
        if self.len == self.cap {
            let capacity = cmp::max(self.len, 1) * 2;
            self.resize(capacity);
        }
        unsafe {
            let offset = self.get_back_offset() as int;
            let slot = self.ptr.offset(offset);
            ptr::write(slot, value);
            self.len += 1;
        }
    }

    /// Remove the last element from a ring buffer and return it, or `None` if
    /// it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut ringbuf = RingBuf::new();
    /// ringbuf.push(1i);
    /// assert_eq!(Some(1), ringbuf.pop_back());
    /// assert_eq!(None, ringbuf.pop_back());
    /// ```
    #[inline]
    fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let offset = self.get_offset(self.len - 1) as int;
                self.len -= 1;
                Some(ptr::read(self.ptr.offset(offset) as *const T))
            }
        }
    }
}

impl<T> Default for RingBuf<T> {
    #[inline]
    fn default() -> RingBuf<T> { RingBuf::new() }
}

impl<T> RingBuf<T> {
    /// Construct a new, empty `RingBuf`.
    ///
    /// The ring buffer will not allocate until elements are pushed onto it.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut ringbuf: RingBuf<int> = RingBuf::new();
    /// ```
    pub fn new() -> RingBuf<T> {
        RingBuf::with_capacity(0)
    }

    /// Constructs a new, empty `RingBuf` with the specified capacity.
    ///
    /// The ring will be able to hold exactly `capacity` elements without
    /// reallocating. If `capacity` is 0, the ringbuf will not allocate.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let ring: RingBuf<int> = RingBuf::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: uint) -> RingBuf<T> {
        let ptr: *mut T = unsafe { alloc(capacity) };
        RingBuf { lo: 0, len: 0, cap: capacity, ptr: ptr }
    }

    /// Constructs a new `RingBuf` from the elements in a `Vec`.
    ///
    /// No copying will be done, and the new ring buffer will have the same
    /// capacity as the provided vec.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut vec = vec![1i, 2, 3];
    /// let ringbuf = RingBuf::from_vec(vec);
    /// ```
    pub fn from_vec(mut vec: Vec<T>) -> RingBuf<T> {
        let len = vec.len();
        let cap = vec.capacity();
        let ptr = vec.as_mut_ptr();
        let ringbuf = RingBuf { lo: 0, len: len, cap: cap, ptr: ptr };
        unsafe { mem::forget(vec); }
        ringbuf
    }

    /// Constructs a new `Vec` from the elements in a `RingBuf`.
    ///
    /// May require copying and temporary allocation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut ringbuf = RingBuf::new();
    /// ringbuf.push_front(1i);
    /// ringbuf.push(2);
    /// let vec = ringbuf.into_vec();
    /// assert_eq!(&[1i, 2], vec.as_slice());
    /// ```
    pub fn into_vec(mut self) -> Vec<T> {
        self.reset();

        let vec;
        unsafe {
            vec = Vec::from_raw_parts(self.len, self.cap, self.ptr);
            mem::forget(self);
        }
        vec
    }

    /// Retrieve an element in the RingBuf by index
    ///
    /// Fails if there is no element with the given index
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(3i);
    /// buf.push(4);
    /// buf.push(5);
    /// assert_eq!(buf.get(1), &4);
    /// ```
    pub fn get<'a>(&'a self, index: uint) -> &'a T {
        assert!(index < self.len);
        let offset = self.get_offset(index) as int;
        unsafe { &*self.ptr.offset(offset) }
    }

    /// Retrieve an element in the RingBuf by index
    ///
    /// Fails if there is no element with the given index
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(3i);
    /// buf.push(4);
    /// buf.push(5);
    /// *buf.get_mut(1) = 7;
    /// assert_eq!(buf.get(1), &7);
    /// ```
    pub fn get_mut<'a>(&'a mut self, index: uint) -> &'a mut T {
        assert!(index < self.len);
        let offset = self.get_offset(index) as int;
        unsafe { &mut *self.ptr.offset(offset) }
    }

    /// Swap elements at indices `i` and `j`
    ///
    /// `i` and `j` may be equal.
    ///
    /// # Failure
    ///
    /// Fails if there is no element with the given index
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(3i);
    /// buf.push(4);
    /// buf.push(5);
    /// buf.swap(0, 2);
    /// assert_eq!(buf.get(0), &5);
    /// assert_eq!(buf.get(2), &3);
    /// ```
    pub fn swap(&mut self, i: uint, j: uint) {
        assert!(i < self.len());
        assert!(j < self.len());
        let i_offset = self.get_offset(i) as int;
        let j_offset = self.get_offset(j) as int;
        unsafe {
            ptr::swap(self.ptr.offset(i_offset), self.ptr.offset(j_offset));
        }
    }

    /// Returns the number of elements the ringbuf can hold without
    /// reallocating.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let ringbuf: RingBuf<int> = RingBuf::with_capacity(10);
    /// assert_eq!(ringbuf.capacity(), 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> uint {
        self.cap
    }

    /// Reserves capacity for exactly `capacity` elements in the given ring
    /// buffer.
    ///
    /// If the capacity for `self` is already equal to or greater than the
    /// requested capacity, then no action is taken.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut ringbuf: RingBuf<int> = RingBuf::with_capacity(10);
    /// ringbuf.reserve_exact(11);
    /// assert_eq!(ringbuf.capacity(), 11);
    /// ```
    pub fn reserve_exact(&mut self, capacity: uint) {
        if capacity > self.cap {
            self.resize(capacity);
        }
    }

    /// Reserves capacity for at least `n` elements in the given ringbuf.
    ///
    /// This function will over-allocate in order to amortize the allocation
    /// costs in scenarios where the caller may need to repeatedly reserve
    /// additional space.
    ///
    /// If the capacity for `self` is already equal to or greater than the
    /// requested capacity, then no action is taken.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut ringbuf = RingBuf::<int>::new();
    /// ringbuf.reserve(10);
    /// assert!(ringbuf.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, capacity: uint) {
        self.reserve_exact(num::next_power_of_two(capacity))
    }

    /// Reserves capacity for at least `n` additional elements in the given
    /// ring buffer.
    ///
    /// # Failure
    ///
    /// Fails if the new capacity overflows `uint`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut ringbuf: RingBuf<int> = RingBuf::with_capacity(1);
    /// ringbuf.reserve_additional(10);
    /// assert!(ringbuf.capacity() >= 11);
    /// ```
    pub fn reserve_additional(&mut self, extra: uint) {
        if self.cap - self.len < extra {
            let size = self.len.checked_add(&extra).expect("length overflow");
            self.reserve(size);
        }
    }

    /// Shrink the capacity of the ring buffer as much as possible
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut ringbuf = RingBuf::new();
    /// ringbuf.push(1i);
    /// ringbuf.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&mut self) {
        let len = self.len;
        self.resize(len);
    }

    /// Sets the length of a ring buffer.
    ///
    /// This will explicitly set the size of the ring buffer, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// ring buf is actually the specified size.
    pub unsafe fn set_len(&mut self, len: uint) {
        self.len = len;
    }

    /// Shorten a ring buffer, dropping excess elements from the end.
    ///
    /// If `len` is greater than the ring buffer's current length, this has no
    /// effect.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut ringbuf = RingBuf::from_vec(vec![1i, 2, 3, 4]);
    /// ringbuf.truncate(2);
    /// assert_eq!(ringbuf.into_vec(), vec![1i, 2]);
    /// ```
    pub fn truncate(&mut self, len: uint) {
        for _ in range(len, self.len) { self.pop_back(); }
    }

    /// Work with `self` as a pair of slices.
    ///
    /// Either or both slices may be empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut rb = RingBuf::new();
    /// rb.push(1i);
    /// rb.push_front(0);
    /// let (slice1, slice2) = rb.as_slices();
    /// assert_eq!(slice1, &[0]);
    /// assert_eq!(slice2, &[1]);
    /// ```
    #[inline]
    pub fn as_slices<'a>(&'a self) -> (&'a [T], &'a [T]) {
        let (ptr1, len1, ptr2, len2) = self.get_slice_ptrs();
        unsafe {
            (mem::transmute(Slice { data: ptr1, len: len1 }),
             mem::transmute(Slice { data: ptr2, len: len2 }))
        }
    }

    /// Work with `self` as a pair of mutable slices.
    ///
    /// Either or both slices may be empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::{RingBuf, Deque};
    /// let mut rb = RingBuf::new();
    /// rb.push_front(1i);
    /// rb.push(2);
    /// let (slice1, slice2) = rb.as_mut_slices();
    /// assert_eq!(slice1, &[1]);
    /// assert_eq!(slice2, &[2]);
    /// ```
    #[inline]
    pub fn as_mut_slices<'a>(&'a mut self) -> (&'a mut [T], &'a mut [T]) {
        let (ptr1, len1, ptr2, len2) = self.get_slice_ptrs();
        unsafe {
            (mem::transmute(Slice { data: ptr1, len: len1 }),
             mem::transmute(Slice { data: ptr2, len: len2 }))
        }
    }

    /// Front-to-back iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(5i);
    /// buf.push(3);
    /// buf.push(4);
    /// assert_eq!(buf.iter().collect::<Vec<&int>>().as_slice(), &[&5, &3, &4]);
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        let (slice1, slice2) = self.as_slices();
        slice1.iter().chain(slice2.iter())
    }

    /// Front-to-back iterator which returns mutable values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(5i);
    /// buf.push(3);
    /// buf.push(4);
    /// for num in buf.mut_iter() {
    ///     *num = *num - 2;
    /// }
    /// assert_eq!(buf.mut_iter().collect::<Vec<&mut int>>().as_slice(), &[&mut 3, &mut 1, &mut 2]);
    /// ```
    #[inline]
    pub fn mut_iter<'a>(&'a mut self) -> MutItems<'a,T> {
        let (slice1, slice2) = self.as_mut_slices();
        slice1.mut_iter().chain(slice2.mut_iter())
    }

    /// Creates a consuming iterator, that is, one that moves each
    /// value out of the ringbuf (from front to back).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::RingBuf;
    /// let mut rb = RingBuf::new();
    /// rb.push("a".to_string());
    /// rb.push("b".to_string());
    /// for s in rb.move_iter() {
    ///     // s has type String, not &String
    ///     println!("{}", s);
    /// }
    /// ```
    #[inline]
    pub fn move_iter(self) -> MoveItems<T> {
        MoveItems { ringbuf: self }
    }
}

/// Allocate a buffer with the provided capacity.
// FIXME: #13996: need a way to mark the return value as `noalias`
#[inline(never)]
unsafe fn alloc<T>(capacity: uint) -> *mut T {
    let mut vec = Vec::<T>::with_capacity(capacity);
    let ptr = vec.as_mut_ptr();
    mem::forget(vec);
    ptr
}

/// Deallocate a buffer of the provided capacity.
#[inline]
unsafe fn dealloc<T>(ptr: *mut T, capacity: uint) {
    Vec::from_raw_parts(0, capacity, ptr);
}

impl<T> RingBuf<T> {

    /// Calculates the start and length of the slices in this ringbuf.
    #[inline]
    fn get_slice_ptrs(&self) -> (*const T, uint, *const T, uint) {
        let ptr1;
        let ptr2;
        let len1;
        let len2;
        unsafe {
            if self.lo > self.cap - self.len {
                ptr1 = self.ptr.offset(self.lo as int);
                ptr2 = self.ptr;
                len1 = self.cap - self.lo;
                len2 = self.len - len1;
            } else {
                ptr1 = self.ptr.offset(self.lo as int);
                ptr2 = self.ptr;
                len1 = self.len;
                len2 = 0;
            }
        }
        (ptr1 as *const T, len1, ptr2 as *const T, len2)
    }

    /// Resize the `RingBuf` to the specified capacity.
    ///
    /// # Failure
    ///
    /// Fails if the number of elements in the ring buffer is greater than
    /// the requested capacity.
    fn resize(&mut self, capacity: uint) {
        assert!(capacity >= self.len, "capacity underflow");

        if capacity == self.cap { return }
        if mem::size_of::<T>() == 0 { return }
        let ptr;
        unsafe {
            if capacity == 0 {
                ptr = 0 as *mut T;
            } else {
                let (slice1, slice2) = self.as_slices();
                ptr = alloc::<T>(capacity) as *mut T;
                let len1 = slice1.len();
                ptr::copy_nonoverlapping_memory(ptr, slice1.as_ptr(), len1);
                ptr::copy_nonoverlapping_memory(ptr.offset(len1 as int),
                slice2.as_ptr(),
                slice2.len());
            }
            if self.cap != 0 {
                dealloc(self.ptr, self.cap);
            }
        }
        self.ptr = ptr;
        self.cap = capacity;
        self.lo = 0;
    }

    /// Return the offset of the next back slot
    #[inline]
    fn get_back_offset(&self) -> uint {
        self.get_offset(self.len)
    }

    /// Return the offset of the next front slot
    #[inline]
    fn get_front_offset(&self) -> uint {
        if self.lo == 0 {
            self.cap - 1
        } else {
            self.lo - 1
        }
    }

    /// Return the offset of the given index in the underlying buffer.
    #[inline]
    fn get_offset(&self, index: uint) -> uint {
        // The order of these operations preserves numerical stability
        if self.lo >= self.cap - index {
            index - (self.cap - self.lo)
        } else {
            self.lo + index
        }
    }

    /// Reset the `lo` index to 0. This may require copying and temporary
    /// allocation.
    fn reset(&mut self) {
        if self.lo == 0 { return }
        // Shift elements to start of buffer
        {
            // `slice1` begins at the `lo` index.
            // `slice2` begins at the `0` index.
            let (slice1, slice2) = self.as_slices();
            let len1 = slice1.len();
            let len2 = slice2.len();
            if len1 == 0 {
                // Nothing to do
            } if len2 == 0 {
                // The buffer does not wrap. Move slice1.
                //
                //   lo
                //    V
                // +-+-+-+-+-+-+-+
                // | |x|x|x|x|x| |
                // +-+-+-+-+-+-+-+
                unsafe {
                    ptr::copy_memory(self.ptr,
                                     self.ptr.offset(self.lo as int) as *const T,
                                     self.len);
                }
            } if len1 <= (self.cap - len1) - len2 {
                // There is sufficient space to move slice2 without overwriting
                // slice1.
                //
                //           lo
                //            V
                // +-+-+-+-+-+-+-+
                // |x|x|x| | |x|x|
                // +-+-+-+-+-+-+-+
                unsafe {
                    ptr::copy_memory(self.ptr.offset(slice1.len() as int),
                                     slice2.as_ptr(),
                                     slice2.len());
                    ptr::copy_memory(self.ptr,
                                     slice1.as_ptr(),
                                     slice1.len());
                }
            } else if len1 < len2 {
                // Copy slice1 and move slice2.
                //
                //           lo
                //            V
                // +-+-+-+-+-+-+-+
                // |x|x|x|x| |x|x|
                // +-+-+-+-+-+-+-+
                unsafe {
                    let tmp = alloc(len1);
                    ptr::copy_nonoverlapping_memory(tmp,
                                                    slice1.as_ptr(),
                                                    len1);
                    ptr::copy_memory(self.ptr.offset(len1 as int),
                                     slice2.as_ptr(),
                                     len2);
                    ptr::copy_nonoverlapping_memory(self.ptr,
                                                    tmp as *const T,
                                                    len1);
                    dealloc(tmp, len1);
                }
            } else {
                // Copy slice2 and move slice1.
                //
                //         lo
                //          V
                // +-+-+-+-+-+-+-+
                // |x|x| | |x|x|x|
                // +-+-+-+-+-+-+-+
                unsafe {
                    let tmp = alloc(len2);
                    ptr::copy_nonoverlapping_memory(tmp,
                                                    slice2.as_ptr(),
                                                    len2);
                    ptr::copy_memory(self.ptr,
                                     slice1.as_ptr(),
                                     len1);
                    ptr::copy_nonoverlapping_memory(self.ptr.offset(len1 as int),
                    tmp as *const T,
                    len2);
                    dealloc(tmp, len2);
                }
            }
        }
        self.lo = 0;
    }
}

impl<T:Clone> Clone for RingBuf<T> {
    fn clone(&self) -> RingBuf<T> {
        let mut ringbuf = RingBuf::with_capacity(self.len);
        // Unsafe code so this can be optimised to a memcpy (or something
        // similarly fast) when T is Copy. LLVM is easily confused, so any
        // extra operations during the loop can prevent this optimisation
        {
            let (slice1, slice2) = self.as_slices();
            while ringbuf.len < slice1.len() {
                unsafe {
                    let len = ringbuf.len;
                    ptr::write(
                        ringbuf.ptr.offset(len as int),
                        slice1.unsafe_ref(len).clone());
                }
                ringbuf.len += 1;
            }
            while ringbuf.len < slice1.len() + slice2.len() {
                unsafe {
                    let len = ringbuf.len;
                    ptr::write(
                        ringbuf.ptr.offset(len as int),
                        slice2.unsafe_ref(len - slice1.len()).clone());
                }
                ringbuf.len += 1;
            }
        }
        ringbuf
    }

    fn clone_from(&mut self, source: &RingBuf<T>) {
        // drop anything in self that will not be overwritten
        if self.len() > source.len() {
            self.truncate(source.len())
        }
        // reuse the contained values' allocations/resources.
        for (place, thing) in self.mut_iter().zip(source.iter()) {
            place.clone_from(thing)
        }
        // self.len <= source.len due to the truncate above, so the
        // slice here is always in-bounds.
        let len = self.len();
        self.extend(source.iter().skip(len).map(|x| x.clone()));
    }
}

/// RingBuf iterator.
pub type Items<'a, T> = Chain<slice::Items<'a, T>, slice::Items<'a, T>>;

/// RingBuf mutable iterator.
pub type MutItems<'a, T> = Chain<slice::MutItems<'a, T>, slice::MutItems<'a, T>>;

/// An iterator that moves out of a RingBuf.
pub struct MoveItems<T> {
    ringbuf: RingBuf<T>
}

impl<T> Iterator<T> for MoveItems<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.ringbuf.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.ringbuf.len(), Some(self.ringbuf.len()))
    }
}

impl<T> DoubleEndedIterator<T> for MoveItems<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.ringbuf.pop_back()
    }
}

impl<T: PartialEq> PartialEq for RingBuf<T> {
    #[inline]
    fn eq(&self, other: &RingBuf<T>) -> bool {
        self.len == other.len
            && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for RingBuf<T> {}

impl<T: PartialOrd> PartialOrd for RingBuf<T> {
    #[inline]
    fn partial_cmp(&self, other: &RingBuf<T>) -> Option<Ordering> {
        for (a, b) in self.iter().zip(other.iter()) {
            let cmp = a.partial_cmp(b);
            if cmp != Some(Equal) {
                return cmp;
            }
        }
        Some(self.len.cmp(&other.len))
    }
}

impl<T: Ord> Ord for RingBuf<T> {
    #[inline]
    fn cmp(&self, other: &RingBuf<T>) -> Ordering {
        for (a, b) in self.iter().zip(other.iter()) {
            let cmp = a.cmp(b);
            if cmp != Equal {
                return cmp;
            }
        }
        self.len.cmp(&other.len)
    }
}

impl<S: Writer, A: Hash<S>> Hash<S> for RingBuf<A> {
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<T> FromIterator<T> for RingBuf<T> {
    fn from_iter<I:Iterator<T>>(mut iterator: I) -> RingBuf<T> {
        RingBuf::from_vec(iterator.collect())
    }
}

impl<T> Extendable<T> for RingBuf<T> {
    fn extend<I: Iterator<T>>(&mut self, mut iterator: I) {
        let (lower, _) = iterator.size_hint();
        self.reserve_additional(lower);
        for element in iterator {
            self.push(element)
        }
    }
}

impl<T: fmt::Show> fmt::Show for RingBuf<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));

        for (i, e) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}", *e));
        }

        write!(f, "]")
    }
}

#[unsafe_destructor]
impl<T> Drop for RingBuf<T> {
    fn drop(&mut self) {
        unsafe {
            for x in self.iter() {
                ptr::read(x);
            }
            dealloc(self.ptr, self.cap);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Show;
    use std::prelude::*;
    use std::gc::{GC, Gc};
    use std::hash;
    use test::Bencher;
    use test;

    use {Deque, Mutable, MutableSeq};
    use super::RingBuf;
    use vec::Vec;

    #[test]
    fn test_simple() {
        let mut d = RingBuf::new();
        assert_eq!(d.len(), 0u);
        d.push_front(17i);
        d.push_front(42i);
        d.push(137);
        assert_eq!(d.len(), 3u);
        d.push(137);
        assert_eq!(d.len(), 4u);
        debug!("{:?}", d.front());
        assert_eq!(*d.front().unwrap(), 42);
        debug!("{:?}", d.back());
        assert_eq!(*d.back().unwrap(), 137);
        let mut i = d.pop_front();
        debug!("{:?}", i);
        assert_eq!(i, Some(42));
        i = d.pop_back();
        debug!("{:?}", i);
        assert_eq!(i, Some(137));
        i = d.pop_back();
        debug!("{:?}", i);
        assert_eq!(i, Some(137));
        i = d.pop_back();
        debug!("{:?}", i);
        assert_eq!(i, Some(17));
        assert_eq!(d.len(), 0u);
        d.push(3);
        assert_eq!(d.len(), 1u);
        d.push_front(2);
        assert_eq!(d.len(), 2u);
        d.push(4);
        assert_eq!(d.len(), 3u);
        d.push_front(1);
        assert_eq!(d.len(), 4u);
        debug!("{:?}", d.get(0));
        debug!("{:?}", d.get(1));
        debug!("{:?}", d.get(2));
        debug!("{:?}", d.get(3));
        assert_eq!(*d.get(0), 1);
        assert_eq!(*d.get(1), 2);
        assert_eq!(*d.get(2), 3);
        assert_eq!(*d.get(3), 4);
    }

    #[test]
    fn test_boxes() {
        let a: Gc<int> = box(GC) 5;
        let b: Gc<int> = box(GC) 72;
        let c: Gc<int> = box(GC) 64;
        let d: Gc<int> = box(GC) 175;

        let mut deq = RingBuf::new();
        assert_eq!(deq.len(), 0);
        deq.push_front(a);
        deq.push_front(b);
        deq.push(c);
        assert_eq!(deq.len(), 3);
        deq.push(d);
        assert_eq!(deq.len(), 4);
        assert_eq!(deq.front(), Some(&b));
        assert_eq!(deq.back(), Some(&d));
        assert_eq!(deq.pop_front(), Some(b));
        assert_eq!(deq.pop_back(), Some(d));
        assert_eq!(deq.pop_back(), Some(c));
        assert_eq!(deq.pop_back(), Some(a));
        assert_eq!(deq.len(), 0);
        deq.push(c);
        assert_eq!(deq.len(), 1);
        deq.push_front(b);
        assert_eq!(deq.len(), 2);
        deq.push(d);
        assert_eq!(deq.len(), 3);
        deq.push_front(a);
        assert_eq!(deq.len(), 4);
        assert_eq!(*deq.get(0), a);
        assert_eq!(*deq.get(1), b);
        assert_eq!(*deq.get(2), c);
        assert_eq!(*deq.get(3), d);
    }

    #[cfg(test)]
    fn test_parameterized<T:Clone + PartialEq + Show>(a: T, b: T, c: T, d: T) {
        let mut deq = RingBuf::new();
        assert_eq!(deq.len(), 0);
        deq.push_front(a.clone());
        deq.push_front(b.clone());
        deq.push(c.clone());
        assert_eq!(deq.len(), 3);
        deq.push(d.clone());
        assert_eq!(deq.len(), 4);
        assert_eq!((*deq.front().unwrap()).clone(), b.clone());
        assert_eq!((*deq.back().unwrap()).clone(), d.clone());
        assert_eq!(deq.pop_front().unwrap(), b.clone());
        assert_eq!(deq.pop_back().unwrap(), d.clone());
        assert_eq!(deq.pop_back().unwrap(), c.clone());
        assert_eq!(deq.pop_back().unwrap(), a.clone());
        assert_eq!(deq.len(), 0);
        deq.push(c.clone());
        assert_eq!(deq.len(), 1);
        deq.push_front(b.clone());
        assert_eq!(deq.len(), 2);
        deq.push(d.clone());
        assert_eq!(deq.len(), 3);
        deq.push_front(a.clone());
        assert_eq!(deq.len(), 4);
        assert_eq!((*deq.get(0)).clone(), a.clone());
        assert_eq!((*deq.get(1)).clone(), b.clone());
        assert_eq!((*deq.get(2)).clone(), c.clone());
        assert_eq!((*deq.get(3)).clone(), d.clone());
    }

    #[test]
    fn test_push_front_grow() {
        let mut deq = RingBuf::new();
        for i in range(0u, 66) {
            deq.push_front(i);
        }
        assert_eq!(deq.len(), 66);

        for i in range(0u, 66) {
            assert_eq!(*deq.get(i), 65 - i);
        }

        let mut deq = RingBuf::new();
        for i in range(0u, 66) {
            deq.push(i);
        }

        for i in range(0u, 66) {
            assert_eq!(*deq.get(i), i);
        }
    }

    #[bench]
    fn bench_new(b: &mut test::Bencher) {
        b.iter(|| {
            let _: RingBuf<u64> = RingBuf::new();
        })
    }

    #[bench]
    fn bench_push_back(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            deq.push_back(0i);
        })
    }

    #[bench]
    fn bench_push_front(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            deq.push_front(0i);
        })
    }

    #[bench]
    fn bench_grow(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            for _ in range(0i, 65) {
                deq.push_front(1i);
            }
        })
    }

    #[deriving(Clone, PartialEq, Show)]
    enum Taggy {
        One(int),
        Two(int, int),
        Three(int, int, int),
    }

    #[deriving(Clone, PartialEq, Show)]
    enum Taggypar<T> {
        Onepar(int),
        Twopar(int, int),
        Threepar(int, int, int),
    }

    #[deriving(Clone, PartialEq, Show)]
    struct RecCy {
        x: int,
        y: int,
        t: Taggy
    }

    #[test]
    fn test_param_int() {
        test_parameterized::<int>(5, 72, 64, 175);
    }

    #[test]
    fn test_param_at_int() {
        test_parameterized::<Gc<int>>(box(GC) 5, box(GC) 72,
                                      box(GC) 64, box(GC) 175);
    }

    #[test]
    fn test_param_taggy() {
        test_parameterized::<Taggy>(One(1), Two(1, 2), Three(1, 2, 3), Two(17, 42));
    }

    #[test]
    fn test_param_taggypar() {
        test_parameterized::<Taggypar<int>>(Onepar::<int>(1),
                                            Twopar::<int>(1, 2),
                                            Threepar::<int>(1, 2, 3),
                                            Twopar::<int>(17, 42));
    }

    #[test]
    fn test_param_reccy() {
        let reccy1 = RecCy { x: 1, y: 2, t: One(1) };
        let reccy2 = RecCy { x: 345, y: 2, t: Two(1, 2) };
        let reccy3 = RecCy { x: 1, y: 777, t: Three(1, 2, 3) };
        let reccy4 = RecCy { x: 19, y: 252, t: Two(17, 42) };
        test_parameterized::<RecCy>(reccy1, reccy2, reccy3, reccy4);
    }

    #[test]
    fn test_with_capacity() {
        let mut d = RingBuf::with_capacity(0);
        d.push(1i);
        assert_eq!(d.len(), 1);
        let mut d = RingBuf::with_capacity(50);
        d.push(1i);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_reserve_exact() {
        let mut d = RingBuf::new();
        d.push(0u64);
        d.reserve_exact(50);
        assert_eq!(d.capacity(), 50);
        let mut d = RingBuf::new();
        d.push(0u32);
        d.reserve_exact(50);
        assert_eq!(d.capacity(), 50);
    }

    #[test]
    fn test_reserve() {
        let mut d = RingBuf::new();
        d.push(0u64);
        d.reserve(50);
        assert_eq!(d.capacity(), 64);
        let mut d = RingBuf::new();
        d.push(0u32);
        d.reserve(50);
        assert_eq!(d.capacity(), 64);
    }

    #[test]
    fn test_swap() {
        let mut d: RingBuf<int> = range(0i, 5).collect();
        d.pop_front();
        d.swap(0, 3);
        assert_eq!(d.iter().map(|&x|x).collect::<Vec<int>>(), vec!(4, 2, 3, 1));
    }

    #[test]
    fn test_iter() {
        let mut d = RingBuf::new();
        assert_eq!(d.iter().next(), None);
        assert_eq!(d.iter().size_hint(), (0, Some(0)));

        for i in range(0i, 5) {
            d.push(i);
        }
        assert_eq!(d.iter().collect::<Vec<&int>>().as_slice(), &[&0,&1,&2,&3,&4]);

        for i in range(6i, 9) {
            d.push_front(i);
        }
        assert_eq!(d.iter().collect::<Vec<&int>>().as_slice(), &[&8,&7,&6,&0,&1,&2,&3,&4]);

        let mut it = d.iter();
        let mut len = d.len();
        loop {
            match it.next() {
                None => break,
                _ => { len -= 1; assert_eq!(it.size_hint(), (len, Some(len))) }
            }
        }
    }

    #[test]
    fn test_rev_iter() {
        let mut d = RingBuf::new();
        assert_eq!(d.iter().rev().next(), None);

        for i in range(0i, 5) {
            d.push(i);
        }
        assert_eq!(d.iter().rev().collect::<Vec<&int>>().as_slice(), &[&4,&3,&2,&1,&0]);

        for i in range(6i, 9) {
            d.push_front(i);
        }
        assert_eq!(d.iter().rev().collect::<Vec<&int>>().as_slice(), &[&4,&3,&2,&1,&0,&6,&7,&8]);
    }

    #[test]
    fn test_mut_rev_iter_wrap() {
        let mut d = RingBuf::with_capacity(3);
        assert!(d.mut_iter().rev().next().is_none());

        d.push(1i);
        d.push(2);
        d.push(3);
        assert_eq!(d.pop_front(), Some(1));
        d.push(4);

        assert_eq!(d.mut_iter().rev().map(|x| *x).collect::<Vec<int>>(),
                   vec!(4, 3, 2));
    }

    #[test]
    fn test_mut_iter() {
        let mut d = RingBuf::new();
        assert!(d.mut_iter().next().is_none());

        for i in range(0u, 3) {
            d.push_front(i);
        }

        for (i, elt) in d.mut_iter().enumerate() {
            assert_eq!(*elt, 2 - i);
            *elt = i;
        }

        {
            let mut it = d.mut_iter();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_mut_rev_iter() {
        let mut d = RingBuf::new();
        assert!(d.mut_iter().rev().next().is_none());

        for i in range(0u, 3) {
            d.push_front(i);
        }

        for (i, elt) in d.mut_iter().rev().enumerate() {
            assert_eq!(*elt, i);
            *elt = i;
        }

        {
            let mut it = d.mut_iter().rev();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_from_iter() {
        use std::iter;
        let v = vec!(1i,2,3,4,5,6,7);
        let deq: RingBuf<int> = v.iter().map(|&x| x).collect();
        let u: Vec<int> = deq.iter().map(|&x| x).collect();
        assert_eq!(u, v);

        let mut seq = iter::count(0u, 2).take(256);
        let deq: RingBuf<uint> = seq.collect();
        for (i, &x) in deq.iter().enumerate() {
            assert_eq!(2*i, x);
        }
        assert_eq!(deq.len(), 256);
    }

    #[test]
    fn test_clone() {
        let mut d = RingBuf::new();
        d.push_front(17i);
        d.push_front(42);
        d.push(137);
        d.push(137);
        assert_eq!(d.len(), 4u);
        let mut e = d.clone();
        assert_eq!(e.len(), 4u);
        while !d.is_empty() {
            assert_eq!(d.pop_back(), e.pop_back());
        }
        assert_eq!(d.len(), 0u);
        assert_eq!(e.len(), 0u);
    }

    #[test]
    fn test_eq() {
        let mut d = RingBuf::new();
        assert!(d == RingBuf::with_capacity(0));
        d.push_front(137i);
        d.push_front(17);
        d.push_front(42);
        d.push(137);
        let mut e = RingBuf::with_capacity(0);
        e.push(42);
        e.push(17);
        e.push(137);
        e.push(137);
        assert!(&e == &d);
        e.pop_back();
        e.push(0);
        assert!(e != d);
        e.clear();
        assert!(e == RingBuf::new());
    }

    #[test]
    fn test_hash() {
      let mut x = RingBuf::new();
      let mut y = RingBuf::new();

      x.push(1i);
      x.push(2);
      x.push(3);

      y.push(0i);
      y.push(1i);
      y.pop_front();
      y.push(2);
      y.push(3);

      assert!(hash::hash(&x) == hash::hash(&y));
    }

    #[test]
    fn test_ord() {
        let x = RingBuf::new();
        let mut y = RingBuf::new();
        y.push(1i);
        y.push(2);
        y.push(3);
        assert!(x < y);
        assert!(y > x);
        assert!(x <= x);
        assert!(x >= x);
    }

    #[test]
    fn test_show() {
        let ringbuf: RingBuf<int> = range(0i, 10).collect();
        assert!(format!("{}", ringbuf).as_slice() == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

        let ringbuf: RingBuf<&str> = vec!["just", "one", "test", "more"].iter()
                                                                        .map(|&s| s)
                                                                        .collect();
        assert!(format!("{}", ringbuf).as_slice() == "[just, one, test, more]");
    }
}
