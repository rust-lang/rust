// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A growable list type with heap-allocated contents, written `Vec<T>` but pronounced 'vector.'
//!
//! Vectors have `O(1)` indexing, push (to the end) and pop (from the end).
//!
//! # Examples
//!
//! Explicitly creating a `Vec<T>` with `new()`:
//!
//! ```
//! let xs: Vec<i32> = Vec::new();
//! ```
//!
//! Using the `vec!` macro:
//!
//! ```
//! let ys: Vec<i32> = vec![];
//!
//! let zs = vec![1i32, 2, 3, 4, 5];
//! ```
//!
//! Push:
//!
//! ```
//! let mut xs = vec![1i32, 2];
//!
//! xs.push(3);
//! ```
//!
//! And pop:
//!
//! ```
//! let mut xs = vec![1i32, 2];
//!
//! let two = xs.pop();
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

use core::prelude::*;

use alloc::boxed::Box;
use alloc::heap::{EMPTY, allocate, reallocate, deallocate};
use core::cmp::max;
use core::cmp::{Ordering};
use core::default::Default;
use core::fmt;
use core::hash::{self, Hash};
use core::intrinsics::assume;
use core::iter::{repeat, FromIterator, IntoIterator};
use core::marker::PhantomData;
use core::mem;
use core::num::{Int, UnsignedInt};
use core::ops::{Index, IndexMut, Deref, Add};
use core::ops;
use core::ptr;
use core::ptr::Unique;
use core::raw::Slice as RawSlice;
use core::slice;
use core::usize;

use borrow::{Cow, IntoCow};

/// A growable list type, written `Vec<T>` but pronounced 'vector.'
///
/// # Examples
///
/// ```
/// let mut vec = Vec::new();
/// vec.push(1);
/// vec.push(2);
///
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec[0], 1);
///
/// assert_eq!(vec.pop(), Some(2));
/// assert_eq!(vec.len(), 1);
///
/// vec[0] = 7;
/// assert_eq!(vec[0], 7);
///
/// vec.push_all(&[1, 2, 3]);
///
/// for x in vec.iter() {
///     println!("{}", x);
/// }
/// assert_eq!(vec, vec![7, 1, 2, 3]);
/// ```
///
/// The `vec!` macro is provided to make initialization more convenient:
///
/// ```
/// let mut vec = vec![1, 2, 3];
/// vec.push(4);
/// assert_eq!(vec, vec![1, 2, 3, 4]);
/// ```
///
/// Use a `Vec<T>` as an efficient stack:
///
/// ```
/// let mut stack = Vec::new();
///
/// stack.push(1);
/// stack.push(2);
/// stack.push(3);
///
/// loop {
///     let top = match stack.pop() {
///         None => break, // empty
///         Some(x) => x,
///     };
///     // Prints 3, 2, 1
///     println!("{}", top);
/// }
/// ```
///
/// # Capacity and reallocation
///
/// The capacity of a vector is the amount of space allocated for any future elements that will be
/// added onto the vector. This is not to be confused with the *length* of a vector, which
/// specifies the number of actual elements within the vector. If a vector's length exceeds its
/// capacity, its capacity will automatically be increased, but its elements will have to be
/// reallocated.
///
/// For example, a vector with capacity 10 and length 0 would be an empty vector with space for 10
/// more elements. Pushing 10 or fewer elements onto the vector will not change its capacity or
/// cause reallocation to occur. However, if the vector's length is increased to 11, it will have
/// to reallocate, which can be slow. For this reason, it is recommended to use
/// `Vec::with_capacity` whenever possible to specify how big the vector is expected to get.
#[unsafe_no_drop_flag]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Vec<T> {
    ptr: Unique<T>,
    len: usize,
    cap: usize,
}

unsafe impl<T: Send> Send for Vec<T> { }
unsafe impl<T: Sync> Sync for Vec<T> { }

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////

impl<T> Vec<T> {
    /// Constructs a new, empty `Vec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec: Vec<i32> = Vec::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Vec<T> {
        // We want ptr to never be NULL so instead we set it to some arbitrary
        // non-null value which is fine since we never call deallocate on the ptr
        // if cap is 0. The reason for this is because the pointer of a slice
        // being NULL would break the null pointer optimization for enums.
        unsafe { Vec::from_raw_parts(EMPTY as *mut T, 0, 0) }
    }

    /// Constructs a new, empty `Vec<T>` with the specified capacity.
    ///
    /// The vector will be able to hold exactly `capacity` elements without reallocating. If
    /// `capacity` is 0, the vector will not allocate.
    ///
    /// It is important to note that this function does not specify the *length* of the returned
    /// vector, but only the *capacity*. (For an explanation of the difference between length and
    /// capacity, see the main `Vec<T>` docs above, 'Capacity and reallocation'.)
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec: Vec<_> = Vec::with_capacity(10);
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     vec.push(i);
    /// }
    ///
    /// // ...but this may make the vector reallocate
    /// vec.push(11);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        if mem::size_of::<T>() == 0 {
            unsafe { Vec::from_raw_parts(EMPTY as *mut T, 0, usize::MAX) }
        } else if capacity == 0 {
            Vec::new()
        } else {
            let size = capacity.checked_mul(mem::size_of::<T>())
                               .expect("capacity overflow");
            let ptr = unsafe { allocate(size, mem::min_align_of::<T>()) };
            if ptr.is_null() { ::alloc::oom() }
            unsafe { Vec::from_raw_parts(ptr as *mut T, 0, capacity) }
        }
    }

    /// Creates a `Vec<T>` directly from the raw components of another vector.
    ///
    /// This is highly unsafe, due to the number of invariants that aren't checked.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr;
    /// use std::mem;
    ///
    /// fn main() {
    ///     let mut v = vec![1, 2, 3];
    ///
    ///     // Pull out the various important pieces of information about `v`
    ///     let p = v.as_mut_ptr();
    ///     let len = v.len();
    ///     let cap = v.capacity();
    ///
    ///     unsafe {
    ///         // Cast `v` into the void: no destructor run, so we are in
    ///         // complete control of the allocation to which `p` points.
    ///         mem::forget(v);
    ///
    ///         // Overwrite memory with 4, 5, 6
    ///         for i in 0..len as isize {
    ///             ptr::write(p.offset(i), 4 + i);
    ///         }
    ///
    ///         // Put everything back together into a Vec
    ///         let rebuilt = Vec::from_raw_parts(p, len, cap);
    ///         assert_eq!(rebuilt, vec![4, 5, 6]);
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize,
                                 capacity: usize) -> Vec<T> {
        Vec {
            ptr: Unique::new(ptr),
            len: length,
            cap: capacity,
        }
    }

    /// Creates a vector by copying the elements from a raw pointer.
    ///
    /// This function will copy `elts` contiguous elements starting at `ptr` into a new allocation
    /// owned by the returned `Vec<T>`. The elements of the buffer are copied into the vector
    /// without cloning, as if `ptr::read()` were called on them.
    #[inline]
    #[unstable(feature = "collections",
               reason = "may be better expressed via composition")]
    pub unsafe fn from_raw_buf(ptr: *const T, elts: usize) -> Vec<T> {
        let mut dst = Vec::with_capacity(elts);
        dst.set_len(elts);
        ptr::copy_nonoverlapping_memory(dst.as_mut_ptr(), ptr, elts);
        dst
    }

    /// Returns the number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec: Vec<i32> = Vec::with_capacity(10);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the given
    /// `Vec<T>`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        if self.cap - self.len < additional {
            let err_msg = "Vec::reserve: `usize` overflow";
            let new_cap = self.len.checked_add(additional).expect(err_msg)
                .checked_next_power_of_two().expect(err_msg);
            self.grow_capacity(new_cap);
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to
    /// be inserted in the given `Vec<T>`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore capacity can not be relied upon to be precisely
    /// minimal. Prefer `reserve` if future insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.reserve_exact(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        if self.cap - self.len < additional {
            match self.len.checked_add(additional) {
                None => panic!("Vec::reserve: `usize` overflow"),
                Some(new_cap) => self.grow_capacity(new_cap)
            }
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator
    /// may still inform the vector that there is space for a few more elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = Vec::with_capacity(10);
    /// vec.push_all(&[1, 2, 3]);
    /// assert_eq!(vec.capacity(), 10);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        if mem::size_of::<T>() == 0 { return }

        if self.len == 0 {
            if self.cap != 0 {
                unsafe {
                    dealloc(*self.ptr, self.cap)
                }
                self.cap = 0;
            }
        } else if self.cap != self.len {
            unsafe {
                // Overflow check is unnecessary as the vector is already at
                // least this large.
                let ptr = reallocate(*self.ptr as *mut u8,
                                     self.cap * mem::size_of::<T>(),
                                     self.len * mem::size_of::<T>(),
                                     mem::min_align_of::<T>()) as *mut T;
                if ptr.is_null() { ::alloc::oom() }
                self.ptr = Unique::new(ptr);
            }
            self.cap = self.len;
        }
    }

    /// Convert the vector into Box<[T]>.
    ///
    /// Note that this will drop any excess capacity. Calling this and
    /// converting back to a vector with `into_vec()` is equivalent to calling
    /// `shrink_to_fit()`.
    #[unstable(feature = "collections")]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        self.shrink_to_fit();
        unsafe {
            let xs: Box<[T]> = mem::transmute(&mut *self);
            mem::forget(self);
            xs
        }
    }

    /// Shorten a vector, dropping excess elements.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.truncate(2);
    /// assert_eq!(vec, vec![1, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, len: usize) {
        unsafe {
            // drop any extra elements
            while len < self.len {
                // decrement len before the read(), so a panic on Drop doesn't
                // re-drop the just-failed value.
                self.len -= 1;
                ptr::read(self.get_unchecked(self.len));
            }
        }
    }

    /// Returns a mutable slice of the elements of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// fn foo(slice: &mut [i32]) {}
    ///
    /// let mut vec = vec![1, 2];
    /// foo(vec.as_mut_slice());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            mem::transmute(RawSlice {
                data: *self.ptr,
                len: self.len,
            })
        }
    }

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the vector (from start to end). The vector cannot be used after calling
    /// this.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec!["a".to_string(), "b".to_string()];
    /// for s in v.into_iter() {
    ///     // s has type String, not &String
    ///     println!("{}", s);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_iter(self) -> IntoIter<T> {
        unsafe {
            let ptr = *self.ptr;
            let cap = self.cap;
            let begin = ptr as *const T;
            let end = if mem::size_of::<T>() == 0 {
                (ptr as usize + self.len()) as *const T
            } else {
                ptr.offset(self.len() as isize) as *const T
            };
            mem::forget(self);
            IntoIter { allocation: ptr, cap: cap, ptr: begin, end: end }
        }
    }

    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// vector is actually the specified size.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3, 4];
    /// unsafe {
    ///     v.set_len(1);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    /// Removes an element from anywhere in the vector and return it, replacing
    /// it with the last element.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec!["foo", "bar", "baz", "qux"];
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(v, vec!["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v, vec!["baz", "qux"]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap_remove(&mut self, index: usize) -> T {
        let length = self.len();
        self.swap(index, length - 1);
        self.pop().unwrap()
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after position `i` one position to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not between `0` and the vector's length (both
    /// bounds inclusive).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, vec![1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, vec![1, 4, 2, 3, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(index <= len);
        // space for the new element
        self.reserve(1);

        unsafe { // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().offset(index as isize);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy_memory(p.offset(1), &*p, len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(&mut *p, element);
            }
            self.set_len(len + 1);
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after position `index` one position to the left.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v, vec![1, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        unsafe { // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().offset(index as isize);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy_memory(ptr, &*ptr.offset(1), len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns false.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, vec![2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn retain<F>(&mut self, mut f: F) where F: FnMut(&T) -> bool {
        let len = self.len();
        let mut del = 0;
        {
            let v = &mut **self;

            for i in 0..len {
                if !f(&v[i]) {
                    del += 1;
                } else if del > 0 {
                    v.swap(i-del, i);
                }
            }
        }
        if del > 0 {
            self.truncate(len - del);
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2);
    /// vec.push(3);
    /// assert_eq!(vec, vec!(1, 2, 3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, value: T) {
        if mem::size_of::<T>() == 0 {
            // zero-size types consume no memory, so we can't rely on the
            // address space running out
            self.len = self.len.checked_add(1).expect("length overflow");
            unsafe { mem::forget(value); }
            return
        }
        if self.len == self.cap {
            let old_size = self.cap * mem::size_of::<T>();
            let size = max(old_size, 2 * mem::size_of::<T>()) * 2;
            if old_size > size { panic!("capacity overflow") }
            unsafe {
                let ptr = alloc_or_realloc(*self.ptr, old_size, size);
                if ptr.is_null() { ::alloc::oom() }
                self.ptr = Unique::new(ptr);
            }
            self.cap = max(self.cap, 2) * 2;
        }

        unsafe {
            let end = (*self.ptr).offset(self.len as isize);
            ptr::write(&mut *end, value);
            self.len += 1;
        }
    }

    /// Removes the last element from a vector and returns it, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut vec = vec![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, vec![1, 2]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(ptr::read(self.get_unchecked(self.len())))
            }
        }
    }

    /// Moves all the elements of `other` into `Self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// let mut vec2 = vec![4, 5, 6];
    /// vec.append(&mut vec2);
    /// assert_eq!(vec, vec![1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, vec![]);
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "new API, waiting for dust to settle")]
    pub fn append(&mut self, other: &mut Self) {
        if mem::size_of::<T>() == 0 {
            // zero-size types consume no memory, so we can't rely on the
            // address space running out
            self.len = self.len.checked_add(other.len()).expect("length overflow");
            unsafe { other.set_len(0) }
            return;
        }
        self.reserve(other.len());
        let len = self.len();
        unsafe {
            ptr::copy_nonoverlapping_memory(
                self.get_unchecked_mut(len),
                other.as_ptr(),
                other.len());
        }

        self.len += other.len();
        unsafe { other.set_len(0); }
    }

    /// Creates a draining iterator that clears the `Vec` and iterates over
    /// the removed items from start to end.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec!["a".to_string(), "b".to_string()];
    /// for s in v.drain() {
    ///     // s has type String, not &String
    ///     println!("{}", s);
    /// }
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn drain(&mut self) -> Drain<T> {
        unsafe {
            let begin = *self.ptr as *const T;
            let end = if mem::size_of::<T>() == 0 {
                (*self.ptr as usize + self.len()) as *const T
            } else {
                (*self.ptr).offset(self.len() as isize) as *const T
            };
            self.set_len(0);
            Drain {
                ptr: begin,
                end: end,
                marker: PhantomData,
            }
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.truncate(0)
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = vec![1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.len }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = Vec::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Converts a `Vec<T>` to a `Vec<U>` where `T` and `U` have the same
    /// size and in case they are not zero-sized the same minimal alignment.
    ///
    /// # Panics
    ///
    /// Panics if `T` and `U` have differing sizes or are not zero-sized and
    /// have differing minimal alignments.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec![0, 1, 2];
    /// let w = v.map_in_place(|i| i + 3);
    /// assert_eq!(w.as_slice(), [3, 4, 5].as_slice());
    ///
    /// #[derive(PartialEq, Debug)]
    /// struct Newtype(u8);
    /// let bytes = vec![0x11, 0x22];
    /// let newtyped_bytes = bytes.map_in_place(|x| Newtype(x));
    /// assert_eq!(newtyped_bytes.as_slice(), [Newtype(0x11), Newtype(0x22)].as_slice());
    /// ```
    #[unstable(feature = "collections",
               reason = "API may change to provide stronger guarantees")]
    pub fn map_in_place<U, F>(self, mut f: F) -> Vec<U> where F: FnMut(T) -> U {
        // FIXME: Assert statically that the types `T` and `U` have the same
        // size.
        assert!(mem::size_of::<T>() == mem::size_of::<U>());

        let mut vec = self;

        if mem::size_of::<T>() != 0 {
            // FIXME: Assert statically that the types `T` and `U` have the
            // same minimal alignment in case they are not zero-sized.

            // These asserts are necessary because the `min_align_of` of the
            // types are passed to the allocator by `Vec`.
            assert!(mem::min_align_of::<T>() == mem::min_align_of::<U>());

            // This `as isize` cast is safe, because the size of the elements of the
            // vector is not 0, and:
            //
            // 1) If the size of the elements in the vector is 1, the `isize` may
            //    overflow, but it has the correct bit pattern so that the
            //    `.offset()` function will work.
            //
            //    Example:
            //        Address space 0x0-0xF.
            //        `u8` array at: 0x1.
            //        Size of `u8` array: 0x8.
            //        Calculated `offset`: -0x8.
            //        After `array.offset(offset)`: 0x9.
            //        (0x1 + 0x8 = 0x1 - 0x8)
            //
            // 2) If the size of the elements in the vector is >1, the `usize` ->
            //    `isize` conversion can't overflow.
            let offset = vec.len() as isize;
            let start = vec.as_mut_ptr();

            let mut pv = PartialVecNonZeroSized {
                vec: vec,

                start_t: start,
                // This points inside the vector, as the vector has length
                // `offset`.
                end_t: unsafe { start.offset(offset) },
                start_u: start as *mut U,
                end_u: start as *mut U,

                _marker: PhantomData,
            };
            //  start_t
            //  start_u
            //  |
            // +-+-+-+-+-+-+
            // |T|T|T|...|T|
            // +-+-+-+-+-+-+
            //  |           |
            //  end_u       end_t

            while pv.end_u as *mut T != pv.end_t {
                unsafe {
                    //  start_u start_t
                    //  |       |
                    // +-+-+-+-+-+-+-+-+-+
                    // |U|...|U|T|T|...|T|
                    // +-+-+-+-+-+-+-+-+-+
                    //          |         |
                    //          end_u     end_t

                    let t = ptr::read(pv.start_t);
                    //  start_u start_t
                    //  |       |
                    // +-+-+-+-+-+-+-+-+-+
                    // |U|...|U|X|T|...|T|
                    // +-+-+-+-+-+-+-+-+-+
                    //          |         |
                    //          end_u     end_t
                    // We must not panic here, one cell is marked as `T`
                    // although it is not `T`.

                    pv.start_t = pv.start_t.offset(1);
                    //  start_u   start_t
                    //  |         |
                    // +-+-+-+-+-+-+-+-+-+
                    // |U|...|U|X|T|...|T|
                    // +-+-+-+-+-+-+-+-+-+
                    //          |         |
                    //          end_u     end_t
                    // We may panic again.

                    // The function given by the user might panic.
                    let u = f(t);

                    ptr::write(pv.end_u, u);
                    //  start_u   start_t
                    //  |         |
                    // +-+-+-+-+-+-+-+-+-+
                    // |U|...|U|U|T|...|T|
                    // +-+-+-+-+-+-+-+-+-+
                    //          |         |
                    //          end_u     end_t
                    // We should not panic here, because that would leak the `U`
                    // pointed to by `end_u`.

                    pv.end_u = pv.end_u.offset(1);
                    //  start_u   start_t
                    //  |         |
                    // +-+-+-+-+-+-+-+-+-+
                    // |U|...|U|U|T|...|T|
                    // +-+-+-+-+-+-+-+-+-+
                    //            |       |
                    //            end_u   end_t
                    // We may panic again.
                }
            }

            //  start_u     start_t
            //  |           |
            // +-+-+-+-+-+-+
            // |U|...|U|U|U|
            // +-+-+-+-+-+-+
            //              |
            //              end_t
            //              end_u
            // Extract `vec` and prevent the destructor of
            // `PartialVecNonZeroSized` from running. Note that none of the
            // function calls can panic, thus no resources can be leaked (as the
            // `vec` member of `PartialVec` is the only one which holds
            // allocations -- and it is returned from this function. None of
            // this can panic.
            unsafe {
                let vec_len = pv.vec.len();
                let vec_cap = pv.vec.capacity();
                let vec_ptr = pv.vec.as_mut_ptr() as *mut U;
                mem::forget(pv);
                Vec::from_raw_parts(vec_ptr, vec_len, vec_cap)
            }
        } else {
            // Put the `Vec` into the `PartialVecZeroSized` structure and
            // prevent the destructor of the `Vec` from running. Since the
            // `Vec` contained zero-sized objects, it did not allocate, so we
            // are not leaking memory here.
            let mut pv = PartialVecZeroSized::<T,U> {
                num_t: vec.len(),
                num_u: 0,
                marker: PhantomData,
            };
            unsafe { mem::forget(vec); }

            while pv.num_t != 0 {
                unsafe {
                    // Create a `T` out of thin air and decrement `num_t`. This
                    // must not panic between these steps, as otherwise a
                    // destructor of `T` which doesn't exist runs.
                    let t = mem::uninitialized();
                    pv.num_t -= 1;

                    // The function given by the user might panic.
                    let u = f(t);

                    // Forget the `U` and increment `num_u`. This increment
                    // cannot overflow the `usize` as we only do this for a
                    // number of times that fits into a `usize` (and start with
                    // `0`). Again, we should not panic between these steps.
                    mem::forget(u);
                    pv.num_u += 1;
                }
            }
            // Create a `Vec` from our `PartialVecZeroSized` and make sure the
            // destructor of the latter will not run. None of this can panic.
            let mut result = Vec::new();
            unsafe {
                result.set_len(pv.num_u);
                mem::forget(pv);
            }
            result
        }
    }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated `Self`. `self` contains elements `[0, at)`,
    /// and the returned `Self` contains elements `[at, len)`.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1,2,3];
    /// let vec2 = vec.split_off(1);
    /// assert_eq!(vec, vec![1]);
    /// assert_eq!(vec2, vec![2, 3]);
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "new API, waiting for dust to settle")]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "`at` out of bounds");

        let other_len = self.len - at;
        let mut other = Vec::with_capacity(other_len);

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            ptr::copy_nonoverlapping_memory(
                other.as_mut_ptr(),
                self.as_ptr().offset(at as isize),
                other.len());
        }
        other
    }

}

impl<T: Clone> Vec<T> {
    /// Resizes the `Vec` in-place so that `len()` is equal to `new_len`.
    ///
    /// Calls either `extend()` or `truncate()` depending on whether `new_len`
    /// is larger than the current value of `len()` or not.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec!["hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(vec, vec!["hello", "world", "world"]);
    ///
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.resize(2, 0);
    /// assert_eq!(vec, vec![1, 2]);
    /// ```
    #[unstable(feature = "collections",
               reason = "matches collection reform specification; waiting for dust to settle")]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend(repeat(value).take(new_len - len));
        } else {
            self.truncate(new_len);
        }
    }

    /// Appends all elements in a slice to the `Vec`.
    ///
    /// Iterates over the slice `other`, clones each element, and then appends
    /// it to this `Vec`. The `other` vector is traversed in-order.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.push_all(&[2, 3, 4]);
    /// assert_eq!(vec, vec![1, 2, 3, 4]);
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "likely to be replaced by a more optimized extend")]
    pub fn push_all(&mut self, other: &[T]) {
        self.reserve(other.len());

        for i in 0..other.len() {
            let len = self.len();

            // Unsafe code so this can be optimised to a memcpy (or something similarly
            // fast) when T is Copy. LLVM is easily confused, so any extra operations
            // during the loop can prevent this optimisation.
            unsafe {
                ptr::write(
                    self.get_unchecked_mut(len),
                    other.get_unchecked(i).clone());
                self.set_len(len + 1);
            }
        }
    }
}

impl<T: PartialEq> Vec<T> {
    /// Removes consecutive repeated elements in the vector.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 2, 3, 2];
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(vec, vec![1, 2, 3, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn dedup(&mut self) {
        unsafe {
            // Although we have a mutable reference to `self`, we cannot make
            // *arbitrary* changes. The `PartialEq` comparisons could panic, so we
            // must ensure that the vector is in a valid state at all time.
            //
            // The way that we handle this is by using swaps; we iterate
            // over all the elements, swapping as we go so that at the end
            // the elements we wish to keep are in the front, and those we
            // wish to reject are at the back. We can then truncate the
            // vector. This operation is still O(n).
            //
            // Example: We start in this state, where `r` represents "next
            // read" and `w` represents "next_write`.
            //
            //           r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //           w
            //
            // Comparing self[r] against self[w-1], this is not a duplicate, so
            // we swap self[r] and self[w] (no effect as r==w) and then increment both
            // r and w, leaving us with:
            //
            //               r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //               w
            //
            // Comparing self[r] against self[w-1], this value is a duplicate,
            // so we increment `r` but leave everything else unchanged:
            //
            //                   r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //               w
            //
            // Comparing self[r] against self[w-1], this is not a duplicate,
            // so swap self[r] and self[w] and advance r and w:
            //
            //                       r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 2 | 1 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //                   w
            //
            // Not a duplicate, repeat:
            //
            //                           r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 2 | 3 | 1 | 3 |
            //     +---+---+---+---+---+---+
            //                       w
            //
            // Duplicate, advance r. End of vec. Truncate to w.

            let ln = self.len();
            if ln < 1 { return; }

            // Avoid bounds checks by using unsafe pointers.
            let p = self.as_mut_ptr();
            let mut r = 1;
            let mut w = 1;

            while r < ln {
                let p_r = p.offset(r as isize);
                let p_wm1 = p.offset((w - 1) as isize);
                if *p_r != *p_wm1 {
                    if r != w {
                        let p_w = p_wm1.offset(1);
                        mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
                r += 1;
            }

            self.truncate(w);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal methods and functions
////////////////////////////////////////////////////////////////////////////////

impl<T> Vec<T> {
    /// Reserves capacity for exactly `capacity` elements in the given vector.
    ///
    /// If the capacity for `self` is already equal to or greater than the
    /// requested capacity, then no action is taken.
    fn grow_capacity(&mut self, capacity: usize) {
        if mem::size_of::<T>() == 0 { return }

        if capacity > self.cap {
            let size = capacity.checked_mul(mem::size_of::<T>())
                               .expect("capacity overflow");
            unsafe {
                let ptr = alloc_or_realloc(*self.ptr, self.cap * mem::size_of::<T>(), size);
                if ptr.is_null() { ::alloc::oom() }
                self.ptr = Unique::new(ptr);
            }
            self.cap = capacity;
        }
    }
}

// FIXME: #13996: need a way to mark the return value as `noalias`
#[inline(never)]
unsafe fn alloc_or_realloc<T>(ptr: *mut T, old_size: usize, size: usize) -> *mut T {
    if old_size == 0 {
        allocate(size, mem::min_align_of::<T>()) as *mut T
    } else {
        reallocate(ptr as *mut u8, old_size, size, mem::min_align_of::<T>()) as *mut T
    }
}

#[inline]
unsafe fn dealloc<T>(ptr: *mut T, len: usize) {
    if mem::size_of::<T>() != 0 {
        deallocate(ptr as *mut u8,
                   len * mem::size_of::<T>(),
                   mem::min_align_of::<T>())
    }
}

#[doc(hidden)]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_elem<T: Clone>(elem: T, n: usize) -> Vec<T> {
    unsafe {
        let mut v = Vec::with_capacity(n);
        let mut ptr = v.as_mut_ptr();

        // Write all elements except the last one
        for i in 1..n {
            ptr::write(ptr, Clone::clone(&elem));
            ptr = ptr.offset(1);
            v.set_len(i); // Increment the length in every step in case Clone::clone() panics
        }

        if n > 0 {
            // We can write the last element directly without cloning needlessly
            ptr::write(ptr, elem);
            v.set_len(n);
        }

        v
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common trait implementations for Vec
////////////////////////////////////////////////////////////////////////////////

#[unstable(feature = "collections")]
impl<T:Clone> Clone for Vec<T> {
    fn clone(&self) -> Vec<T> { ::slice::SliceExt::to_vec(&**self) }

    fn clone_from(&mut self, other: &Vec<T>) {
        // drop anything in self that will not be overwritten
        if self.len() > other.len() {
            self.truncate(other.len())
        }

        // reuse the contained values' allocations/resources.
        for (place, thing) in self.iter_mut().zip(other.iter()) {
            place.clone_from(thing)
        }

        // self.len <= other.len due to the truncate above, so the
        // slice here is always in-bounds.
        let slice = &other[self.len()..];
        self.push_all(slice);
    }
}

#[cfg(stage0)]
impl<S: hash::Writer + hash::Hasher, T: Hash<S>> Hash<S> for Vec<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        Hash::hash(&**self, state)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(not(stage0))]
impl<T: Hash> Hash for Vec<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Index<usize> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: &usize) -> &T {
        // NB built-in indexing via `&[T]`
        &(**self)[*index]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IndexMut<usize> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: &usize) -> &mut T {
        // NB built-in indexing via `&mut [T]`
        &mut (**self)[*index]
    }
}


#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::Range<usize>> for Vec<T> {
    type Output = [T];
    #[inline]
    fn index(&self, index: &ops::Range<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeTo<usize>> for Vec<T> {
    type Output = [T];
    #[inline]
    fn index(&self, index: &ops::RangeTo<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeFrom<usize>> for Vec<T> {
    type Output = [T];
    #[inline]
    fn index(&self, index: &ops::RangeFrom<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeFull> for Vec<T> {
    type Output = [T];
    #[inline]
    fn index(&self, _index: &ops::RangeFull) -> &[T] {
        self.as_slice()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::Range<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: &ops::Range<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeTo<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: &ops::RangeTo<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeFrom<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: &ops::RangeFrom<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeFull> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, _index: &ops::RangeFull) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] { self.as_slice() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> FromIterator<T> for Vec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item=T>>(iterable: I) -> Vec<T> {
        let mut iterator = iterable.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut vector = Vec::with_capacity(lower);

        // This function should be the moral equivalent of:
        //
        //      for item in iterator {
        //          vector.push(item);
        //      }
        //
        // This equivalent crucially runs the iterator precisely once. Below we
        // actually in theory run the iterator twice (one without bounds checks
        // and one with). To achieve the "moral equivalent", we use the `if`
        // statement below to break out early.
        //
        // If the first loop has terminated, then we have one of two conditions.
        //
        // 1. The underlying iterator returned `None`. In this case we are
        //    guaranteed that less than `vector.capacity()` elements have been
        //    returned, so we break out early.
        // 2. The underlying iterator yielded `vector.capacity()` elements and
        //    has not yielded `None` yet. In this case we run the iterator to
        //    its end below.
        for element in iterator.by_ref().take(vector.capacity()) {
            let len = vector.len();
            unsafe {
                ptr::write(vector.get_unchecked_mut(len), element);
                vector.set_len(len + 1);
            }
        }

        if vector.len() == vector.capacity() {
            for element in iterator {
                vector.push(element);
            }
        }
        vector
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(mut self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

#[unstable(feature = "collections", reason = "waiting on Extend stability")]
impl<T> Extend<T> for Vec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item=T>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        let (lower, _) = iterator.size_hint();
        self.reserve(lower);
        for element in iterator {
            self.push(element)
        }
    }
}

impl<A, B> PartialEq<Vec<B>> for Vec<A> where A: PartialEq<B> {
    #[inline]
    fn eq(&self, other: &Vec<B>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Vec<B>) -> bool { PartialEq::ne(&**self, &**other) }
}

macro_rules! impl_eq {
    ($lhs:ty, $rhs:ty) => {
        impl<'b, A, B> PartialEq<$rhs> for $lhs where A: PartialEq<B> {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { PartialEq::ne(&**self, &**other) }
        }

        impl<'b, A, B> PartialEq<$lhs> for $rhs where B: PartialEq<A> {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$lhs) -> bool { PartialEq::ne(&**self, &**other) }
        }
    }
}

impl_eq! { Vec<A>, &'b [B] }
impl_eq! { Vec<A>, &'b mut [B] }

impl<'a, A, B> PartialEq<Vec<B>> for Cow<'a, [A]> where A: PartialEq<B> + Clone {
    #[inline]
    fn eq(&self, other: &Vec<B>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Vec<B>) -> bool { PartialEq::ne(&**self, &**other) }
}

impl<'a, A, B> PartialEq<Cow<'a, [A]>> for Vec<B> where A: Clone, B: PartialEq<A> {
    #[inline]
    fn eq(&self, other: &Cow<'a, [A]>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Cow<'a, [A]>) -> bool { PartialEq::ne(&**self, &**other) }
}

macro_rules! impl_eq_for_cowvec {
    ($rhs:ty) => {
        impl<'a, 'b, A, B> PartialEq<$rhs> for Cow<'a, [A]> where A: PartialEq<B> + Clone {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { PartialEq::ne(&**self, &**other) }
        }

        impl<'a, 'b, A, B> PartialEq<Cow<'a, [A]>> for $rhs where A: Clone, B: PartialEq<A> {
            #[inline]
            fn eq(&self, other: &Cow<'a, [A]>) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &Cow<'a, [A]>) -> bool { PartialEq::ne(&**self, &**other) }
        }
    }
}

impl_eq_for_cowvec! { &'b [B] }
impl_eq_for_cowvec! { &'b mut [B] }

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for Vec<T> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for Vec<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T> AsSlice<T> for Vec<T> {
    /// Returns a slice into `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// fn foo(slice: &[i32]) {}
    ///
    /// let vec = vec![1, 2];
    /// foo(vec.as_slice());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_slice(&self) -> &[T] {
        unsafe {
            let p = *self.ptr;
            if cfg!(not(stage0)) { // NOTE remove cfg after next snapshot
                assume(p != 0 as *mut T);
            }
            mem::transmute(RawSlice {
                data: p,
                len: self.len
            })
        }
    }
}

#[unstable(feature = "collections",
           reason = "recent addition, needs more experience")]
impl<'a, T: Clone> Add<&'a [T]> for Vec<T> {
    type Output = Vec<T>;

    #[inline]
    fn add(mut self, rhs: &[T]) -> Vec<T> {
        self.push_all(rhs);
        self
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        // This is (and should always remain) a no-op if the fields are
        // zeroed (when moving out, because of #[unsafe_no_drop_flag]).
        if self.cap != 0 {
            unsafe {
                for x in &*self {
                    ptr::read(x);
                }
                dealloc(*self.ptr, self.cap)
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for Vec<T> {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> Vec<T> {
        Vec::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Clone-on-write
////////////////////////////////////////////////////////////////////////////////

/// A clone-on-write vector
#[deprecated(since = "1.0.0", reason = "use Cow<'a, [T]> instead")]
#[unstable(feature = "collections")]
pub type CowVec<'a, T> = Cow<'a, [T]>;

#[unstable(feature = "collections")]
impl<'a, T> FromIterator<T> for Cow<'a, [T]> where T: Clone {
    fn from_iter<I: IntoIterator<Item=T>>(it: I) -> Cow<'a, [T]> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}

impl<'a, T: 'a> IntoCow<'a, [T]> for Vec<T> where T: Clone {
    fn into_cow(self) -> Cow<'a, [T]> {
        Cow::Owned(self)
    }
}

impl<'a, T> IntoCow<'a, [T]> for &'a [T] where T: Clone {
    fn into_cow(self) -> Cow<'a, [T]> {
        Cow::Borrowed(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////////////////////////

/// An iterator that moves out of a vector.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    allocation: *mut T, // the block of memory allocated for the vector
    cap: usize, // the capacity of the vector
    ptr: *const T,
    end: *const T
}

unsafe impl<T: Send> Send for IntoIter<T> { }
unsafe impl<T: Sync> Sync for IntoIter<T> { }

impl<T> IntoIter<T> {
    #[inline]
    /// Drops all items that have not yet been moved and returns the empty vector.
    #[unstable(feature = "collections")]
    pub fn into_inner(mut self) -> Vec<T> {
        unsafe {
            for _x in self.by_ref() { }
            let IntoIter { allocation, cap, ptr: _ptr, end: _end } = self;
            mem::forget(self);
            Vec::from_raw_parts(allocation, 0, cap)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.ptr == self.end {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // purposefully don't use 'ptr.offset' because for
                    // vectors with 0-size elements this would return the
                    // same pointer.
                    self.ptr = mem::transmute(self.ptr as usize + 1);

                    // Use a non-null pointer value
                    Some(ptr::read(EMPTY as *mut T))
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(1);

                    Some(ptr::read(old))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff = (self.end as usize) - (self.ptr as usize);
        let size = mem::size_of::<T>();
        let exact = diff / (if size == 0 {1} else {size});
        (exact, Some(exact))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            if self.end == self.ptr {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // See above for why 'ptr.offset' isn't used
                    self.end = mem::transmute(self.end as usize - 1);

                    // Use a non-null pointer value
                    Some(ptr::read(EMPTY as *mut T))
                } else {
                    self.end = self.end.offset(-1);

                    Some(ptr::read(mem::transmute(self.end)))
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        if self.cap != 0 {
            for _x in self.by_ref() {}
            unsafe {
                dealloc(self.allocation, self.cap);
            }
        }
    }
}

/// An iterator that drains a vector.
#[unsafe_no_drop_flag]
#[unstable(feature = "collections",
           reason = "recently added as part of collections reform 2")]
pub struct Drain<'a, T:'a> {
    ptr: *const T,
    end: *const T,
    marker: PhantomData<&'a T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.ptr == self.end {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // purposefully don't use 'ptr.offset' because for
                    // vectors with 0-size elements this would return the
                    // same pointer.
                    self.ptr = mem::transmute(self.ptr as usize + 1);

                    // Use a non-null pointer value
                    Some(ptr::read(EMPTY as *mut T))
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(1);

                    Some(ptr::read(old))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff = (self.end as usize) - (self.ptr as usize);
        let size = mem::size_of::<T>();
        let exact = diff / (if size == 0 {1} else {size});
        (exact, Some(exact))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            if self.end == self.ptr {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // See above for why 'ptr.offset' isn't used
                    self.end = mem::transmute(self.end as usize - 1);

                    // Use a non-null pointer value
                    Some(ptr::read(EMPTY as *mut T))
                } else {
                    self.end = self.end.offset(-1);

                    Some(ptr::read(self.end))
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // self.ptr == self.end == null if drop has already been called,
        // so we can use #[unsafe_no_drop_flag].

        // destroy the remaining elements
        for _x in self.by_ref() {}
    }
}

////////////////////////////////////////////////////////////////////////////////
// Conversion from &[T] to &Vec<T>
////////////////////////////////////////////////////////////////////////////////

/// Wrapper type providing a `&Vec<T>` reference via `Deref`.
#[unstable(feature = "collections")]
pub struct DerefVec<'a, T:'a> {
    x: Vec<T>,
    l: PhantomData<&'a T>,
}

#[unstable(feature = "collections")]
impl<'a, T> Deref for DerefVec<'a, T> {
    type Target = Vec<T>;

    fn deref<'b>(&'b self) -> &'b Vec<T> {
        &self.x
    }
}

// Prevent the inner `Vec<T>` from attempting to deallocate memory.
#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Drop for DerefVec<'a, T> {
    fn drop(&mut self) {
        self.x.len = 0;
        self.x.cap = 0;
    }
}

/// Convert a slice to a wrapper type providing a `&Vec<T>` reference.
#[unstable(feature = "collections")]
pub fn as_vec<'a, T>(x: &'a [T]) -> DerefVec<'a, T> {
    unsafe {
        DerefVec {
            x: Vec::from_raw_parts(x.as_ptr() as *mut T, x.len(), x.len()),
            l: PhantomData,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Partial vec, used for map_in_place
////////////////////////////////////////////////////////////////////////////////

/// An owned, partially type-converted vector of elements with non-zero size.
///
/// `T` and `U` must have the same, non-zero size. They must also have the same
/// alignment.
///
/// When the destructor of this struct runs, all `U`s from `start_u` (incl.) to
/// `end_u` (excl.) and all `T`s from `start_t` (incl.) to `end_t` (excl.) are
/// destructed. Additionally the underlying storage of `vec` will be freed.
struct PartialVecNonZeroSized<T,U> {
    vec: Vec<T>,

    start_u: *mut U,
    end_u: *mut U,
    start_t: *mut T,
    end_t: *mut T,

    _marker: PhantomData<U>,
}

/// An owned, partially type-converted vector of zero-sized elements.
///
/// When the destructor of this struct runs, all `num_t` `T`s and `num_u` `U`s
/// are destructed.
struct PartialVecZeroSized<T,U> {
    num_t: usize,
    num_u: usize,
    marker: PhantomData<::core::cell::Cell<(T,U)>>,
}

#[unsafe_destructor]
impl<T,U> Drop for PartialVecNonZeroSized<T,U> {
    fn drop(&mut self) {
        unsafe {
            // `vec` hasn't been modified until now. As it has a length
            // currently, this would run destructors of `T`s which might not be
            // there. So at first, set `vec`s length to `0`. This must be done
            // at first to remain memory-safe as the destructors of `U` or `T`
            // might cause unwinding where `vec`s destructor would be executed.
            self.vec.set_len(0);

            // We have instances of `U`s and `T`s in `vec`. Destruct them.
            while self.start_u != self.end_u {
                let _ = ptr::read(self.start_u); // Run a `U` destructor.
                self.start_u = self.start_u.offset(1);
            }
            while self.start_t != self.end_t {
                let _ = ptr::read(self.start_t); // Run a `T` destructor.
                self.start_t = self.start_t.offset(1);
            }
            // After this destructor ran, the destructor of `vec` will run,
            // deallocating the underlying memory.
        }
    }
}

#[unsafe_destructor]
impl<T,U> Drop for PartialVecZeroSized<T,U> {
    fn drop(&mut self) {
        unsafe {
            // Destruct the instances of `T` and `U` this struct owns.
            while self.num_t != 0 {
                let _: T = mem::uninitialized(); // Run a `T` destructor.
                self.num_t -= 1;
            }
            while self.num_u != 0 {
                let _: U = mem::uninitialized(); // Run a `U` destructor.
                self.num_u -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use core::mem::size_of;
    use core::iter::repeat;
    use test::Bencher;
    use super::as_vec;

    struct DropCounter<'a> {
        count: &'a mut u32
    }

    #[unsafe_destructor]
    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            *self.count += 1;
        }
    }

    #[test]
    fn test_as_vec() {
        let xs = [1u8, 2u8, 3u8];
        assert_eq!(&**as_vec(&xs), xs);
    }

    #[test]
    fn test_as_vec_dtor() {
        let (mut count_x, mut count_y) = (0, 0);
        {
            let xs = &[DropCounter { count: &mut count_x }, DropCounter { count: &mut count_y }];
            assert_eq!(as_vec(xs).len(), 2);
        }
        assert_eq!(count_x, 1);
        assert_eq!(count_y, 1);
    }

    #[test]
    fn test_small_vec_struct() {
        assert!(size_of::<Vec<u8>>() == size_of::<usize>() * 3);
    }

    #[test]
    fn test_double_drop() {
        struct TwoVec<T> {
            x: Vec<T>,
            y: Vec<T>
        }

        let (mut count_x, mut count_y) = (0, 0);
        {
            let mut tv = TwoVec {
                x: Vec::new(),
                y: Vec::new()
            };
            tv.x.push(DropCounter {count: &mut count_x});
            tv.y.push(DropCounter {count: &mut count_y});

            // If Vec had a drop flag, here is where it would be zeroed.
            // Instead, it should rely on its internal state to prevent
            // doing anything significant when dropped multiple times.
            drop(tv.x);

            // Here tv goes out of scope, tv.y should be dropped, but not tv.x.
        }

        assert_eq!(count_x, 1);
        assert_eq!(count_y, 1);
    }

    #[test]
    fn test_reserve() {
        let mut v = Vec::new();
        assert_eq!(v.capacity(), 0);

        v.reserve(2);
        assert!(v.capacity() >= 2);

        for i in 0..16 {
            v.push(i);
        }

        assert!(v.capacity() >= 16);
        v.reserve(16);
        assert!(v.capacity() >= 32);

        v.push(16);

        v.reserve(16);
        assert!(v.capacity() >= 33)
    }

    #[test]
    fn test_extend() {
        let mut v = Vec::new();
        let mut w = Vec::new();

        v.extend(0..3);
        for i in 0..3 { w.push(i) }

        assert_eq!(v, w);

        v.extend(3..10);
        for i in 3..10 { w.push(i) }

        assert_eq!(v, w);
    }

    #[test]
    fn test_slice_from_mut() {
        let mut values = vec![1, 2, 3, 4, 5];
        {
            let slice = &mut values[2 ..];
            assert!(slice == [3, 4, 5]);
            for p in slice {
                *p += 2;
            }
        }

        assert!(values == [1, 2, 5, 6, 7]);
    }

    #[test]
    fn test_slice_to_mut() {
        let mut values = vec![1, 2, 3, 4, 5];
        {
            let slice = &mut values[.. 2];
            assert!(slice == [1, 2]);
            for p in slice {
                *p += 1;
            }
        }

        assert!(values == [2, 3, 3, 4, 5]);
    }

    #[test]
    fn test_split_at_mut() {
        let mut values = vec![1, 2, 3, 4, 5];
        {
            let (left, right) = values.split_at_mut(2);
            {
                let left: &[_] = left;
                assert!(&left[..left.len()] == &[1, 2][]);
            }
            for p in left {
                *p += 1;
            }

            {
                let right: &[_] = right;
                assert!(&right[..right.len()] == &[3, 4, 5][]);
            }
            for p in right {
                *p += 2;
            }
        }

        assert!(values == vec![2, 3, 5, 6, 7]);
    }

    #[test]
    fn test_clone() {
        let v: Vec<i32> = vec![];
        let w = vec!(1, 2, 3);

        assert_eq!(v, v.clone());

        let z = w.clone();
        assert_eq!(w, z);
        // they should be disjoint in memory.
        assert!(w.as_ptr() != z.as_ptr())
    }

    #[test]
    fn test_clone_from() {
        let mut v = vec!();
        let three = vec!(box 1, box 2, box 3);
        let two = vec!(box 4, box 5);
        // zero, long
        v.clone_from(&three);
        assert_eq!(v, three);

        // equal
        v.clone_from(&three);
        assert_eq!(v, three);

        // long, short
        v.clone_from(&two);
        assert_eq!(v, two);

        // short, long
        v.clone_from(&three);
        assert_eq!(v, three)
    }

    #[test]
    fn test_retain() {
        let mut vec = vec![1, 2, 3, 4];
        vec.retain(|&x| x % 2 == 0);
        assert!(vec == vec![2, 4]);
    }

    #[test]
    fn zero_sized_values() {
        let mut v = Vec::new();
        assert_eq!(v.len(), 0);
        v.push(());
        assert_eq!(v.len(), 1);
        v.push(());
        assert_eq!(v.len(), 2);
        assert_eq!(v.pop(), Some(()));
        assert_eq!(v.pop(), Some(()));
        assert_eq!(v.pop(), None);

        assert_eq!(v.iter().count(), 0);
        v.push(());
        assert_eq!(v.iter().count(), 1);
        v.push(());
        assert_eq!(v.iter().count(), 2);

        for &() in &v {}

        assert_eq!(v.iter_mut().count(), 2);
        v.push(());
        assert_eq!(v.iter_mut().count(), 3);
        v.push(());
        assert_eq!(v.iter_mut().count(), 4);

        for &mut () in &mut v {}
        unsafe { v.set_len(0); }
        assert_eq!(v.iter_mut().count(), 0);
    }

    #[test]
    fn test_partition() {
        assert_eq!(vec![].into_iter().partition(|x: &i32| *x < 3), (vec![], vec![]));
        assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 4), (vec![1, 2, 3], vec![]));
        assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 2), (vec![1], vec![2, 3]));
        assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 0), (vec![], vec![1, 2, 3]));
    }

    #[test]
    fn test_zip_unzip() {
        let z1 = vec![(1, 4), (2, 5), (3, 6)];

        let (left, right): (Vec<_>, Vec<_>) = z1.iter().cloned().unzip();

        assert_eq!((1, 4), (left[0], right[0]));
        assert_eq!((2, 5), (left[1], right[1]));
        assert_eq!((3, 6), (left[2], right[2]));
    }

    #[test]
    fn test_unsafe_ptrs() {
        unsafe {
            // Test on-stack copy-from-buf.
            let a = [1, 2, 3];
            let ptr = a.as_ptr();
            let b = Vec::from_raw_buf(ptr, 3);
            assert_eq!(b, vec![1, 2, 3]);

            // Test on-heap copy-from-buf.
            let c = vec![1, 2, 3, 4, 5];
            let ptr = c.as_ptr();
            let d = Vec::from_raw_buf(ptr, 5);
            assert_eq!(d, vec![1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn test_vec_truncate_drop() {
        static mut drops: u32 = 0;
        struct Elem(i32);
        impl Drop for Elem {
            fn drop(&mut self) {
                unsafe { drops += 1; }
            }
        }

        let mut v = vec![Elem(1), Elem(2), Elem(3), Elem(4), Elem(5)];
        assert_eq!(unsafe { drops }, 0);
        v.truncate(3);
        assert_eq!(unsafe { drops }, 2);
        v.truncate(0);
        assert_eq!(unsafe { drops }, 5);
    }

    #[test]
    #[should_fail]
    fn test_vec_truncate_fail() {
        struct BadElem(i32);
        impl Drop for BadElem {
            fn drop(&mut self) {
                let BadElem(ref mut x) = *self;
                if *x == 0xbadbeef {
                    panic!("BadElem panic: 0xbadbeef")
                }
            }
        }

        let mut v = vec![BadElem(1), BadElem(2), BadElem(0xbadbeef), BadElem(4)];
        v.truncate(0);
    }

    #[test]
    fn test_index() {
        let vec = vec![1, 2, 3];
        assert!(vec[1] == 2);
    }

    #[test]
    #[should_fail]
    fn test_index_out_of_bounds() {
        let vec = vec![1, 2, 3];
        let _ = vec[3];
    }

    #[test]
    #[should_fail]
    fn test_slice_out_of_bounds_1() {
        let x = vec![1, 2, 3, 4, 5];
        &x[-1..];
    }

    #[test]
    #[should_fail]
    fn test_slice_out_of_bounds_2() {
        let x = vec![1, 2, 3, 4, 5];
        &x[..6];
    }

    #[test]
    #[should_fail]
    fn test_slice_out_of_bounds_3() {
        let x = vec![1, 2, 3, 4, 5];
        &x[-1..4];
    }

    #[test]
    #[should_fail]
    fn test_slice_out_of_bounds_4() {
        let x = vec![1, 2, 3, 4, 5];
        &x[1..6];
    }

    #[test]
    #[should_fail]
    fn test_slice_out_of_bounds_5() {
        let x = vec![1, 2, 3, 4, 5];
        &x[3..2];
    }

    #[test]
    #[should_fail]
    fn test_swap_remove_empty() {
        let mut vec= Vec::<i32>::new();
        vec.swap_remove(0);
    }

    #[test]
    fn test_move_iter_unwrap() {
        let mut vec = Vec::with_capacity(7);
        vec.push(1);
        vec.push(2);
        let ptr = vec.as_ptr();
        vec = vec.into_iter().into_inner();
        assert_eq!(vec.as_ptr(), ptr);
        assert_eq!(vec.capacity(), 7);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    #[should_fail]
    fn test_map_in_place_incompatible_types_fail() {
        let v = vec![0, 1, 2];
        v.map_in_place(|_| ());
    }

    #[test]
    fn test_map_in_place() {
        let v = vec![0, 1, 2];
        assert_eq!(v.map_in_place(|i: u32| i as i32 - 1), [-1, 0, 1]);
    }

    #[test]
    fn test_map_in_place_zero_sized() {
        let v = vec![(), ()];
        #[derive(PartialEq, Debug)]
        struct ZeroSized;
        assert_eq!(v.map_in_place(|_| ZeroSized), [ZeroSized, ZeroSized]);
    }

    #[test]
    fn test_map_in_place_zero_drop_count() {
        use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

        #[derive(Clone, PartialEq, Debug)]
        struct Nothing;
        impl Drop for Nothing { fn drop(&mut self) { } }

        #[derive(Clone, PartialEq, Debug)]
        struct ZeroSized;
        impl Drop for ZeroSized {
            fn drop(&mut self) {
                DROP_COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        }
        const NUM_ELEMENTS: usize = 2;
        static DROP_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

        let v = repeat(Nothing).take(NUM_ELEMENTS).collect::<Vec<_>>();

        DROP_COUNTER.store(0, Ordering::Relaxed);

        let v = v.map_in_place(|_| ZeroSized);
        assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 0);
        drop(v);
        assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), NUM_ELEMENTS);
    }

    #[test]
    fn test_move_items() {
        let vec = vec![1, 2, 3];
        let mut vec2 = vec![];
        for i in vec {
            vec2.push(i);
        }
        assert!(vec2 == vec![1, 2, 3]);
    }

    #[test]
    fn test_move_items_reverse() {
        let vec = vec![1, 2, 3];
        let mut vec2 = vec![];
        for i in vec.into_iter().rev() {
            vec2.push(i);
        }
        assert!(vec2 == vec![3, 2, 1]);
    }

    #[test]
    fn test_move_items_zero_sized() {
        let vec = vec![(), (), ()];
        let mut vec2 = vec![];
        for i in vec {
            vec2.push(i);
        }
        assert!(vec2 == vec![(), (), ()]);
    }

    #[test]
    fn test_drain_items() {
        let mut vec = vec![1, 2, 3];
        let mut vec2 = vec![];
        for i in vec.drain() {
            vec2.push(i);
        }
        assert_eq!(vec, []);
        assert_eq!(vec2, [ 1, 2, 3 ]);
    }

    #[test]
    fn test_drain_items_reverse() {
        let mut vec = vec![1, 2, 3];
        let mut vec2 = vec![];
        for i in vec.drain().rev() {
            vec2.push(i);
        }
        assert_eq!(vec, []);
        assert_eq!(vec2, [3, 2, 1]);
    }

    #[test]
    fn test_drain_items_zero_sized() {
        let mut vec = vec![(), (), ()];
        let mut vec2 = vec![];
        for i in vec.drain() {
            vec2.push(i);
        }
        assert_eq!(vec, []);
        assert_eq!(vec2, [(), (), ()]);
    }

    #[test]
    fn test_into_boxed_slice() {
        let xs = vec![1, 2, 3];
        let ys = xs.into_boxed_slice();
        assert_eq!(ys, [1, 2, 3]);
    }

    #[test]
    fn test_append() {
        let mut vec = vec![1, 2, 3];
        let mut vec2 = vec![4, 5, 6];
        vec.append(&mut vec2);
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(vec2, vec![]);
    }

    #[test]
    fn test_split_off() {
        let mut vec = vec![1, 2, 3, 4, 5, 6];
        let vec2 = vec.split_off(4);
        assert_eq!(vec, vec![1, 2, 3, 4]);
        assert_eq!(vec2, vec![5, 6]);
    }

    #[bench]
    fn bench_new(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<u32> = Vec::new();
            assert_eq!(v.len(), 0);
            assert_eq!(v.capacity(), 0);
        })
    }

    fn do_bench_with_capacity(b: &mut Bencher, src_len: usize) {
        b.bytes = src_len as u64;

        b.iter(|| {
            let v: Vec<u32> = Vec::with_capacity(src_len);
            assert_eq!(v.len(), 0);
            assert_eq!(v.capacity(), src_len);
        })
    }

    #[bench]
    fn bench_with_capacity_0000(b: &mut Bencher) {
        do_bench_with_capacity(b, 0)
    }

    #[bench]
    fn bench_with_capacity_0010(b: &mut Bencher) {
        do_bench_with_capacity(b, 10)
    }

    #[bench]
    fn bench_with_capacity_0100(b: &mut Bencher) {
        do_bench_with_capacity(b, 100)
    }

    #[bench]
    fn bench_with_capacity_1000(b: &mut Bencher) {
        do_bench_with_capacity(b, 1000)
    }

    fn do_bench_from_fn(b: &mut Bencher, src_len: usize) {
        b.bytes = src_len as u64;

        b.iter(|| {
            let dst = (0..src_len).collect::<Vec<_>>();
            assert_eq!(dst.len(), src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        })
    }

    #[bench]
    fn bench_from_fn_0000(b: &mut Bencher) {
        do_bench_from_fn(b, 0)
    }

    #[bench]
    fn bench_from_fn_0010(b: &mut Bencher) {
        do_bench_from_fn(b, 10)
    }

    #[bench]
    fn bench_from_fn_0100(b: &mut Bencher) {
        do_bench_from_fn(b, 100)
    }

    #[bench]
    fn bench_from_fn_1000(b: &mut Bencher) {
        do_bench_from_fn(b, 1000)
    }

    fn do_bench_from_elem(b: &mut Bencher, src_len: usize) {
        b.bytes = src_len as u64;

        b.iter(|| {
            let dst: Vec<usize> = repeat(5).take(src_len).collect();
            assert_eq!(dst.len(), src_len);
            assert!(dst.iter().all(|x| *x == 5));
        })
    }

    #[bench]
    fn bench_from_elem_0000(b: &mut Bencher) {
        do_bench_from_elem(b, 0)
    }

    #[bench]
    fn bench_from_elem_0010(b: &mut Bencher) {
        do_bench_from_elem(b, 10)
    }

    #[bench]
    fn bench_from_elem_0100(b: &mut Bencher) {
        do_bench_from_elem(b, 100)
    }

    #[bench]
    fn bench_from_elem_1000(b: &mut Bencher) {
        do_bench_from_elem(b, 1000)
    }

    fn do_bench_from_slice(b: &mut Bencher, src_len: usize) {
        let src: Vec<_> = FromIterator::from_iter(0..src_len);

        b.bytes = src_len as u64;

        b.iter(|| {
            let dst = src.clone()[..].to_vec();
            assert_eq!(dst.len(), src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        });
    }

    #[bench]
    fn bench_from_slice_0000(b: &mut Bencher) {
        do_bench_from_slice(b, 0)
    }

    #[bench]
    fn bench_from_slice_0010(b: &mut Bencher) {
        do_bench_from_slice(b, 10)
    }

    #[bench]
    fn bench_from_slice_0100(b: &mut Bencher) {
        do_bench_from_slice(b, 100)
    }

    #[bench]
    fn bench_from_slice_1000(b: &mut Bencher) {
        do_bench_from_slice(b, 1000)
    }

    fn do_bench_from_iter(b: &mut Bencher, src_len: usize) {
        let src: Vec<_> = FromIterator::from_iter(0..src_len);

        b.bytes = src_len as u64;

        b.iter(|| {
            let dst: Vec<_> = FromIterator::from_iter(src.clone().into_iter());
            assert_eq!(dst.len(), src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        });
    }

    #[bench]
    fn bench_from_iter_0000(b: &mut Bencher) {
        do_bench_from_iter(b, 0)
    }

    #[bench]
    fn bench_from_iter_0010(b: &mut Bencher) {
        do_bench_from_iter(b, 10)
    }

    #[bench]
    fn bench_from_iter_0100(b: &mut Bencher) {
        do_bench_from_iter(b, 100)
    }

    #[bench]
    fn bench_from_iter_1000(b: &mut Bencher) {
        do_bench_from_iter(b, 1000)
    }

    fn do_bench_extend(b: &mut Bencher, dst_len: usize, src_len: usize) {
        let dst: Vec<_> = FromIterator::from_iter(0..dst_len);
        let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

        b.bytes = src_len as u64;

        b.iter(|| {
            let mut dst = dst.clone();
            dst.extend(src.clone().into_iter());
            assert_eq!(dst.len(), dst_len + src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        });
    }

    #[bench]
    fn bench_extend_0000_0000(b: &mut Bencher) {
        do_bench_extend(b, 0, 0)
    }

    #[bench]
    fn bench_extend_0000_0010(b: &mut Bencher) {
        do_bench_extend(b, 0, 10)
    }

    #[bench]
    fn bench_extend_0000_0100(b: &mut Bencher) {
        do_bench_extend(b, 0, 100)
    }

    #[bench]
    fn bench_extend_0000_1000(b: &mut Bencher) {
        do_bench_extend(b, 0, 1000)
    }

    #[bench]
    fn bench_extend_0010_0010(b: &mut Bencher) {
        do_bench_extend(b, 10, 10)
    }

    #[bench]
    fn bench_extend_0100_0100(b: &mut Bencher) {
        do_bench_extend(b, 100, 100)
    }

    #[bench]
    fn bench_extend_1000_1000(b: &mut Bencher) {
        do_bench_extend(b, 1000, 1000)
    }

    fn do_bench_push_all(b: &mut Bencher, dst_len: usize, src_len: usize) {
        let dst: Vec<_> = FromIterator::from_iter(0..dst_len);
        let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

        b.bytes = src_len as u64;

        b.iter(|| {
            let mut dst = dst.clone();
            dst.push_all(&src);
            assert_eq!(dst.len(), dst_len + src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        });
    }

    #[bench]
    fn bench_push_all_0000_0000(b: &mut Bencher) {
        do_bench_push_all(b, 0, 0)
    }

    #[bench]
    fn bench_push_all_0000_0010(b: &mut Bencher) {
        do_bench_push_all(b, 0, 10)
    }

    #[bench]
    fn bench_push_all_0000_0100(b: &mut Bencher) {
        do_bench_push_all(b, 0, 100)
    }

    #[bench]
    fn bench_push_all_0000_1000(b: &mut Bencher) {
        do_bench_push_all(b, 0, 1000)
    }

    #[bench]
    fn bench_push_all_0010_0010(b: &mut Bencher) {
        do_bench_push_all(b, 10, 10)
    }

    #[bench]
    fn bench_push_all_0100_0100(b: &mut Bencher) {
        do_bench_push_all(b, 100, 100)
    }

    #[bench]
    fn bench_push_all_1000_1000(b: &mut Bencher) {
        do_bench_push_all(b, 1000, 1000)
    }

    fn do_bench_push_all_move(b: &mut Bencher, dst_len: usize, src_len: usize) {
        let dst: Vec<_> = FromIterator::from_iter(0..dst_len);
        let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

        b.bytes = src_len as u64;

        b.iter(|| {
            let mut dst = dst.clone();
            dst.extend(src.clone().into_iter());
            assert_eq!(dst.len(), dst_len + src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        });
    }

    #[bench]
    fn bench_push_all_move_0000_0000(b: &mut Bencher) {
        do_bench_push_all_move(b, 0, 0)
    }

    #[bench]
    fn bench_push_all_move_0000_0010(b: &mut Bencher) {
        do_bench_push_all_move(b, 0, 10)
    }

    #[bench]
    fn bench_push_all_move_0000_0100(b: &mut Bencher) {
        do_bench_push_all_move(b, 0, 100)
    }

    #[bench]
    fn bench_push_all_move_0000_1000(b: &mut Bencher) {
        do_bench_push_all_move(b, 0, 1000)
    }

    #[bench]
    fn bench_push_all_move_0010_0010(b: &mut Bencher) {
        do_bench_push_all_move(b, 10, 10)
    }

    #[bench]
    fn bench_push_all_move_0100_0100(b: &mut Bencher) {
        do_bench_push_all_move(b, 100, 100)
    }

    #[bench]
    fn bench_push_all_move_1000_1000(b: &mut Bencher) {
        do_bench_push_all_move(b, 1000, 1000)
    }

    fn do_bench_clone(b: &mut Bencher, src_len: usize) {
        let src: Vec<usize> = FromIterator::from_iter(0..src_len);

        b.bytes = src_len as u64;

        b.iter(|| {
            let dst = src.clone();
            assert_eq!(dst.len(), src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
        });
    }

    #[bench]
    fn bench_clone_0000(b: &mut Bencher) {
        do_bench_clone(b, 0)
    }

    #[bench]
    fn bench_clone_0010(b: &mut Bencher) {
        do_bench_clone(b, 10)
    }

    #[bench]
    fn bench_clone_0100(b: &mut Bencher) {
        do_bench_clone(b, 100)
    }

    #[bench]
    fn bench_clone_1000(b: &mut Bencher) {
        do_bench_clone(b, 1000)
    }

    fn do_bench_clone_from(b: &mut Bencher, times: usize, dst_len: usize, src_len: usize) {
        let dst: Vec<_> = FromIterator::from_iter(0..src_len);
        let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

        b.bytes = (times * src_len) as u64;

        b.iter(|| {
            let mut dst = dst.clone();

            for _ in 0..times {
                dst.clone_from(&src);

                assert_eq!(dst.len(), src_len);
                assert!(dst.iter().enumerate().all(|(i, x)| dst_len + i == *x));
            }
        });
    }

    #[bench]
    fn bench_clone_from_01_0000_0000(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 0, 0)
    }

    #[bench]
    fn bench_clone_from_01_0000_0010(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 0, 10)
    }

    #[bench]
    fn bench_clone_from_01_0000_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 0, 100)
    }

    #[bench]
    fn bench_clone_from_01_0000_1000(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 0, 1000)
    }

    #[bench]
    fn bench_clone_from_01_0010_0010(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 10, 10)
    }

    #[bench]
    fn bench_clone_from_01_0100_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 100, 100)
    }

    #[bench]
    fn bench_clone_from_01_1000_1000(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 1000, 1000)
    }

    #[bench]
    fn bench_clone_from_01_0010_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 10, 100)
    }

    #[bench]
    fn bench_clone_from_01_0100_1000(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 100, 1000)
    }

    #[bench]
    fn bench_clone_from_01_0010_0000(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 10, 0)
    }

    #[bench]
    fn bench_clone_from_01_0100_0010(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 100, 10)
    }

    #[bench]
    fn bench_clone_from_01_1000_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 1, 1000, 100)
    }

    #[bench]
    fn bench_clone_from_10_0000_0000(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 0, 0)
    }

    #[bench]
    fn bench_clone_from_10_0000_0010(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 0, 10)
    }

    #[bench]
    fn bench_clone_from_10_0000_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 0, 100)
    }

    #[bench]
    fn bench_clone_from_10_0000_1000(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 0, 1000)
    }

    #[bench]
    fn bench_clone_from_10_0010_0010(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 10, 10)
    }

    #[bench]
    fn bench_clone_from_10_0100_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 100, 100)
    }

    #[bench]
    fn bench_clone_from_10_1000_1000(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 1000, 1000)
    }

    #[bench]
    fn bench_clone_from_10_0010_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 10, 100)
    }

    #[bench]
    fn bench_clone_from_10_0100_1000(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 100, 1000)
    }

    #[bench]
    fn bench_clone_from_10_0010_0000(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 10, 0)
    }

    #[bench]
    fn bench_clone_from_10_0100_0010(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 100, 10)
    }

    #[bench]
    fn bench_clone_from_10_1000_0100(b: &mut Bencher) {
        do_bench_clone_from(b, 10, 1000, 100)
    }
}
