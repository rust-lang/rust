// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An owned, growable vector.

use core::prelude::*;

use alloc::heap::{allocate, reallocate, deallocate};
use core::raw::Slice;
use core::cmp::max;
use core::default::Default;
use core::fmt;
use core::mem;
use core::num::{CheckedMul, CheckedAdd};
use core::num;
use core::ptr;
use core::uint;

use {Collection, Mutable};
use slice::{MutableOrdVector, MutableVectorAllocating, CloneableVector};
use slice::{Items, MutItems};

/// An owned, growable vector.
///
/// # Examples
///
/// ```rust
/// # use std::vec::Vec;
/// let mut vec = Vec::new();
/// vec.push(1);
/// vec.push(2);
///
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec.get(0), &1);
///
/// assert_eq!(vec.pop(), Some(2));
/// assert_eq!(vec.len(), 1);
/// ```
///
/// The `vec!` macro is provided to make initialization more convenient:
///
/// ```rust
/// let mut vec = vec!(1, 2, 3);
/// vec.push(4);
/// assert_eq!(vec, vec!(1, 2, 3, 4));
/// ```
#[unsafe_no_drop_flag]
pub struct Vec<T> {
    len: uint,
    cap: uint,
    ptr: *mut T
}

impl<T> Vec<T> {
    /// Constructs a new, empty `Vec`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let mut vec: Vec<int> = Vec::new();
    /// ```
    #[inline]
    pub fn new() -> Vec<T> {
        Vec { len: 0, cap: 0, ptr: 0 as *mut T }
    }

    /// Constructs a new, empty `Vec` with the specified capacity.
    ///
    /// The vector will be able to hold exactly `capacity` elements without
    /// reallocating. If `capacity` is 0, the vector will not allocate.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let vec: Vec<int> = Vec::with_capacity(10);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: uint) -> Vec<T> {
        if mem::size_of::<T>() == 0 {
            Vec { len: 0, cap: uint::MAX, ptr: 0 as *mut T }
        } else if capacity == 0 {
            Vec::new()
        } else {
            let size = capacity.checked_mul(&mem::size_of::<T>())
                               .expect("capacity overflow");
            let ptr = unsafe { allocate(size, mem::min_align_of::<T>()) };
            Vec { len: 0, cap: capacity, ptr: ptr as *mut T }
        }
    }

    /// Creates and initializes a `Vec`.
    ///
    /// Creates a `Vec` of size `length` and initializes the elements to the
    /// value returned by the closure `op`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let vec = Vec::from_fn(3, |idx| idx * 2);
    /// assert_eq!(vec, vec!(0, 2, 4));
    /// ```
    #[inline]
    pub fn from_fn(length: uint, op: |uint| -> T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                let len = xs.len;
                ptr::write(xs.as_mut_slice().unsafe_mut_ref(len), op(len));
                xs.len += 1;
            }
            xs
        }
    }

    /// Create a `Vec<T>` directly from the raw constituents.
    ///
    /// This is highly unsafe:
    ///
    /// - if `ptr` is null, then `length` and `capacity` should be 0
    /// - `ptr` must point to an allocation of size `capacity`
    /// - there must be `length` valid instances of type `T` at the
    ///   beginning of that allocation
    /// - `ptr` must be allocated by the default `Vec` allocator
    pub unsafe fn from_raw_parts(length: uint, capacity: uint,
                                 ptr: *mut T) -> Vec<T> {
        Vec { len: length, cap: capacity, ptr: ptr }
    }

    /// Consumes the `Vec`, partitioning it based on a predicate.
    ///
    /// Partitions the `Vec` into two `Vec`s `(A,B)`, where all elements of `A`
    /// satisfy `f` and all elements of `B` do not. The order of elements is
    /// preserved.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3, 4);
    /// let (even, odd) = vec.partition(|&n| n % 2 == 0);
    /// assert_eq!(even, vec!(2, 4));
    /// assert_eq!(odd, vec!(1, 3));
    /// ```
    #[inline]
    pub fn partition(self, f: |&T| -> bool) -> (Vec<T>, Vec<T>) {
        let mut lefts  = Vec::new();
        let mut rights = Vec::new();

        for elt in self.move_iter() {
            if f(&elt) {
                lefts.push(elt);
            } else {
                rights.push(elt);
            }
        }

        (lefts, rights)
    }
}

impl<T: Clone> Vec<T> {
    /// Iterates over the `second` vector, copying each element and appending it to
    /// the `first`. Afterwards, the `first` is then returned for use again.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2);
    /// let vec = vec.append([3, 4]);
    /// assert_eq!(vec, vec!(1, 2, 3, 4));
    /// ```
    #[inline]
    pub fn append(mut self, second: &[T]) -> Vec<T> {
        self.push_all(second);
        self
    }

    /// Constructs a `Vec` by cloning elements of a slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let slice = [1, 2, 3];
    /// let vec = Vec::from_slice(slice);
    /// ```
    #[inline]
    pub fn from_slice(values: &[T]) -> Vec<T> {
        values.iter().map(|x| x.clone()).collect()
    }

    /// Constructs a `Vec` with copies of a value.
    ///
    /// Creates a `Vec` with `length` copies of `value`.
    ///
    /// # Example
    /// ```rust
    /// # use std::vec::Vec;
    /// let vec = Vec::from_elem(3, "hi");
    /// println!("{}", vec); // prints [hi, hi, hi]
    /// ```
    #[inline]
    pub fn from_elem(length: uint, value: T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                let len = xs.len;
                ptr::write(xs.as_mut_slice().unsafe_mut_ref(len),
                           value.clone());
                xs.len += 1;
            }
            xs
        }
    }

    /// Appends all elements in a slice to the `Vec`.
    ///
    /// Iterates over the slice `other`, clones each element, and then appends
    /// it to this `Vec`. The `other` vector is traversed in-order.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1);
    /// vec.push_all([2, 3, 4]);
    /// assert_eq!(vec, vec!(1, 2, 3, 4));
    /// ```
    #[inline]
    pub fn push_all(&mut self, other: &[T]) {
        self.extend(other.iter().map(|e| e.clone()));
    }

    /// Grows the `Vec` in-place.
    ///
    /// Adds `n` copies of `value` to the `Vec`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!("hello");
    /// vec.grow(2, &("world"));
    /// assert_eq!(vec, vec!("hello", "world", "world"));
    /// ```
    pub fn grow(&mut self, n: uint, value: &T) {
        let new_len = self.len() + n;
        self.reserve(new_len);
        let mut i: uint = 0u;

        while i < n {
            self.push((*value).clone());
            i += 1u;
        }
    }

    /// Sets the value of a vector element at a given index, growing the vector
    /// as needed.
    ///
    /// Sets the element at position `index` to `value`. If `index` is past the
    /// end of the vector, expands the vector by replicating `initval` to fill
    /// the intervening space.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!("a", "b", "c");
    /// vec.grow_set(1, &("fill"), "d");
    /// vec.grow_set(4, &("fill"), "e");
    /// assert_eq!(vec, vec!("a", "d", "c", "fill", "e"));
    /// ```
    pub fn grow_set(&mut self, index: uint, initval: &T, value: T) {
        let l = self.len();
        if index >= l {
            self.grow(index - l + 1u, initval);
        }
        *self.get_mut(index) = value;
    }

    /// Partitions a vector based on a predicate.
    ///
    /// Clones the elements of the vector, partitioning them into two `Vec`s
    /// `(A,B)`, where all elements of `A` satisfy `f` and all elements of `B`
    /// do not. The order of elements is preserved.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3, 4);
    /// let (even, odd) = vec.partitioned(|&n| n % 2 == 0);
    /// assert_eq!(even, vec!(2, 4));
    /// assert_eq!(odd, vec!(1, 3));
    /// ```
    pub fn partitioned(&self, f: |&T| -> bool) -> (Vec<T>, Vec<T>) {
        let mut lefts = Vec::new();
        let mut rights = Vec::new();

        for elt in self.iter() {
            if f(elt) {
                lefts.push(elt.clone());
            } else {
                rights.push(elt.clone());
            }
        }

        (lefts, rights)
    }
}

impl<T:Clone> Clone for Vec<T> {
    fn clone(&self) -> Vec<T> {
        let len = self.len;
        let mut vector = Vec::with_capacity(len);
        // Unsafe code so this can be optimised to a memcpy (or something
        // similarly fast) when T is Copy. LLVM is easily confused, so any
        // extra operations during the loop can prevent this optimisation
        {
            let this_slice = self.as_slice();
            while vector.len < len {
                unsafe {
                    let len = vector.len;
                    ptr::write(
                        vector.as_mut_slice().unsafe_mut_ref(len),
                        this_slice.unsafe_ref(len).clone());
                }
                vector.len += 1;
            }
        }
        vector
    }

    fn clone_from(&mut self, other: &Vec<T>) {
        // drop anything in self that will not be overwritten
        if self.len() > other.len() {
            self.truncate(other.len())
        }

        // reuse the contained values' allocations/resources.
        for (place, thing) in self.mut_iter().zip(other.iter()) {
            place.clone_from(thing)
        }

        // self.len <= other.len due to the truncate above, so the
        // slice here is always in-bounds.
        let len = self.len();
        self.extend(other.slice_from(len).iter().map(|x| x.clone()));
    }
}

impl<T> FromIterator<T> for Vec<T> {
    #[inline]
    fn from_iter<I:Iterator<T>>(mut iterator: I) -> Vec<T> {
        let (lower, _) = iterator.size_hint();
        let mut vector = Vec::with_capacity(lower);
        for element in iterator {
            vector.push(element)
        }
        vector
    }
}

impl<T> Extendable<T> for Vec<T> {
    #[inline]
    fn extend<I: Iterator<T>>(&mut self, mut iterator: I) {
        let (lower, _) = iterator.size_hint();
        self.reserve_additional(lower);
        for element in iterator {
            self.push(element)
        }
    }
}

impl<T: PartialEq> PartialEq for Vec<T> {
    #[inline]
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialOrd> PartialOrd for Vec<T> {
    #[inline]
    fn lt(&self, other: &Vec<T>) -> bool {
        self.as_slice() < other.as_slice()
    }
}

impl<T: Eq> Eq for Vec<T> {}

impl<T: PartialEq, V: Vector<T>> Equiv<V> for Vec<T> {
    #[inline]
    fn equiv(&self, other: &V) -> bool { self.as_slice() == other.as_slice() }
}

impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        self.as_slice().cmp(&other.as_slice())
    }
}

impl<T> Collection for Vec<T> {
    #[inline]
    fn len(&self) -> uint {
        self.len
    }
}

impl<T: Clone> CloneableVector<T> for Vec<T> {
    fn to_owned(&self) -> Vec<T> { self.clone() }
    fn into_owned(self) -> Vec<T> { self }
}

// FIXME: #13996: need a way to mark the return value as `noalias`
#[inline(never)]
unsafe fn alloc_or_realloc<T>(ptr: *mut T, size: uint, old_size: uint) -> *mut T {
    if old_size == 0 {
        allocate(size, mem::min_align_of::<T>()) as *mut T
    } else {
        reallocate(ptr as *mut u8, size,
                   mem::min_align_of::<T>(), old_size) as *mut T
    }
}

#[inline]
unsafe fn dealloc<T>(ptr: *mut T, len: uint) {
    if mem::size_of::<T>() != 0 {
        deallocate(ptr as *mut u8,
                   len * mem::size_of::<T>(),
                   mem::min_align_of::<T>())
    }
}

impl<T> Vec<T> {
    /// Returns the number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let vec: Vec<int> = Vec::with_capacity(10);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> uint {
        self.cap
    }

     /// Reserves capacity for at least `n` additional elements in the given
     /// vector.
     ///
     /// # Failure
     ///
     /// Fails if the new capacity overflows `uint`.
     ///
     /// # Example
     ///
     /// ```rust
     /// # use std::vec::Vec;
     /// let mut vec: Vec<int> = vec!(1);
     /// vec.reserve_additional(10);
     /// assert!(vec.capacity() >= 11);
     /// ```
    pub fn reserve_additional(&mut self, extra: uint) {
        if self.cap - self.len < extra {
            match self.len.checked_add(&extra) {
                None => fail!("Vec::reserve_additional: `uint` overflow"),
                Some(new_cap) => self.reserve(new_cap)
            }
        }
    }

    /// Reserves capacity for at least `n` elements in the given vector.
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
    /// let mut vec = vec!(1, 2, 3);
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, capacity: uint) {
        if capacity >= self.len {
            self.reserve_exact(num::next_power_of_two(capacity))
        }
    }

    /// Reserves capacity for exactly `capacity` elements in the given vector.
    ///
    /// If the capacity for `self` is already equal to or greater than the
    /// requested capacity, then no action is taken.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let mut vec: Vec<int> = Vec::with_capacity(10);
    /// vec.reserve_exact(11);
    /// assert_eq!(vec.capacity(), 11);
    /// ```
    pub fn reserve_exact(&mut self, capacity: uint) {
        if mem::size_of::<T>() == 0 { return }

        if capacity > self.cap {
            let size = capacity.checked_mul(&mem::size_of::<T>())
                               .expect("capacity overflow");
            unsafe {
                self.ptr = alloc_or_realloc(self.ptr, size,
                                            self.cap * mem::size_of::<T>());
            }
            self.cap = capacity;
        }
    }

    /// Shrink the capacity of the vector as much as possible
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// vec.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&mut self) {
        if mem::size_of::<T>() == 0 { return }

        if self.len == 0 {
            if self.cap != 0 {
                unsafe {
                    dealloc(self.ptr, self.cap)
                }
                self.cap = 0;
            }
        } else {
            unsafe {
                // Overflow check is unnecessary as the vector is already at
                // least this large.
                self.ptr = reallocate(self.ptr as *mut u8,
                                      self.len * mem::size_of::<T>(),
                                      mem::min_align_of::<T>(),
                                      self.cap * mem::size_of::<T>()) as *mut T;
            }
            self.cap = self.len;
        }
    }

    /// Remove the last element from a vector and return it, or `None` if it is
    /// empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, vec!(1, 2));
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(ptr::read(self.as_slice().unsafe_ref(self.len())))
            }
        }
    }

    /// Append an element to a vector.
    ///
    /// # Failure
    ///
    /// Fails if the number of elements in the vector overflows a `uint`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2);
    /// vec.push(3);
    /// assert_eq!(vec, vec!(1, 2, 3));
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        if mem::size_of::<T>() == 0 {
            // zero-size types consume no memory, so we can't rely on the address space running out
            self.len = self.len.checked_add(&1).expect("length overflow");
            unsafe { mem::forget(value); }
            return
        }
        if self.len == self.cap {
            let old_size = self.cap * mem::size_of::<T>();
            let size = max(old_size, 2 * mem::size_of::<T>()) * 2;
            if old_size > size { fail!("capacity overflow") }
            unsafe {
                self.ptr = alloc_or_realloc(self.ptr, size,
                                            self.cap * mem::size_of::<T>());
            }
            self.cap = max(self.cap, 2) * 2;
        }

        unsafe {
            let end = (self.ptr as *T).offset(self.len as int) as *mut T;
            ptr::write(&mut *end, value);
            self.len += 1;
        }
    }

    /// Appends one element to the vector provided. The vector itself is then
    /// returned for use again.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2);
    /// let vec = vec.append_one(3);
    /// assert_eq!(vec, vec!(1, 2, 3));
    /// ```
    #[inline]
    pub fn append_one(mut self, x: T) -> Vec<T> {
        self.push(x);
        self
    }

    /// Shorten a vector, dropping excess elements.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3, 4);
    /// vec.truncate(2);
    /// assert_eq!(vec, vec!(1, 2));
    /// ```
    pub fn truncate(&mut self, len: uint) {
        unsafe {
            // drop any extra elements
            while len < self.len {
                // decrement len before the read(), so a failure on Drop doesn't
                // re-drop the just-failed value.
                self.len -= 1;
                ptr::read(self.as_slice().unsafe_ref(self.len));
            }
        }
    }

    /// Work with `self` as a mutable slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// fn foo(slice: &mut [int]) {}
    ///
    /// let mut vec = vec!(1, 2);
    /// foo(vec.as_mut_slice());
    /// ```
    #[inline]
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        unsafe {
            mem::transmute(Slice { data: self.as_mut_ptr() as *T, len: self.len })
        }
    }

    /// Creates a consuming iterator, that is, one that moves each
    /// value out of the vector (from start to end). The vector cannot
    /// be used after calling this.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v = vec!("a".to_string(), "b".to_string());
    /// for s in v.move_iter() {
    ///     // s has type String, not &String
    ///     println!("{}", s);
    /// }
    /// ```
    #[inline]
    pub fn move_iter(self) -> MoveItems<T> {
        unsafe {
            let iter = mem::transmute(self.as_slice().iter());
            let ptr = self.ptr;
            let cap = self.cap;
            mem::forget(self);
            MoveItems { allocation: ptr, cap: cap, iter: iter }
        }
    }


    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// vector is actually the specified size.
    #[inline]
    pub unsafe fn set_len(&mut self, len: uint) {
        self.len = len;
    }

    /// Returns a reference to the value at index `index`.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.get(1) == &2);
    /// ```
    #[inline]
    pub fn get<'a>(&'a self, index: uint) -> &'a T {
        &self.as_slice()[index]
    }

    /// Returns a mutable reference to the value at index `index`.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// *vec.get_mut(1) = 4;
    /// assert_eq!(vec, vec!(1, 4, 3));
    /// ```
    #[inline]
    pub fn get_mut<'a>(&'a mut self, index: uint) -> &'a mut T {
        &mut self.as_mut_slice()[index]
    }

    /// Returns an iterator over references to the elements of the vector in
    /// order.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// for num in vec.iter() {
    ///     println!("{}", *num);
    /// }
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self) -> Items<'a,T> {
        self.as_slice().iter()
    }


    /// Returns an iterator over mutable references to the elements of the
    /// vector in order.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// for num in vec.mut_iter() {
    ///     *num = 0;
    /// }
    /// ```
    #[inline]
    pub fn mut_iter<'a>(&'a mut self) -> MutItems<'a,T> {
        self.as_mut_slice().mut_iter()
    }

    /// Sort the vector, in place, using `compare` to compare elements.
    ///
    /// This sort is `O(n log n)` worst-case and stable, but allocates
    /// approximately `2 * n`, where `n` is the length of `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec!(5i, 4, 1, 3, 2);
    /// v.sort_by(|a, b| a.cmp(b));
    /// assert_eq!(v, vec!(1, 2, 3, 4, 5));
    ///
    /// // reverse sorting
    /// v.sort_by(|a, b| b.cmp(a));
    /// assert_eq!(v, vec!(5, 4, 3, 2, 1));
    /// ```
    #[inline]
    pub fn sort_by(&mut self, compare: |&T, &T| -> Ordering) {
        self.as_mut_slice().sort_by(compare)
    }

    /// Returns a slice of self spanning the interval [`start`, `end`).
    ///
    /// # Failure
    ///
    /// Fails when the slice (or part of it) is outside the bounds of self, or when
    /// `start` > `end`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3, 4);
    /// assert!(vec.slice(0, 2) == [1, 2]);
    /// ```
    #[inline]
    pub fn slice<'a>(&'a self, start: uint, end: uint) -> &'a [T] {
        self.as_slice().slice(start, end)
    }

    /// Returns a slice containing all but the first element of the vector.
    ///
    /// # Failure
    ///
    /// Fails when the vector is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.tail() == [2, 3]);
    /// ```
    #[inline]
    pub fn tail<'a>(&'a self) -> &'a [T] {
        self.as_slice().tail()
    }

    /// Returns all but the first `n' elements of a vector.
    ///
    /// # Failure
    ///
    /// Fails when there are fewer than `n` elements in the vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3, 4);
    /// assert!(vec.tailn(2) == [3, 4]);
    /// ```
    #[inline]
    pub fn tailn<'a>(&'a self, n: uint) -> &'a [T] {
        self.as_slice().tailn(n)
    }

    /// Returns a reference to the last element of a vector, or `None` if it is
    /// empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.last() == Some(&3));
    /// ```
    #[inline]
    pub fn last<'a>(&'a self) -> Option<&'a T> {
        self.as_slice().last()
    }

    /// Returns a mutable reference to the last element of a vector, or `None`
    /// if it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// *vec.mut_last().unwrap() = 4;
    /// assert_eq!(vec, vec!(1, 2, 4));
    /// ```
    #[inline]
    pub fn mut_last<'a>(&'a mut self) -> Option<&'a mut T> {
        self.as_mut_slice().mut_last()
    }

    /// Remove an element from anywhere in the vector and return it, replacing
    /// it with the last element. This does not preserve ordering, but is O(1).
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// # Example
    /// ```rust
    /// let mut v = vec!("foo".to_string(), "bar".to_string(),
    ///                  "baz".to_string(), "qux".to_string());
    ///
    /// assert_eq!(v.swap_remove(1), Some("bar".to_string()));
    /// assert_eq!(v, vec!("foo".to_string(), "qux".to_string(), "baz".to_string()));
    ///
    /// assert_eq!(v.swap_remove(0), Some("foo".to_string()));
    /// assert_eq!(v, vec!("baz".to_string(), "qux".to_string()));
    ///
    /// assert_eq!(v.swap_remove(2), None);
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, index: uint) -> Option<T> {
        let length = self.len();
        if index < length - 1 {
            self.as_mut_slice().swap(index, length - 1);
        } else if index >= length {
            return None
        }
        self.pop()
    }

    /// Prepend an element to the vector.
    ///
    /// # Warning
    ///
    /// This is an O(n) operation as it requires copying every element in the
    /// vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// vec.unshift(4);
    /// assert_eq!(vec, vec!(4, 1, 2, 3));
    /// ```
    #[inline]
    pub fn unshift(&mut self, element: T) {
        self.insert(0, element)
    }

    /// Removes the first element from a vector and returns it, or `None` if
    /// the vector is empty.
    ///
    /// # Warning
    ///
    /// This is an O(n) operation as it requires copying every element in the
    /// vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// assert!(vec.shift() == Some(1));
    /// assert_eq!(vec, vec!(2, 3));
    /// ```
    #[inline]
    pub fn shift(&mut self) -> Option<T> {
        self.remove(0)
    }

    /// Insert an element at position `index` within the vector, shifting all
    /// elements after position i one position to the right.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds of the vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// vec.insert(1, 4);
    /// assert_eq!(vec, vec!(1, 4, 2, 3));
    /// ```
    pub fn insert(&mut self, index: uint, element: T) {
        let len = self.len();
        assert!(index <= len);
        // space for the new element
        self.reserve(len + 1);

        unsafe { // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().offset(index as int);
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

    /// Remove and return the element at position `index` within the vector,
    /// shifting all elements after position `index` one position to the left.
    /// Returns `None` if `i` is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec!(1, 2, 3);
    /// assert_eq!(v.remove(1), Some(2));
    /// assert_eq!(v, vec!(1, 3));
    ///
    /// assert_eq!(v.remove(4), None);
    /// // v is unchanged:
    /// assert_eq!(v, vec!(1, 3));
    /// ```
    pub fn remove(&mut self, index: uint) -> Option<T> {
        let len = self.len();
        if index < len {
            unsafe { // infallible
                let ret;
                {
                    // the place we are taking from.
                    let ptr = self.as_mut_ptr().offset(index as int);
                    // copy it out, unsafely having a copy of the value on
                    // the stack and in the vector at the same time.
                    ret = Some(ptr::read(ptr as *T));

                    // Shift everything down to fill in that spot.
                    ptr::copy_memory(ptr, &*ptr.offset(1), len - index - 1);
                }
                self.set_len(len - 1);
                ret
            }
        } else {
            None
        }
    }

    /// Takes ownership of the vector `other`, moving all elements into
    /// the current vector. This does not copy any elements, and it is
    /// illegal to use the `other` vector after calling this method
    /// (because it is moved here).
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(box 1);
    /// vec.push_all_move(vec!(box 2, box 3, box 4));
    /// assert_eq!(vec, vec!(box 1, box 2, box 3, box 4));
    /// ```
    #[inline]
    pub fn push_all_move(&mut self, other: Vec<T>) {
        self.extend(other.move_iter());
    }

    /// Returns a mutable slice of `self` between `start` and `end`.
    ///
    /// # Failure
    ///
    /// Fails when `start` or `end` point outside the bounds of `self`, or when
    /// `start` > `end`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3, 4);
    /// assert!(vec.mut_slice(0, 2) == [1, 2]);
    /// ```
    #[inline]
    pub fn mut_slice<'a>(&'a mut self, start: uint, end: uint)
                         -> &'a mut [T] {
        self.as_mut_slice().mut_slice(start, end)
    }

    /// Returns a mutable slice of self from `start` to the end of the vec.
    ///
    /// # Failure
    ///
    /// Fails when `start` points outside the bounds of self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3, 4);
    /// assert!(vec.mut_slice_from(2) == [3, 4]);
    /// ```
    #[inline]
    pub fn mut_slice_from<'a>(&'a mut self, start: uint) -> &'a mut [T] {
        self.as_mut_slice().mut_slice_from(start)
    }

    /// Returns a mutable slice of self from the start of the vec to `end`.
    ///
    /// # Failure
    ///
    /// Fails when `end` points outside the bounds of self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3, 4);
    /// assert!(vec.mut_slice_to(2) == [1, 2]);
    /// ```
    #[inline]
    pub fn mut_slice_to<'a>(&'a mut self, end: uint) -> &'a mut [T] {
        self.as_mut_slice().mut_slice_to(end)
    }

    /// Returns a pair of mutable slices that divides the vec at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Failure
    ///
    /// Fails if `mid > len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3, 4, 5, 6);
    ///
    /// // scoped to restrict the lifetime of the borrows
    /// {
    ///    let (left, right) = vec.mut_split_at(0);
    ///    assert!(left == &mut []);
    ///    assert!(right == &mut [1, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = vec.mut_split_at(2);
    ///     assert!(left == &mut [1, 2]);
    ///     assert!(right == &mut [3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = vec.mut_split_at(6);
    ///     assert!(left == &mut [1, 2, 3, 4, 5, 6]);
    ///     assert!(right == &mut []);
    /// }
    /// ```
    #[inline]
    pub fn mut_split_at<'a>(&'a mut self, mid: uint) -> (&'a mut [T], &'a mut [T]) {
        self.as_mut_slice().mut_split_at(mid)
    }

    /// Reverse the order of elements in a vector, in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec!(1, 2, 3);
    /// v.reverse();
    /// assert_eq!(v, vec!(3, 2, 1));
    /// ```
    #[inline]
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse()
    }

    /// Returns a slice of `self` from `start` to the end of the vec.
    ///
    /// # Failure
    ///
    /// Fails when `start` points outside the bounds of self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.slice_from(1) == [2, 3]);
    /// ```
    #[inline]
    pub fn slice_from<'a>(&'a self, start: uint) -> &'a [T] {
        self.as_slice().slice_from(start)
    }

    /// Returns a slice of self from the start of the vec to `end`.
    ///
    /// # Failure
    ///
    /// Fails when `end` points outside the bounds of self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.slice_to(2) == [1, 2]);
    /// ```
    #[inline]
    pub fn slice_to<'a>(&'a self, end: uint) -> &'a [T] {
        self.as_slice().slice_to(end)
    }

    /// Returns a slice containing all but the last element of the vector.
    ///
    /// # Failure
    ///
    /// Fails if the vector is empty
    #[inline]
    pub fn init<'a>(&'a self) -> &'a [T] {
        self.slice(0, self.len() - 1)
    }


    /// Returns an unsafe pointer to the vector's buffer.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the vector may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[inline]
    pub fn as_ptr(&self) -> *T {
        // If we have a 0-sized vector, then the base pointer should not be NULL
        // because an iterator over the slice will attempt to yield the base
        // pointer as the first element in the vector, but this will end up
        // being Some(NULL) which is optimized to None.
        if mem::size_of::<T>() == 0 {
            1 as *T
        } else {
            self.ptr as *T
        }
    }

    /// Returns a mutable unsafe pointer to the vector's buffer.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the vector may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        // see above for the 0-size check
        if mem::size_of::<T>() == 0 {
            1 as *mut T
        } else {
            self.ptr
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns false.
    /// This method operates in place and preserves the order the retained elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1i, 2, 3, 4);
    /// vec.retain(|x| x%2 == 0);
    /// assert_eq!(vec, vec!(2, 4));
    /// ```
    pub fn retain(&mut self, f: |&T| -> bool) {
        let len = self.len();
        let mut del = 0u;
        {
            let v = self.as_mut_slice();

            for i in range(0u, len) {
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

    /// Expands a vector in place, initializing the new elements to the result of a function.
    ///
    /// The vector is grown by `n` elements. The i-th new element are initialized to the value
    /// returned by `f(i)` where `i` is in the range [0, n).
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(0u, 1);
    /// vec.grow_fn(3, |i| i);
    /// assert_eq!(vec, vec!(0, 1, 0, 1, 2));
    /// ```
    pub fn grow_fn(&mut self, n: uint, f: |uint| -> T) {
        self.reserve_additional(n);
        for i in range(0u, n) {
            self.push(f(i));
        }
    }
}

impl<T:Ord> Vec<T> {
    /// Sorts the vector in place.
    ///
    /// This sort is `O(n log n)` worst-case and stable, but allocates
    /// approximately `2 * n`, where `n` is the length of `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(3i, 1, 2);
    /// vec.sort();
    /// assert_eq!(vec, vec!(1, 2, 3));
    /// ```
    pub fn sort(&mut self) {
        self.as_mut_slice().sort()
    }
}

impl<T> Mutable for Vec<T> {
    #[inline]
    fn clear(&mut self) {
        self.truncate(0)
    }
}

impl<T:PartialEq> Vec<T> {
    /// Return true if a vector contains an element with the given value
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.contains(&1));
    /// ```
    #[inline]
    pub fn contains(&self, x: &T) -> bool {
        self.as_slice().contains(x)
    }

    /// Remove consecutive repeated elements in the vector.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 2, 3, 2);
    /// vec.dedup();
    /// assert_eq!(vec, vec!(1, 2, 3, 2));
    /// ```
    pub fn dedup(&mut self) {
        unsafe {
            // Although we have a mutable reference to `self`, we cannot make
            // *arbitrary* changes. The `PartialEq` comparisons could fail, so we
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
            let p = self.as_mut_slice().as_mut_ptr();
            let mut r = 1;
            let mut w = 1;

            while r < ln {
                let p_r = p.offset(r as int);
                let p_wm1 = p.offset((w - 1) as int);
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

impl<T> Vector<T> for Vec<T> {
    /// Work with `self` as a slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// fn foo(slice: &[int]) {}
    ///
    /// let vec = vec!(1, 2);
    /// foo(vec.as_slice());
    /// ```
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe { mem::transmute(Slice { data: self.as_ptr(), len: self.len }) }
    }
}

impl<T: Clone, V: Vector<T>> Add<V, Vec<T>> for Vec<T> {
    #[inline]
    fn add(&self, rhs: &V) -> Vec<T> {
        let mut res = Vec::with_capacity(self.len() + rhs.as_slice().len());
        res.push_all(self.as_slice());
        res.push_all(rhs.as_slice());
        res
    }
}

#[unsafe_destructor]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        // This is (and should always remain) a no-op if the fields are
        // zeroed (when moving out, because of #[unsafe_no_drop_flag]).
        if self.cap != 0 {
            unsafe {
                for x in self.as_mut_slice().iter() {
                    ptr::read(x);
                }
                dealloc(self.ptr, self.cap)
            }
        }
    }
}

impl<T> Default for Vec<T> {
    fn default() -> Vec<T> {
        Vec::new()
    }
}

impl<T:fmt::Show> fmt::Show for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// An iterator that moves out of a vector.
pub struct MoveItems<T> {
    allocation: *mut T, // the block of memory allocated for the vector
    cap: uint, // the capacity of the vector
    iter: Items<'static, T>
}

impl<T> Iterator<T> for MoveItems<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            self.iter.next().map(|x| ptr::read(x))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator<T> for MoveItems<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            self.iter.next_back().map(|x| ptr::read(x))
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for MoveItems<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        if self.cap != 0 {
            for _x in *self {}
            unsafe {
                dealloc(self.allocation, self.cap);
            }
        }
    }
}

/**
 * Convert an iterator of pairs into a pair of vectors.
 *
 * Returns a tuple containing two vectors where the i-th element of the first
 * vector contains the first element of the i-th tuple of the input iterator,
 * and the i-th element of the second vector contains the second element
 * of the i-th tuple of the input iterator.
 */
pub fn unzip<T, U, V: Iterator<(T, U)>>(mut iter: V) -> (Vec<T>, Vec<U>) {
    let (lo, _) = iter.size_hint();
    let mut ts = Vec::with_capacity(lo);
    let mut us = Vec::with_capacity(lo);
    for (t, u) in iter {
        ts.push(t);
        us.push(u);
    }
    (ts, us)
}

/// Unsafe operations
pub mod raw {
    use super::Vec;
    use core::ptr;

    /// Constructs a vector from an unsafe pointer to a buffer.
    ///
    /// The elements of the buffer are copied into the vector without cloning,
    /// as if `ptr::read()` were called on them.
    #[inline]
    pub unsafe fn from_buf<T>(ptr: *T, elts: uint) -> Vec<T> {
        let mut dst = Vec::with_capacity(elts);
        dst.set_len(elts);
        ptr::copy_nonoverlapping_memory(dst.as_mut_ptr(), ptr, elts);
        dst
    }
}


#[cfg(test)]
mod tests {
    extern crate test;

    use std::prelude::*;
    use std::mem::size_of;
    use test::Bencher;
    use super::{unzip, raw, Vec};

    #[test]
    fn test_small_vec_struct() {
        assert!(size_of::<Vec<u8>>() == size_of::<uint>() * 3);
    }

    #[test]
    fn test_double_drop() {
        struct TwoVec<T> {
            x: Vec<T>,
            y: Vec<T>
        }

        struct DropCounter<'a> {
            count: &'a mut int
        }

        #[unsafe_destructor]
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                *self.count += 1;
            }
        }

        let mut count_x @ mut count_y = 0;
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
    fn test_reserve_additional() {
        let mut v = Vec::new();
        assert_eq!(v.capacity(), 0);

        v.reserve_additional(2);
        assert!(v.capacity() >= 2);

        for i in range(0, 16) {
            v.push(i);
        }

        assert!(v.capacity() >= 16);
        v.reserve_additional(16);
        assert!(v.capacity() >= 32);

        v.push(16);

        v.reserve_additional(16);
        assert!(v.capacity() >= 33)
    }

    #[test]
    fn test_extend() {
        let mut v = Vec::new();
        let mut w = Vec::new();

        v.extend(range(0, 3));
        for i in range(0, 3) { w.push(i) }

        assert_eq!(v, w);

        v.extend(range(3, 10));
        for i in range(3, 10) { w.push(i) }

        assert_eq!(v, w);
    }

    #[test]
    fn test_mut_slice_from() {
        let mut values = Vec::from_slice([1u8,2,3,4,5]);
        {
            let slice = values.mut_slice_from(2);
            assert!(slice == [3, 4, 5]);
            for p in slice.mut_iter() {
                *p += 2;
            }
        }

        assert!(values.as_slice() == [1, 2, 5, 6, 7]);
    }

    #[test]
    fn test_mut_slice_to() {
        let mut values = Vec::from_slice([1u8,2,3,4,5]);
        {
            let slice = values.mut_slice_to(2);
            assert!(slice == [1, 2]);
            for p in slice.mut_iter() {
                *p += 1;
            }
        }

        assert!(values.as_slice() == [2, 3, 3, 4, 5]);
    }

    #[test]
    fn test_mut_split_at() {
        let mut values = Vec::from_slice([1u8,2,3,4,5]);
        {
            let (left, right) = values.mut_split_at(2);
            assert!(left.slice(0, left.len()) == [1, 2]);
            for p in left.mut_iter() {
                *p += 1;
            }

            assert!(right.slice(0, right.len()) == [3, 4, 5]);
            for p in right.mut_iter() {
                *p += 2;
            }
        }

        assert!(values == Vec::from_slice([2u8, 3, 5, 6, 7]));
    }

    #[test]
    fn test_clone() {
        let v: Vec<int> = vec!();
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
    fn test_grow_fn() {
        let mut v = Vec::from_slice([0u, 1]);
        v.grow_fn(3, |i| i);
        assert!(v == Vec::from_slice([0u, 1, 0, 1, 2]));
    }

    #[test]
    fn test_retain() {
        let mut vec = Vec::from_slice([1u, 2, 3, 4]);
        vec.retain(|x| x%2 == 0);
        assert!(vec == Vec::from_slice([2u, 4]));
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

        for &() in v.iter() {}

        assert_eq!(v.mut_iter().count(), 2);
        v.push(());
        assert_eq!(v.mut_iter().count(), 3);
        v.push(());
        assert_eq!(v.mut_iter().count(), 4);

        for &() in v.mut_iter() {}
        unsafe { v.set_len(0); }
        assert_eq!(v.mut_iter().count(), 0);
    }

    #[test]
    fn test_partition() {
        assert_eq!(vec![].partition(|x: &int| *x < 3), (vec![], vec![]));
        assert_eq!(vec![1, 2, 3].partition(|x: &int| *x < 4), (vec![1, 2, 3], vec![]));
        assert_eq!(vec![1, 2, 3].partition(|x: &int| *x < 2), (vec![1], vec![2, 3]));
        assert_eq!(vec![1, 2, 3].partition(|x: &int| *x < 0), (vec![], vec![1, 2, 3]));
    }

    #[test]
    fn test_partitioned() {
        assert_eq!(vec![].partitioned(|x: &int| *x < 3), (vec![], vec![]))
        assert_eq!(vec![1, 2, 3].partitioned(|x: &int| *x < 4), (vec![1, 2, 3], vec![]));
        assert_eq!(vec![1, 2, 3].partitioned(|x: &int| *x < 2), (vec![1], vec![2, 3]));
        assert_eq!(vec![1, 2, 3].partitioned(|x: &int| *x < 0), (vec![], vec![1, 2, 3]));
    }

    #[test]
    fn test_zip_unzip() {
        let z1 = vec![(1, 4), (2, 5), (3, 6)];

        let (left, right) = unzip(z1.iter().map(|&x| x));

        let (left, right) = (left.as_slice(), right.as_slice());
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
            let b = raw::from_buf(ptr, 3u);
            assert_eq!(b, vec![1, 2, 3]);

            // Test on-heap copy-from-buf.
            let c = vec![1, 2, 3, 4, 5];
            let ptr = c.as_ptr();
            let d = raw::from_buf(ptr, 5u);
            assert_eq!(d, vec![1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn test_vec_truncate_drop() {
        static mut drops: uint = 0;
        struct Elem(int);
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
        struct BadElem(int);
        impl Drop for BadElem {
            fn drop(&mut self) {
                let BadElem(ref mut x) = *self;
                if *x == 0xbadbeef {
                    fail!("BadElem failure: 0xbadbeef")
                }
            }
        }

        let mut v = vec![BadElem(1), BadElem(2), BadElem(0xbadbeef), BadElem(4)];
        v.truncate(0);
    }

    #[bench]
    fn bench_new(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::new();
            assert_eq!(v.capacity(), 0);
            assert!(v.as_slice() == []);
        })
    }

    #[bench]
    fn bench_with_capacity_0(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::with_capacity(0);
            assert_eq!(v.capacity(), 0);
            assert!(v.as_slice() == []);
        })
    }


    #[bench]
    fn bench_with_capacity_5(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::with_capacity(5);
            assert_eq!(v.capacity(), 5);
            assert!(v.as_slice() == []);
        })
    }

    #[bench]
    fn bench_with_capacity_100(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::with_capacity(100);
            assert_eq!(v.capacity(), 100);
            assert!(v.as_slice() == []);
        })
    }

    #[bench]
    fn bench_from_fn_0(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::from_fn(0, |_| 5);
            assert!(v.as_slice() == []);
        })
    }

    #[bench]
    fn bench_from_fn_5(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::from_fn(5, |_| 5);
            assert!(v.as_slice() == [5, 5, 5, 5, 5]);
        })
    }

    #[bench]
    fn bench_from_slice_0(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::from_slice([]);
            assert!(v.as_slice() == []);
        })
    }

    #[bench]
    fn bench_from_slice_5(b: &mut Bencher) {
        b.iter(|| {
            let v: Vec<int> = Vec::from_slice([1, 2, 3, 4, 5]);
            assert!(v.as_slice() == [1, 2, 3, 4, 5]);
        })
    }

    #[bench]
    fn bench_from_iter_0(b: &mut Bencher) {
        b.iter(|| {
            let v0: Vec<int> = vec!();
            let v1: Vec<int> = FromIterator::from_iter(v0.move_iter());
            assert!(v1.as_slice() == []);
        })
    }

    #[bench]
    fn bench_from_iter_5(b: &mut Bencher) {
        b.iter(|| {
            let v0: Vec<int> = vec!(1, 2, 3, 4, 5);
            let v1: Vec<int> = FromIterator::from_iter(v0.move_iter());
            assert!(v1.as_slice() == [1, 2, 3, 4, 5]);
        })
    }

    #[bench]
    fn bench_extend_0(b: &mut Bencher) {
        b.iter(|| {
            let v0: Vec<int> = vec!();
            let mut v1: Vec<int> = vec!(1, 2, 3, 4, 5);
            v1.extend(v0.move_iter());
            assert!(v1.as_slice() == [1, 2, 3, 4, 5]);
        })
    }

    #[bench]
    fn bench_extend_5(b: &mut Bencher) {
        b.iter(|| {
            let v0: Vec<int> = vec!(1, 2, 3, 4, 5);
            let mut v1: Vec<int> = vec!(1, 2, 3, 4, 5);
            v1.extend(v0.move_iter());
            assert!(v1.as_slice() == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]);
        })
    }
}
