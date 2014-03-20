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

use cast::{forget, transmute};
use clone::Clone;
use cmp::{Ord, Eq, Ordering, TotalEq, TotalOrd};
use container::{Container, Mutable};
use default::Default;
use fmt;
use iter::{DoubleEndedIterator, FromIterator, Extendable, Iterator};
use libc::{free, c_void};
use mem::{size_of, move_val_init};
use mem;
use num;
use num::{CheckedMul, CheckedAdd};
use ops::Drop;
use option::{None, Option, Some};
use ptr::RawPtr;
use ptr;
use rt::global_heap::{malloc_raw, realloc_raw};
use raw::Slice;
use slice::{ImmutableEqVector, ImmutableVector, Items, MutItems, MutableVector};
use slice::{MutableTotalOrdVector};

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
    priv len: uint,
    priv cap: uint,
    priv ptr: *mut T
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
    pub fn with_capacity(capacity: uint) -> Vec<T> {
        if capacity == 0 {
            Vec::new()
        } else {
            let size = capacity.checked_mul(&size_of::<T>()).expect("capacity overflow");
            let ptr = unsafe { malloc_raw(size) };
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
    pub fn from_fn(length: uint, op: |uint| -> T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                move_val_init(xs.as_mut_slice().unsafe_mut_ref(xs.len), op(xs.len));
                xs.len += 1;
            }
            xs
        }
    }

    /// Consumes the `Vec`, partitioning it based on a predcate.
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
    /// Constructs a `Vec` by cloning elements of a slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::vec::Vec;
    /// let slice = [1, 2, 3];
    /// let vec = Vec::from_slice(slice);
    /// ```
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
    pub fn from_elem(length: uint, value: T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                move_val_init(xs.as_mut_slice().unsafe_mut_ref(xs.len), value.clone());
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
        for element in other.iter() {
            self.push((*element).clone())
        }
    }

    /// Grows the `Vec` in-place.
    ///
    /// Adds `n` copies of `value` to the `Vec`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!("hello");
    /// vec.grow(2, & &"world");
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
    /// vec.grow_set(1, & &"fill", "d");
    /// vec.grow_set(4, & &"fill", "e");
    /// assert_eq!(vec, vec!("a", "d", "c", "fill", "e"));
    /// ```
    pub fn grow_set(&mut self, index: uint, initval: &T, value: T) {
        let l = self.len();
        if index >= l {
            self.grow(index - l + 1u, initval);
        }
        *self.get_mut(index) = value;
    }

    /// Partitions a vector based on a predcate.
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
        let mut vector = Vec::with_capacity(self.len());
        for element in self.iter() {
            vector.push((*element).clone())
        }
        vector
    }
}

impl<T> FromIterator<T> for Vec<T> {
    fn from_iterator<I:Iterator<T>>(iterator: &mut I) -> Vec<T> {
        let (lower, _) = iterator.size_hint();
        let mut vector = Vec::with_capacity(lower);
        for element in *iterator {
            vector.push(element)
        }
        vector
    }
}

impl<T> Extendable<T> for Vec<T> {
    fn extend<I: Iterator<T>>(&mut self, iterator: &mut I) {
        let (lower, _) = iterator.size_hint();
        self.reserve_additional(lower);
        for element in *iterator {
            self.push(element)
        }
    }
}

impl<T: Eq> Eq for Vec<T> {
    #[inline]
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn lt(&self, other: &Vec<T>) -> bool {
        self.as_slice() < other.as_slice()
    }
}

impl<T: TotalEq> TotalEq for Vec<T> {
    #[inline]
    fn equals(&self, other: &Vec<T>) -> bool {
        self.as_slice().equals(&other.as_slice())
    }
}

impl<T: TotalOrd> TotalOrd for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        self.as_slice().cmp(&other.as_slice())
    }
}

impl<T> Container for Vec<T> {
    #[inline]
    fn len(&self) -> uint {
        self.len
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
        if capacity >= self.len {
            let size = capacity.checked_mul(&size_of::<T>()).expect("capacity overflow");
            self.cap = capacity;
            unsafe {
                self.ptr = realloc_raw(self.ptr as *mut u8, size) as *mut T;
            }
        }
    }

    /// Shrink the capacity of the vector to match the length
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1, 2, 3);
    /// vec.shrink_to_fit();
    /// assert_eq!(vec.capacity(), vec.len());
    /// ```
    pub fn shrink_to_fit(&mut self) {
        if self.len == 0 {
            unsafe { free(self.ptr as *mut c_void) };
            self.cap = 0;
            self.ptr = 0 as *mut T;
        } else {
            unsafe {
                // Overflow check is unnecessary as the vector is already at least this large.
                self.ptr = realloc_raw(self.ptr as *mut u8, self.len * size_of::<T>()) as *mut T;
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
        if self.len == self.cap {
            if self.cap == 0 { self.cap += 2 }
            let old_size = self.cap * size_of::<T>();
            self.cap = self.cap * 2;
            let size = old_size * 2;
            if old_size > size { fail!("capacity overflow") }
            unsafe {
                self.ptr = realloc_raw(self.ptr as *mut u8, size) as *mut T;
            }
        }

        unsafe {
            let end = (self.ptr as *T).offset(self.len as int) as *mut T;
            move_val_init(&mut *end, value);
            self.len += 1;
        }
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
            let mut i = len;
            // drop any extra elements
            while i < self.len {
                ptr::read(self.as_slice().unsafe_ref(i));
                i += 1;
            }
        }
        self.len = len;
    }

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
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        let slice = Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
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
        let slice = Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    /// Creates a consuming iterator, that is, one that moves each
    /// value out of the vector (from start to end). The vector cannot
    /// be used after calling this.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v = vec!(~"a", ~"b");
    /// for s in v.move_iter() {
    ///     // s has type ~str, not &~str
    ///     println!("{}", s);
    /// }
    /// ```
    #[inline]
    pub fn move_iter(self) -> MoveItems<T> {
        unsafe {
            let iter = transmute(self.as_slice().iter());
            let ptr = self.ptr as *mut c_void;
            forget(self);
            MoveItems { allocation: ptr, iter: iter }
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

    /// Returns a slice of `self` between `start` and `end`.
    ///
    /// # Failure
    ///
    /// Fails when `start` or `end` point outside the bounds of `self`, or when
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
    /// let mut v = vec!(~"foo", ~"bar", ~"baz", ~"qux");
    ///
    /// assert_eq!(v.swap_remove(1), Some(~"bar"));
    /// assert_eq!(v, vec!(~"foo", ~"qux", ~"baz"));
    ///
    /// assert_eq!(v.swap_remove(0), Some(~"foo"));
    /// assert_eq!(v, vec!(~"baz", ~"qux"));
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
                move_val_init(&mut *p, element);
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

    ///Apply a function to each element of a vector and return the results.
    #[inline]
    #[deprecated="Use `xs.iter().map(closure)` instead."]
    pub fn map<U>(&self, f: |t: &T| -> U) -> Vec<U> {
        self.iter().map(f).collect()
    }

    /// Takes ownership of the vector `other`, moving all elements into
    /// the current vector. This does not copy any elements, and it is
    /// illegal to use the `other` vector after calling this method
    /// (because it is moved here).
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(~1);
    /// vec.push_all_move(vec!(~2, ~3, ~4));
    /// assert_eq!(vec, vec!(~1, ~2, ~3, ~4));
    /// ```
    pub fn push_all_move(&mut self, other: Vec<T>) {
        for element in other.move_iter() {
            self.push(element)
        }
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
        self.as_slice().as_ptr()
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
        self.as_mut_slice().as_mut_ptr()
    }
}

impl<T:TotalOrd> Vec<T> {
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

impl<T:Eq> Vec<T> {
    /// Return true if a vector contains an element with the given value
    ///
    /// # Example
    ///
    /// ```rust
    /// let vec = vec!(1, 2, 3);
    /// assert!(vec.contains(&1));
    /// ```
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
            // *arbitrary* changes. The `Eq` comparisons could fail, so we
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
            // Comparing self[r] against self[w-1], tis is not a duplicate, so
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

/// Iterates over the `second` vector, copying each element and appending it to
/// the `first`. Afterwards, the `first` is then returned for use again.
///
/// # Example
///
/// ```rust
/// let vec = vec!(1, 2);
/// let vec = std::vec::append(vec, [3, 4]);
/// assert_eq!(vec, vec!(1, 2, 3, 4));
/// ```
#[inline]
pub fn append<T:Clone>(mut first: Vec<T>, second: &[T]) -> Vec<T> {
    first.push_all(second);
    first
}

/// Appends one element to the vector provided. The vector itself is then
/// returned for use again.
///
/// # Example
///
/// ```rust
/// let vec = vec!(1, 2);
/// let vec = std::vec::append_one(vec, 3);
/// assert_eq!(vec, vec!(1, 2, 3));
/// ```
#[inline]
pub fn append_one<T>(mut lhs: Vec<T>, x: T) -> Vec<T> {
    lhs.push(x);
    lhs
}

#[unsafe_destructor]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        // This is (and should always remain) a no-op if the fields are
        // zeroed (when moving out, because of #[unsafe_no_drop_flag]).
        unsafe {
            for x in self.as_mut_slice().iter() {
                ptr::read(x);
            }
            free(self.ptr as *mut c_void)
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
    priv allocation: *mut c_void, // the block of memory allocated for the vector
    priv iter: Items<'static, T>
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
        for _x in *self {}
        unsafe {
            free(self.allocation)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Vec;
    use iter::{Iterator, range, Extendable};
    use mem::{drop, size_of};
    use ops::Drop;
    use option::{Some, None};
    use ptr;

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

        v.extend(&mut range(0, 3));
        for i in range(0, 3) { w.push(i) }

        assert_eq!(v, w);

        v.extend(&mut range(3, 10));
        for i in range(3, 10) { w.push(i) }

        assert_eq!(v, w);
    }
}
