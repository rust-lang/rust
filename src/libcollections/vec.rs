// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A contiguous growable array type with heap-allocated contents, written
//! `Vec<T>` but pronounced 'vector.'
//!
//! Vectors have `O(1)` indexing, amortized `O(1)` push (to the end) and
//! `O(1)` pop (from the end).
//!
//! # Examples
//!
//! You can explicitly create a [`Vec<T>`] with [`new`]:
//!
//! ```
//! let v: Vec<i32> = Vec::new();
//! ```
//!
//! ...or by using the [`vec!`] macro:
//!
//! ```
//! let v: Vec<i32> = vec![];
//!
//! let v = vec![1, 2, 3, 4, 5];
//!
//! let v = vec![0; 10]; // ten zeroes
//! ```
//!
//! You can [`push`] values onto the end of a vector (which will grow the vector
//! as needed):
//!
//! ```
//! let mut v = vec![1, 2];
//!
//! v.push(3);
//! ```
//!
//! Popping values works in much the same way:
//!
//! ```
//! let mut v = vec![1, 2];
//!
//! let two = v.pop();
//! ```
//!
//! Vectors also support indexing (through the [`Index`] and [`IndexMut`] traits):
//!
//! ```
//! let mut v = vec![1, 2, 3];
//! let three = v[2];
//! v[1] = v[1] + 5;
//! ```
//!
//! [`Vec<T>`]: ../../std/vec/struct.Vec.html
//! [`new`]: ../../std/vec/struct.Vec.html#method.new
//! [`push`]: ../../std/vec/struct.Vec.html#method.push
//! [`Index`]: ../../std/ops/trait.Index.html
//! [`IndexMut`]: ../../std/ops/trait.IndexMut.html
//! [`vec!`]: ../../std/macro.vec.html

#![stable(feature = "rust1", since = "1.0.0")]

use alloc::boxed::Box;
use alloc::heap::EMPTY;
use alloc::raw_vec::RawVec;
use borrow::ToOwned;
use borrow::Cow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{self, Hash};
use core::intrinsics::{arith_offset, assume};
use core::iter::{FromIterator, FusedIterator, TrustedLen};
use core::mem;
#[cfg(not(test))]
use core::num::Float;
use core::ops::{InPlace, Index, IndexMut, Place, Placer};
use core::ops;
use core::ptr;
use core::ptr::Shared;
use core::slice;

use super::range::RangeArgument;
use Bound::{Excluded, Included, Unbounded};

/// A contiguous growable array type, written `Vec<T>` but pronounced 'vector'.
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
/// vec.extend([1, 2, 3].iter().cloned());
///
/// for x in &vec {
///     println!("{}", x);
/// }
/// assert_eq!(vec, [7, 1, 2, 3]);
/// ```
///
/// The [`vec!`] macro is provided to make initialization more convenient:
///
/// ```
/// let mut vec = vec![1, 2, 3];
/// vec.push(4);
/// assert_eq!(vec, [1, 2, 3, 4]);
/// ```
///
/// It can also initialize each element of a `Vec<T>` with a given value:
///
/// ```
/// let vec = vec![0; 5];
/// assert_eq!(vec, [0, 0, 0, 0, 0]);
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
/// while let Some(top) = stack.pop() {
///     // Prints 3, 2, 1
///     println!("{}", top);
/// }
/// ```
///
/// # Indexing
///
/// The `Vec` type allows to access values by index, because it implements the
/// [`Index`] trait. An example will be more explicit:
///
/// ```
/// let v = vec![0, 2, 4, 6];
/// println!("{}", v[1]); // it will display '2'
/// ```
///
/// However be careful: if you try to access an index which isn't in the `Vec`,
/// your software will panic! You cannot do this:
///
/// ```ignore
/// let v = vec![0, 2, 4, 6];
/// println!("{}", v[6]); // it will panic!
/// ```
///
/// In conclusion: always check if the index you want to get really exists
/// before doing it.
///
/// # Slicing
///
/// A `Vec` can be mutable. Slices, on the other hand, are read-only objects.
/// To get a slice, use `&`. Example:
///
/// ```
/// fn read_slice(slice: &[usize]) {
///     // ...
/// }
///
/// let v = vec![0, 1];
/// read_slice(&v);
///
/// // ... and that's all!
/// // you can also do it like this:
/// let x : &[usize] = &v;
/// ```
///
/// In Rust, it's more common to pass slices as arguments rather than vectors
/// when you just want to provide a read access. The same goes for [`String`] and
/// [`&str`].
///
/// # Capacity and reallocation
///
/// The capacity of a vector is the amount of space allocated for any future
/// elements that will be added onto the vector. This is not to be confused with
/// the *length* of a vector, which specifies the number of actual elements
/// within the vector. If a vector's length exceeds its capacity, its capacity
/// will automatically be increased, but its elements will have to be
/// reallocated.
///
/// For example, a vector with capacity 10 and length 0 would be an empty vector
/// with space for 10 more elements. Pushing 10 or fewer elements onto the
/// vector will not change its capacity or cause reallocation to occur. However,
/// if the vector's length is increased to 11, it will have to reallocate, which
/// can be slow. For this reason, it is recommended to use [`Vec::with_capacity`]
/// whenever possible to specify how big the vector is expected to get.
///
/// # Guarantees
///
/// Due to its incredibly fundamental nature, `Vec` makes a lot of guarantees
/// about its design. This ensures that it's as low-overhead as possible in
/// the general case, and can be correctly manipulated in primitive ways
/// by unsafe code. Note that these guarantees refer to an unqualified `Vec<T>`.
/// If additional type parameters are added (e.g. to support custom allocators),
/// overriding their defaults may change the behavior.
///
/// Most fundamentally, `Vec` is and always will be a (pointer, capacity, length)
/// triplet. No more, no less. The order of these fields is completely
/// unspecified, and you should use the appropriate methods to modify these.
/// The pointer will never be null, so this type is null-pointer-optimized.
///
/// However, the pointer may not actually point to allocated memory. In particular,
/// if you construct a `Vec` with capacity 0 via [`Vec::new`], [`vec![]`][`vec!`],
/// [`Vec::with_capacity(0)`][`Vec::with_capacity`], or by calling [`shrink_to_fit`]
/// on an empty Vec, it will not allocate memory. Similarly, if you store zero-sized
/// types inside a `Vec`, it will not allocate space for them. *Note that in this case
/// the `Vec` may not report a [`capacity`] of 0*. `Vec` will allocate if and only
/// if [`mem::size_of::<T>`]` * capacity() > 0`. In general, `Vec`'s allocation
/// details are subtle enough that it is strongly recommended that you only
/// free memory allocated by a `Vec` by creating a new `Vec` and dropping it.
///
/// If a `Vec` *has* allocated memory, then the memory it points to is on the heap
/// (as defined by the allocator Rust is configured to use by default), and its
/// pointer points to [`len`] initialized elements in order (what you would see
/// if you coerced it to a slice), followed by [`capacity`]` - `[`len`]
/// logically uninitialized elements.
///
/// `Vec` will never perform a "small optimization" where elements are actually
/// stored on the stack for two reasons:
///
/// * It would make it more difficult for unsafe code to correctly manipulate
///   a `Vec`. The contents of a `Vec` wouldn't have a stable address if it were
///   only moved, and it would be more difficult to determine if a `Vec` had
///   actually allocated memory.
///
/// * It would penalize the general case, incurring an additional branch
///   on every access.
///
/// `Vec` will never automatically shrink itself, even if completely empty. This
/// ensures no unnecessary allocations or deallocations occur. Emptying a `Vec`
/// and then filling it back up to the same [`len`] should incur no calls to
/// the allocator. If you wish to free up unused memory, use
/// [`shrink_to_fit`][`shrink_to_fit`].
///
/// [`push`] and [`insert`] will never (re)allocate if the reported capacity is
/// sufficient. [`push`] and [`insert`] *will* (re)allocate if
/// [`len`]` == `[`capacity`]. That is, the reported capacity is completely
/// accurate, and can be relied on. It can even be used to manually free the memory
/// allocated by a `Vec` if desired. Bulk insertion methods *may* reallocate, even
/// when not necessary.
///
/// `Vec` does not guarantee any particular growth strategy when reallocating
/// when full, nor when [`reserve`] is called. The current strategy is basic
/// and it may prove desirable to use a non-constant growth factor. Whatever
/// strategy is used will of course guarantee `O(1)` amortized [`push`].
///
/// `vec![x; n]`, `vec![a, b, c, d]`, and
/// [`Vec::with_capacity(n)`][`Vec::with_capacity`], will all produce a `Vec`
/// with exactly the requested capacity. If [`len`]` == `[`capacity`],
/// (as is the case for the [`vec!`] macro), then a `Vec<T>` can be converted to
/// and from a [`Box<[T]>`][owned slice] without reallocating or moving the elements.
///
/// `Vec` will not specifically overwrite any data that is removed from it,
/// but also won't specifically preserve it. Its uninitialized memory is
/// scratch space that it may use however it wants. It will generally just do
/// whatever is most efficient or otherwise easy to implement. Do not rely on
/// removed data to be erased for security purposes. Even if you drop a `Vec`, its
/// buffer may simply be reused by another `Vec`. Even if you zero a `Vec`'s memory
/// first, that may not actually happen because the optimizer does not consider
/// this a side-effect that must be preserved.
///
/// `Vec` does not currently guarantee the order in which elements are dropped
/// (the order has changed in the past, and may change again).
///
/// [`vec!`]: ../../std/macro.vec.html
/// [`Index`]: ../../std/ops/trait.Index.html
/// [`String`]: ../../std/string/struct.String.html
/// [`&str`]: ../../std/primitive.str.html
/// [`Vec::with_capacity`]: ../../std/vec/struct.Vec.html#method.with_capacity
/// [`Vec::new`]: ../../std/vec/struct.Vec.html#method.new
/// [`shrink_to_fit`]: ../../std/vec/struct.Vec.html#method.shrink_to_fit
/// [`capacity`]: ../../std/vec/struct.Vec.html#method.capacity
/// [`mem::size_of::<T>`]: ../../std/mem/fn.size_of.html
/// [`len`]: ../../std/vec/struct.Vec.html#method.len
/// [`push`]: ../../std/vec/struct.Vec.html#method.push
/// [`insert`]: ../../std/vec/struct.Vec.html#method.insert
/// [`reserve`]: ../../std/vec/struct.Vec.html#method.reserve
/// [owned slice]: ../../std/boxed/struct.Box.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Vec<T> {
    buf: RawVec<T>,
    len: usize,
}

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
    /// # #![allow(unused_mut)]
    /// let mut vec: Vec<i32> = Vec::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Vec<T> {
        Vec {
            buf: RawVec::new(),
            len: 0,
        }
    }

    /// Constructs a new, empty `Vec<T>` with the specified capacity.
    ///
    /// The vector will be able to hold exactly `capacity` elements without
    /// reallocating. If `capacity` is 0, the vector will not allocate.
    ///
    /// It is important to note that this function does not specify the *length*
    /// of the returned vector, but only the *capacity*. For an explanation of
    /// the difference between length and capacity, see *[Capacity and reallocation]*.
    ///
    /// [Capacity and reallocation]: #capacity-and-reallocation
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = Vec::with_capacity(10);
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
        Vec {
            buf: RawVec::with_capacity(capacity),
            len: 0,
        }
    }

    /// Creates a `Vec<T>` directly from the raw components of another vector.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via [`String`]/`Vec<T>`
    ///   (at least, it's highly likely to be incorrect if it wasn't).
    /// * `length` needs to be less than or equal to `capacity`.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal datastructures. For example it is **not** safe
    /// to build a `Vec<u8>` from a pointer to a C `char` array and a `size_t`.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `Vec<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// [`String`]: ../../std/string/struct.String.html
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
    ///         assert_eq!(rebuilt, [4, 5, 6]);
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Vec<T> {
        Vec {
            buf: RawVec::from_raw_parts(ptr, capacity),
            len: length,
        }
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
        self.buf.cap()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`. The collection may reserve more space to avoid
    /// frequent reallocations. After calling `reserve`, capacity will be
    /// greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient.
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
        self.buf.reserve(self.len, additional);
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to
    /// be inserted in the given `Vec<T>`. After calling `reserve_exact`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if the capacity is already sufficient.
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
        self.buf.reserve_exact(self.len, additional);
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
    /// vec.extend([1, 2, 3].iter().cloned());
    /// assert_eq!(vec.capacity(), 10);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        self.buf.shrink_to_fit(self.len);
    }

    /// Converts the vector into [`Box<[T]>`][owned slice].
    ///
    /// Note that this will drop any excess capacity. Calling this and
    /// converting back to a vector with [`into_vec`] is equivalent to calling
    /// [`shrink_to_fit`].
    ///
    /// [owned slice]: ../../std/boxed/struct.Box.html
    /// [`into_vec`]: ../../std/primitive.slice.html#method.into_vec
    /// [`shrink_to_fit`]: #method.shrink_to_fit
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec![1, 2, 3];
    ///
    /// let slice = v.into_boxed_slice();
    /// ```
    ///
    /// Any excess capacity is removed:
    ///
    /// ```
    /// let mut vec = Vec::with_capacity(10);
    /// vec.extend([1, 2, 3].iter().cloned());
    ///
    /// assert_eq!(vec.capacity(), 10);
    /// let slice = vec.into_boxed_slice();
    /// assert_eq!(slice.into_vec().capacity(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        unsafe {
            self.shrink_to_fit();
            let buf = ptr::read(&self.buf);
            mem::forget(self);
            buf.into_box()
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// The [`drain`] method can emulate `truncate`, but causes the excess
    /// elements to be returned instead of dropped.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// Truncating a five element vector to two elements:
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4, 5];
    /// vec.truncate(2);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// No truncation occurs when `len` is greater than the vector's current
    /// length:
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.truncate(8);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    ///
    /// Truncating when `len == 0` is equivalent to calling the [`clear`]
    /// method.
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.truncate(0);
    /// assert_eq!(vec, []);
    /// ```
    ///
    /// [`clear`]: #method.clear
    /// [`drain`]: #method.drain
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, len: usize) {
        unsafe {
            // drop any extra elements
            while len < self.len {
                // decrement len before the drop_in_place(), so a panic on Drop
                // doesn't re-drop the just-failed value.
                self.len -= 1;
                let len = self.len;
                ptr::drop_in_place(self.get_unchecked_mut(len));
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Write};
    /// let buffer = vec![1, 2, 3, 5, 8];
    /// io::sink().write(buffer.as_slice()).unwrap();
    /// ```
    #[inline]
    #[stable(feature = "vec_as_slice", since = "1.7.0")]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Read};
    /// let mut buffer = vec![0; 3];
    /// io::repeat(0b101).read_exact(buffer.as_mut_slice()).unwrap();
    /// ```
    #[inline]
    #[stable(feature = "vec_as_slice", since = "1.7.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
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
    /// use std::ptr;
    ///
    /// let mut vec = vec!['r', 'u', 's', 't'];
    ///
    /// unsafe {
    ///     ptr::drop_in_place(&mut vec[3]);
    ///     vec.set_len(3);
    /// }
    /// assert_eq!(vec, ['r', 'u', 's']);
    /// ```
    ///
    /// In this example, there is a memory leak since the memory locations
    /// owned by the inner vectors were not freed prior to the `set_len` call:
    ///
    /// ```
    /// let mut vec = vec![vec![1, 0, 0],
    ///                    vec![0, 1, 0],
    ///                    vec![0, 0, 1]];
    /// unsafe {
    ///     vec.set_len(0);
    /// }
    /// ```
    ///
    /// In this example, the vector gets expanded from zero to four items
    /// without any memory allocations occurring, resulting in vector
    /// values of unallocated memory:
    ///
    /// ```
    /// let mut vec: Vec<char> = Vec::new();
    ///
    /// unsafe {
    ///     vec.set_len(4);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
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
    /// assert_eq!(v, ["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v, ["baz", "qux"]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap_remove(&mut self, index: usize) -> T {
        let length = self.len();
        self.swap(index, length - 1);
        self.pop().unwrap()
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, [1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, [1, 4, 2, 3, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(index <= len);

        // space for the new element
        if len == self.buf.cap() {
            self.buf.double();
        }

        unsafe {
            // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().offset(index as isize);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy(p, p.offset(1), len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.set_len(len + 1);
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v, [1, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().offset(index as isize);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.offset(1), ptr, len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, [2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        let len = self.len();
        let mut del = 0;
        {
            let v = &mut **self;

            for i in 0..len {
                if !f(&v[i]) {
                    del += 1;
                } else if del > 0 {
                    v.swap(i - del, i);
                }
            }
        }
        if del > 0 {
            self.truncate(len - del);
        }
    }

    /// Removes consecutive elements in the vector that resolve to the same key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// ```
    #[stable(feature = "dedup_by", since = "1.16.0")]
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F) where F: FnMut(&mut T) -> K, K: PartialEq {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    /// Removes consecutive elements in the vector according to a predicate.
    ///
    /// The `same_bucket` function is passed references to two elements from the vector, and
    /// returns `true` if the elements compare equal, or `false` if they do not. Only the first
    /// of adjacent equal items is kept.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let mut vec = vec!["foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
    /// ```
    #[stable(feature = "dedup_by", since = "1.16.0")]
    pub fn dedup_by<F>(&mut self, mut same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
        unsafe {
            // Although we have a mutable reference to `self`, we cannot make
            // *arbitrary* changes. The `same_bucket` calls could panic, so we
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
            if ln <= 1 {
                return;
            }

            // Avoid bounds checks by using raw pointers.
            let p = self.as_mut_ptr();
            let mut r: usize = 1;
            let mut w: usize = 1;

            while r < ln {
                let p_r = p.offset(r as isize);
                let p_wm1 = p.offset((w - 1) as isize);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
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

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, value: T) {
        // This will panic or abort if we would allocate > isize::MAX bytes
        // or if the length increment would overflow for zero-sized types.
        if self.len == self.buf.cap() {
            self.buf.double();
        }
        unsafe {
            let end = self.as_mut_ptr().offset(self.len as isize);
            ptr::write(end, value);
            self.len += 1;
        }
    }

    /// Returns a place for insertion at the back of the `Vec`.
    ///
    /// Using this method with placement syntax is equivalent to [`push`](#method.push),
    /// but may be more efficient.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collection_placement)]
    /// #![feature(placement_in_syntax)]
    ///
    /// let mut vec = vec![1, 2];
    /// vec.place_back() <- 3;
    /// vec.place_back() <- 4;
    /// assert_eq!(&vec, &[1, 2, 3, 4]);
    /// ```
    #[unstable(feature = "collection_placement",
               reason = "placement protocol is subject to change",
               issue = "30172")]
    pub fn place_back(&mut self) -> PlaceBack<T> {
        PlaceBack { vec: self }
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
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
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// ```
    #[inline]
    #[stable(feature = "append", since = "1.4.0")]
    pub fn append(&mut self, other: &mut Self) {
        unsafe {
            self.append_elements(other.as_slice() as _);
            other.set_len(0);
        }
    }

    /// Appends elements to `Self` from other buffer.
    #[inline]
    unsafe fn append_elements(&mut self, other: *const [T]) {
        let count = (*other).len();
        self.reserve(count);
        let len = self.len();
        ptr::copy_nonoverlapping(other as *const T, self.get_unchecked_mut(len), count);
        self.len += count;
    }

    /// Creates a draining iterator that removes the specified range in the vector
    /// and yields the removed items.
    ///
    /// Note 1: The element range is removed even if the iterator is only
    /// partially consumed or not consumed at all.
    ///
    /// Note 2: It is unspecified how many elements are removed from the vector
    /// if the `Drain` value is leaked.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v, &[1]);
    /// assert_eq!(u, &[2, 3]);
    ///
    /// // A full range clears the vector
    /// v.drain(..);
    /// assert_eq!(v, &[]);
    /// ```
    #[stable(feature = "drain", since = "1.6.0")]
    pub fn drain<R>(&mut self, range: R) -> Drain<T>
        where R: RangeArgument<usize>
    {
        // Memory safety
        //
        // When the Drain is first created, it shortens the length of
        // the source vector to make sure no uninitalized or moved-from elements
        // are accessible at all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        // When finished, remaining tail of the vec is copied back to cover
        // the hole, and the vector length is restored to the new length.
        //
        let len = self.len();
        let start = match range.start() {
            Included(&n) => n,
            Excluded(&n) => n + 1,
            Unbounded    => 0,
        };
        let end = match range.end() {
            Included(&n) => n + 1,
            Excluded(&n) => n,
            Unbounded    => len,
        };
        assert!(start <= end);
        assert!(end <= len);

        unsafe {
            // set self.vec length's to start, to be safe in case Drain is leaked
            self.set_len(start);
            // Use the borrow in the IterMut to indicate borrowing behavior of the
            // whole Drain iterator (like &mut T).
            let range_slice = slice::from_raw_parts_mut(self.as_mut_ptr().offset(start as isize),
                                                        end - start);
            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: Shared::new(self as *mut _),
            }
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
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

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = vec![1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.len
    }

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
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    /// assert_eq!(vec, [1]);
    /// assert_eq!(vec2, [2, 3]);
    /// ```
    #[inline]
    #[stable(feature = "split_off", since = "1.4.0")]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "`at` out of bounds");

        let other_len = self.len - at;
        let mut other = Vec::with_capacity(other_len);

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            ptr::copy_nonoverlapping(self.as_ptr().offset(at as isize),
                                     other.as_mut_ptr(),
                                     other.len());
        }
        other
    }
}

impl<T: Clone> Vec<T> {
    /// Resizes the `Vec` in-place so that `len()` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len()`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len()`, the `Vec` is simply truncated.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec!["hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(vec, ["hello", "world", "world"]);
    ///
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.resize(2, 0);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    #[stable(feature = "vec_resize", since = "1.5.0")]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend_with_element(new_len - len, value);
        } else {
            self.truncate(new_len);
        }
    }

    /// Extend the vector by `n` additional clones of `value`.
    fn extend_with_element(&mut self, n: usize, value: T) {
        self.reserve(n);

        unsafe {
            let mut ptr = self.as_mut_ptr().offset(self.len() as isize);
            // Use SetLenOnDrop to work around bug where compiler
            // may not realize the store through `ptr` through self.set_len()
            // don't alias.
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // Write all elements except the last one
            for _ in 1..n {
                ptr::write(ptr, value.clone());
                ptr = ptr.offset(1);
                // Increment the length in every step in case clone() panics
                local_len.increment_len(1);
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                ptr::write(ptr, value);
                local_len.increment_len(1);
            }

            // len set by scope guard
        }
    }

    /// Clones and appends all elements in a slice to the `Vec`.
    ///
    /// Iterates over the slice `other`, clones each element, and then appends
    /// it to this `Vec`. The `other` vector is traversed in-order.
    ///
    /// Note that this function is same as `extend` except that it is
    /// specialized to work with slices instead. If and when Rust gets
    /// specialization this function will likely be deprecated (but still
    /// available).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec, [1, 2, 3, 4]);
    /// ```
    #[stable(feature = "vec_extend_from_slice", since = "1.6.0")]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.spec_extend(other.iter())
    }
}

// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop { local_len: *len, len: len }
    }

    #[inline]
    fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }
}

impl<'a> Drop for SetLenOnDrop<'a> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
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
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }

    /// Removes the first instance of `item` from the vector if the item exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(vec_remove_item)]
    /// let mut vec = vec![1, 2, 3, 1];
    ///
    /// vec.remove_item(&1);
    ///
    /// assert_eq!(vec, vec![2, 3, 1]);
    /// ```
    #[unstable(feature = "vec_remove_item", reason = "recently added", issue = "40062")]
    pub fn remove_item(&mut self, item: &T) -> Option<T> {
        let pos = match self.iter().position(|x| *x == *item) {
            Some(x) => x,
            None => return None,
        };
        Some(self.remove(pos))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal methods and functions
////////////////////////////////////////////////////////////////////////////////

#[doc(hidden)]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_elem<T: Clone>(elem: T, n: usize) -> Vec<T> {
    <T as SpecFromElem>::from_elem(elem, n)
}

// Specialization trait used for Vec::from_elem
trait SpecFromElem: Sized {
    fn from_elem(elem: Self, n: usize) -> Vec<Self>;
}

impl<T: Clone> SpecFromElem for T {
    default fn from_elem(elem: Self, n: usize) -> Vec<Self> {
        let mut v = Vec::with_capacity(n);
        v.extend_with_element(n, elem);
        v
    }
}

impl SpecFromElem for u8 {
    #[inline]
    fn from_elem(elem: u8, n: usize) -> Vec<u8> {
        if elem == 0 {
            return Vec {
                buf: RawVec::with_capacity_zeroed(n),
                len: n,
            }
        }
        unsafe {
            let mut v = Vec::with_capacity(n);
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
            v
        }
    }
}

macro_rules! impl_spec_from_elem {
    ($t: ty, $is_zero: expr) => {
        impl SpecFromElem for $t {
            #[inline]
            fn from_elem(elem: $t, n: usize) -> Vec<$t> {
                if $is_zero(elem) {
                    return Vec {
                        buf: RawVec::with_capacity_zeroed(n),
                        len: n,
                    }
                }
                let mut v = Vec::with_capacity(n);
                v.extend_with_element(n, elem);
                v
            }
        }
    };
}

impl_spec_from_elem!(i8, |x| x == 0);
impl_spec_from_elem!(i16, |x| x == 0);
impl_spec_from_elem!(i32, |x| x == 0);
impl_spec_from_elem!(i64, |x| x == 0);
impl_spec_from_elem!(i128, |x| x == 0);
impl_spec_from_elem!(isize, |x| x == 0);

impl_spec_from_elem!(u16, |x| x == 0);
impl_spec_from_elem!(u32, |x| x == 0);
impl_spec_from_elem!(u64, |x| x == 0);
impl_spec_from_elem!(u128, |x| x == 0);
impl_spec_from_elem!(usize, |x| x == 0);

impl_spec_from_elem!(f32, |x: f32| x == 0. && x.is_sign_positive());
impl_spec_from_elem!(f64, |x: f64| x == 0. && x.is_sign_positive());

////////////////////////////////////////////////////////////////////////////////
// Common trait implementations for Vec
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for Vec<T> {
    #[cfg(not(test))]
    fn clone(&self) -> Vec<T> {
        <[T]>::to_vec(&**self)
    }

    // HACK(japaric): with cfg(test) the inherent `[T]::to_vec` method, which is
    // required for this method definition, is not available. Instead use the
    // `slice::to_vec`  function which is only available with cfg(test)
    // NB see the slice::hack module in slice.rs for more information
    #[cfg(test)]
    fn clone(&self) -> Vec<T> {
        ::slice::to_vec(&**self)
    }

    fn clone_from(&mut self, other: &Vec<T>) {
        other.as_slice().clone_into(self);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
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
    fn index(&self, index: usize) -> &T {
        // NB built-in indexing via `&[T]`
        &(**self)[index]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IndexMut<usize> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        // NB built-in indexing via `&mut [T]`
        &mut (**self)[index]
    }
}


#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::Range<usize>> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeTo<usize>> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeFrom<usize>> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Index<ops::RangeFull> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &[T] {
        self
    }
}
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<T> ops::Index<ops::RangeInclusive<usize>> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeInclusive<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<T> ops::Index<ops::RangeToInclusive<usize>> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeToInclusive<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::Range<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeTo<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeFrom<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::IndexMut<ops::RangeFull> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, _index: ops::RangeFull) -> &mut [T] {
        self
    }
}
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<T> ops::IndexMut<ops::RangeInclusive<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeInclusive<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<T> ops::IndexMut<ops::RangeToInclusive<usize>> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeToInclusive<usize>) -> &mut [T] {
        IndexMut::index_mut(&mut **self, index)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf.ptr();
            assume(!p.is_null());
            slice::from_raw_parts(p, self.len)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf.ptr();
            assume(!ptr.is_null());
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> FromIterator<T> for Vec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Vec<T> {
        <Self as SpecExtend<T, I::IntoIter>>::from_iter(iter.into_iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

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
    fn into_iter(mut self) -> IntoIter<T> {
        unsafe {
            let begin = self.as_mut_ptr();
            assume(!begin.is_null());
            let end = if mem::size_of::<T>() == 0 {
                arith_offset(begin as *const i8, self.len() as isize) as *const T
            } else {
                begin.offset(self.len() as isize) as *const T
            };
            let cap = self.buf.cap();
            mem::forget(self);
            IntoIter {
                buf: Shared::new(begin),
                cap: cap,
                ptr: begin,
                end: end,
            }
        }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Extend<T> for Vec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        <Self as SpecExtend<T, I::IntoIter>>::spec_extend(self, iter.into_iter())
    }
}

// Specialization trait used for Vec::from_iter and Vec::extend
trait SpecExtend<T, I> {
    fn from_iter(iter: I) -> Self;
    fn spec_extend(&mut self, iter: I);
}

impl<T, I> SpecExtend<T, I> for Vec<T>
    where I: Iterator<Item=T>,
{
    default fn from_iter(mut iterator: I) -> Self {
        // Unroll the first iteration, as the vector is going to be
        // expanded on this iteration in every case when the iterable is not
        // empty, but the loop in extend_desugared() is not going to see the
        // vector being full in the few subsequent loop iterations.
        // So we get better branch prediction.
        let mut vector = match iterator.next() {
            None => return Vec::new(),
            Some(element) => {
                let (lower, _) = iterator.size_hint();
                let mut vector = Vec::with_capacity(lower.saturating_add(1));
                unsafe {
                    ptr::write(vector.get_unchecked_mut(0), element);
                    vector.set_len(1);
                }
                vector
            }
        };
        <Vec<T> as SpecExtend<T, I>>::spec_extend(&mut vector, iterator);
        vector
    }

    default fn spec_extend(&mut self, iter: I) {
        self.extend_desugared(iter)
    }
}

impl<T, I> SpecExtend<T, I> for Vec<T>
    where I: TrustedLen<Item=T>,
{
    default fn from_iter(iterator: I) -> Self {
        let mut vector = Vec::new();
        vector.spec_extend(iterator);
        vector
    }

    default fn spec_extend(&mut self, iterator: I) {
        // This is the case for a TrustedLen iterator.
        let (low, high) = iterator.size_hint();
        if let Some(high_value) = high {
            debug_assert_eq!(low, high_value,
                             "TrustedLen iterator's size hint is not exact: {:?}",
                             (low, high));
        }
        if let Some(additional) = high {
            self.reserve(additional);
            unsafe {
                let mut ptr = self.as_mut_ptr().offset(self.len() as isize);
                let mut local_len = SetLenOnDrop::new(&mut self.len);
                for element in iterator {
                    ptr::write(ptr, element);
                    ptr = ptr.offset(1);
                    // NB can't overflow since we would have had to alloc the address space
                    local_len.increment_len(1);
                }
            }
        } else {
            self.extend_desugared(iterator)
        }
    }
}

impl<T> SpecExtend<T, IntoIter<T>> for Vec<T> {
    fn from_iter(iterator: IntoIter<T>) -> Self {
        // A common case is passing a vector into a function which immediately
        // re-collects into a vector. We can short circuit this if the IntoIter
        // has not been advanced at all.
        if *iterator.buf == iterator.ptr as *mut T {
            unsafe {
                let vec = Vec::from_raw_parts(*iterator.buf as *mut T,
                                              iterator.len(),
                                              iterator.cap);
                mem::forget(iterator);
                vec
            }
        } else {
            let mut vector = Vec::new();
            vector.spec_extend(iterator);
            vector
        }
    }

    fn spec_extend(&mut self, mut iterator: IntoIter<T>) {
        unsafe {
            self.append_elements(iterator.as_slice() as _);
        }
        iterator.ptr = iterator.end;
    }
}

impl<'a, T: 'a, I> SpecExtend<&'a T, I> for Vec<T>
    where I: Iterator<Item=&'a T>,
          T: Clone,
{
    default fn from_iter(iterator: I) -> Self {
        SpecExtend::from_iter(iterator.cloned())
    }

    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.cloned())
    }
}

impl<'a, T: 'a> SpecExtend<&'a T, slice::Iter<'a, T>> for Vec<T>
    where T: Copy,
{
    fn spec_extend(&mut self, iterator: slice::Iter<'a, T>) {
        let slice = iterator.as_slice();
        self.reserve(slice.len());
        unsafe {
            let len = self.len();
            self.set_len(len + slice.len());
            self.get_unchecked_mut(len..).copy_from_slice(slice);
        }
    }
}

impl<T> Vec<T> {
    fn extend_desugared<I: Iterator<Item = T>>(&mut self, mut iterator: I) {
        // This is the case for a general iterator.
        //
        // This function should be the moral equivalent of:
        //
        //      for item in iterator {
        //          self.push(item);
        //      }
        while let Some(element) = iterator.next() {
            let len = self.len();
            if len == self.capacity() {
                let (lower, _) = iterator.size_hint();
                self.reserve(lower.saturating_add(1));
            }
            unsafe {
                ptr::write(self.get_unchecked_mut(len), element);
                // NB can't overflow since we would have had to alloc the address space
                self.set_len(len + 1);
            }
        }
    }

    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// Note 1: The element range is removed even if the iterator is not
    /// consumed until the end.
    ///
    /// Note 2: It is unspecified how many elements are removed from the vector,
    /// if the `Splice` value is leaked.
    ///
    /// Note 3: The input iterator `replace_with` is only consumed
    /// when the `Splice` value is dropped.
    ///
    /// Note 4: This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields fewer elements than `range`’s length
    /// * or the lower bound of its `size_hint()` is exact.
    ///
    /// Otherwise, a temporary vector is allocated and the tail is moved twice.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(splice)]
    /// let mut v = vec![1, 2, 3];
    /// let new = [7, 8];
    /// let u: Vec<_> = v.splice(..2, new.iter().cloned()).collect();
    /// assert_eq!(v, &[7, 8, 3]);
    /// assert_eq!(u, &[1, 2]);
    /// ```
    #[inline]
    #[unstable(feature = "splice", reason = "recently added", issue = "32310")]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<I::IntoIter>
        where R: RangeArgument<usize>, I: IntoIterator<Item=T>
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter(),
        }
    }

}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Copy> Extend<&'a T> for Vec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.spec_extend(iter.into_iter())
    }
}

macro_rules! __impl_slice_eq1 {
    ($Lhs: ty, $Rhs: ty) => {
        __impl_slice_eq1! { $Lhs, $Rhs, Sized }
    };
    ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
            #[inline]
            fn eq(&self, other: &$Rhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$Rhs) -> bool { self[..] != other[..] }
        }
    }
}

__impl_slice_eq1! { Vec<A>, Vec<B> }
__impl_slice_eq1! { Vec<A>, &'b [B] }
__impl_slice_eq1! { Vec<A>, &'b mut [B] }
__impl_slice_eq1! { Cow<'a, [A]>, &'b [B], Clone }
__impl_slice_eq1! { Cow<'a, [A]>, &'b mut [B], Clone }
__impl_slice_eq1! { Cow<'a, [A]>, Vec<B>, Clone }

macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            // NOTE: some less important impls are omitted to reduce code bloat
            __impl_slice_eq1! { Vec<A>, [B; $N] }
            __impl_slice_eq1! { Vec<A>, &'b [B; $N] }
            // __impl_slice_eq1! { Vec<A>, &'b mut [B; $N] }
            // __impl_slice_eq1! { Cow<'a, [A]>, [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b mut [B; $N], Clone }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

/// Implements comparison of vectors, lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for Vec<T> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for Vec<T> {}

/// Implements ordering of vectors, lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T> Drop for Vec<T> {
    fn drop(&mut self) {
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(&mut self[..]);
        }
        // RawVec handles deallocation
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for Vec<T> {
    /// Creates an empty `Vec<T>`.
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> AsRef<Vec<T>> for Vec<T> {
    fn as_ref(&self) -> &Vec<T> {
        self
    }
}

#[stable(feature = "vec_as_mut", since = "1.5.0")]
impl<T> AsMut<Vec<T>> for Vec<T> {
    fn as_mut(&mut self) -> &mut Vec<T> {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

#[stable(feature = "vec_as_mut", since = "1.5.0")]
impl<T> AsMut<[T]> for Vec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Clone> From<&'a [T]> for Vec<T> {
    #[cfg(not(test))]
    fn from(s: &'a [T]) -> Vec<T> {
        s.to_vec()
    }
    #[cfg(test)]
    fn from(s: &'a [T]) -> Vec<T> {
        ::slice::to_vec(s)
    }
}

#[stable(feature = "vec_from_cow_slice", since = "1.14.0")]
impl<'a, T> From<Cow<'a, [T]>> for Vec<T> where [T]: ToOwned<Owned=Vec<T>> {
    fn from(s: Cow<'a, [T]>) -> Vec<T> {
        s.into_owned()
    }
}

// note: test pulls in libstd, which causes errors here
#[cfg(not(test))]
#[stable(feature = "vec_from_box", since = "1.17.0")]
impl<T> From<Box<[T]>> for Vec<T> {
    fn from(s: Box<[T]>) -> Vec<T> {
        s.into_vec()
    }
}

#[stable(feature = "box_from_vec", since = "1.17.0")]
impl<T> Into<Box<[T]>> for Vec<T> {
    fn into(self) -> Box<[T]> {
        self.into_boxed_slice()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<&'a str> for Vec<u8> {
    fn from(s: &'a str) -> Vec<u8> {
        From::from(s.as_bytes())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Clone-on-write
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "cow_from_vec", since = "1.7.0")]
impl<'a, T: Clone> From<&'a [T]> for Cow<'a, [T]> {
    fn from(s: &'a [T]) -> Cow<'a, [T]> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_vec", since = "1.7.0")]
impl<'a, T: Clone> From<Vec<T>> for Cow<'a, [T]> {
    fn from(v: Vec<T>) -> Cow<'a, [T]> {
        Cow::Owned(v)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> FromIterator<T> for Cow<'a, [T]> where T: Clone {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Cow<'a, [T]> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////////////////////////

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the `into_iter` method on [`Vec`][`Vec`] (provided
/// by the [`IntoIterator`] trait).
///
/// [`Vec`]: struct.Vec.html
/// [`IntoIterator`]: ../../std/iter/trait.IntoIterator.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    buf: Shared<T>,
    cap: usize,
    ptr: *const T,
    end: *const T,
}

#[stable(feature = "vec_intoiter_debug", since = "1.13.0")]
impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

impl<T> IntoIter<T> {
    /// Returns the remaining items of this iterator as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = vec!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// let _ = into_iter.next().unwrap();
    /// assert_eq!(into_iter.as_slice(), &['b', 'c']);
    /// ```
    #[stable(feature = "vec_into_iter_as_slice", since = "1.15.0")]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.len())
        }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = vec!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// into_iter.as_mut_slice()[2] = 'z';
    /// assert_eq!(into_iter.next().unwrap(), 'a');
    /// assert_eq!(into_iter.next().unwrap(), 'b');
    /// assert_eq!(into_iter.next().unwrap(), 'z');
    /// ```
    #[stable(feature = "vec_into_iter_as_slice", since = "1.15.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr as *mut T, self.len())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for IntoIter<T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for IntoIter<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.ptr as *const _ == self.end {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // purposefully don't use 'ptr.offset' because for
                    // vectors with 0-size elements this would return the
                    // same pointer.
                    self.ptr = arith_offset(self.ptr as *const i8, 1) as *mut T;

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
        let exact = match self.ptr.offset_to(self.end) {
            Some(x) => x as usize,
            None => (self.end as usize).wrapping_sub(self.ptr as usize),
        };
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
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
                    self.end = arith_offset(self.end as *const i8, -1) as *mut T;

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
impl<T> ExactSizeIterator for IntoIter<T> {
    fn is_empty(&self) -> bool {
        self.ptr == self.end
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<T> FusedIterator for IntoIter<T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for IntoIter<T> {}

#[stable(feature = "vec_into_iter_clone", since = "1.8.0")]
impl<T: Clone> Clone for IntoIter<T> {
    fn clone(&self) -> IntoIter<T> {
        self.as_slice().to_owned().into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in self.by_ref() {}

        // RawVec handles deallocation
        let _ = unsafe { RawVec::from_raw_parts(self.buf.as_mut_ptr(), self.cap) };
    }
}

/// A draining iterator for `Vec<T>`.
///
/// This `struct` is created by the [`drain`] method on [`Vec`].
///
/// [`drain`]: struct.Vec.html#method.drain
/// [`Vec`]: struct.Vec.html
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<'a, T: 'a> {
    /// Index of tail to preserve
    tail_start: usize,
    /// Length of tail
    tail_len: usize,
    /// Current remaining range to remove
    iter: slice::Iter<'a, T>,
    vec: Shared<Vec<T>>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<'a, T: 'a + fmt::Debug> fmt::Debug for Drain<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Drain")
         .field(&self.iter.as_slice())
         .finish()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<'a, T: Sync> Sync for Drain<'a, T> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<'a, T: Send> Send for Drain<'a, T> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|elt| unsafe { ptr::read(elt as *const _) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|elt| unsafe { ptr::read(elt as *const _) })
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // exhaust self first
        while let Some(_) = self.next() {}

        if self.tail_len > 0 {
            unsafe {
                let source_vec = &mut *self.vec.as_mut_ptr();
                // memmove back untouched tail, update to new length
                let start = source_vec.len();
                let tail = self.tail_start;
                let src = source_vec.as_ptr().offset(tail as isize);
                let dst = source_vec.as_mut_ptr().offset(start as isize);
                ptr::copy(src, dst, self.tail_len);
                source_vec.set_len(start + self.tail_len);
            }
        }
    }
}


#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T> ExactSizeIterator for Drain<'a, T> {
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for Drain<'a, T> {}

/// A place for insertion at the back of a `Vec`.
///
/// See [`Vec::place_back`](struct.Vec.html#method.place_back) for details.
#[must_use = "places do nothing unless written to with `<-` syntax"]
#[unstable(feature = "collection_placement",
           reason = "struct name and placement protocol are subject to change",
           issue = "30172")]
#[derive(Debug)]
pub struct PlaceBack<'a, T: 'a> {
    vec: &'a mut Vec<T>,
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> Placer<T> for PlaceBack<'a, T> {
    type Place = PlaceBack<'a, T>;

    fn make_place(self) -> Self {
        // This will panic or abort if we would allocate > isize::MAX bytes
        // or if the length increment would overflow for zero-sized types.
        if self.vec.len == self.vec.buf.cap() {
            self.vec.buf.double();
        }
        self
    }
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> Place<T> for PlaceBack<'a, T> {
    fn pointer(&mut self) -> *mut T {
        unsafe { self.vec.as_mut_ptr().offset(self.vec.len as isize) }
    }
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> InPlace<T> for PlaceBack<'a, T> {
    type Owner = &'a mut T;

    unsafe fn finalize(mut self) -> &'a mut T {
        let ptr = self.pointer();
        self.vec.len += 1;
        &mut *ptr
    }
}


/// A splicing iterator for `Vec`.
///
/// This struct is created by the [`splice()`] method on [`Vec`]. See its
/// documentation for more.
///
/// [`splice()`]: struct.Vec.html#method.splice
/// [`Vec`]: struct.Vec.html
#[derive(Debug)]
#[unstable(feature = "splice", reason = "recently added", issue = "32310")]
pub struct Splice<'a, I: Iterator + 'a> {
    drain: Drain<'a, I::Item>,
    replace_with: I,
}

#[unstable(feature = "splice", reason = "recently added", issue = "32310")]
impl<'a, I: Iterator> Iterator for Splice<'a, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

#[unstable(feature = "splice", reason = "recently added", issue = "32310")]
impl<'a, I: Iterator> DoubleEndedIterator for Splice<'a, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

#[unstable(feature = "splice", reason = "recently added", issue = "32310")]
impl<'a, I: Iterator> ExactSizeIterator for Splice<'a, I> {}


#[unstable(feature = "splice", reason = "recently added", issue = "32310")]
impl<'a, I: Iterator> Drop for Splice<'a, I> {
    fn drop(&mut self) {
        // exhaust drain first
        while let Some(_) = self.drain.next() {}


        unsafe {
            if self.drain.tail_len == 0 {
                let vec = &mut *self.drain.vec.as_mut_ptr();
                vec.extend(self.replace_with.by_ref());
                return
            }

            // First fill the range left by drain().
            if !self.drain.fill(&mut self.replace_with) {
                return
            }

            // There may be more elements. Use the lower bound as an estimate.
            // FIXME: Is the upper bound a better guess? Or something else?
            let (lower_bound, _upper_bound) = self.replace_with.size_hint();
            if lower_bound > 0  {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return
                }
            }

            // Collect any remaining elements.
            // This is a zero-length vector which does not allocate if `lower_bound` was exact.
            let mut collected = self.replace_with.by_ref().collect::<Vec<I::Item>>().into_iter();
            // Now we have an exact count.
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                let filled = self.drain.fill(&mut collected);
                debug_assert!(filled);
                debug_assert_eq!(collected.len(), 0);
            }
        }
        // Let `Drain::drop` move the tail back if necessary and restore `vec.len`.
    }
}

/// Private helper methods for `Splice::drop`
impl<'a, T> Drain<'a, T> {
    /// The range from `self.vec.len` to `self.tail_start` contains elements
    /// that have been moved out.
    /// Fill that range as much as possible with new elements from the `replace_with` iterator.
    /// Return whether we filled the entire range. (`replace_with.next()` didn’t return `None`.)
    unsafe fn fill<I: Iterator<Item=T>>(&mut self, replace_with: &mut I) -> bool {
        let vec = &mut *self.vec.as_mut_ptr();
        let range_start = vec.len;
        let range_end = self.tail_start;
        let range_slice = slice::from_raw_parts_mut(
            vec.as_mut_ptr().offset(range_start as isize),
            range_end - range_start);

        for place in range_slice {
            if let Some(new_item) = replace_with.next() {
                ptr::write(place, new_item);
                vec.len += 1;
            } else {
                return false
            }
        }
        true
    }

    /// Make room for inserting more elements before the tail.
    unsafe fn move_tail(&mut self, extra_capacity: usize) {
        let vec = &mut *self.vec.as_mut_ptr();
        let used_capacity = self.tail_start + self.tail_len;
        vec.buf.reserve(used_capacity, extra_capacity);

        let new_tail_start = self.tail_start + extra_capacity;
        let src = vec.as_ptr().offset(self.tail_start as isize);
        let dst = vec.as_mut_ptr().offset(new_tail_start as isize);
        ptr::copy(src, dst, self.tail_len);
        self.tail_start = new_tail_start;
    }
}
