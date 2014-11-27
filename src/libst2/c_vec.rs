// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Library to interface with chunks of memory allocated in C.
//!
//! It is often desirable to safely interface with memory allocated from C,
//! encapsulating the unsafety into allocation and destruction time.  Indeed,
//! allocating memory externally is currently the only way to give Rust shared
//! mut state with C programs that keep their own references; vectors are
//! unsuitable because they could be reallocated or moved at any time, and
//! importing C memory into a vector takes a one-time snapshot of the memory.
//!
//! This module simplifies the usage of such external blocks of memory.  Memory
//! is encapsulated into an opaque object after creation; the lifecycle of the
//! memory can be optionally managed by Rust, if an appropriate destructor
//! closure is provided.  Safety is ensured by bounds-checking accesses, which
//! are marshalled through get and set functions.
//!
//! There are three unsafe functions: the two constructors, and the
//! unwrap method. The constructors are unsafe for the
//! obvious reason (they act on a pointer that cannot be checked inside the
//! method), but `unwrap()` is somewhat more subtle in its unsafety.
//! It returns the contained pointer, but at the same time destroys the CVec
//! without running its destructor. This can be used to pass memory back to
//! C, but care must be taken that the ownership of underlying resources are
//! handled correctly, i.e. that allocated memory is eventually freed
//! if necessary.

#![experimental]

use kinds::Send;
use mem;
use ops::Drop;
use option::{Option, Some, None};
use ptr::RawPtr;
use ptr;
use raw;
use slice::AsSlice;

/// The type representing a foreign chunk of memory
pub struct CVec<T> {
    base: *mut T,
    len: uint,
    dtor: Option<proc():Send>,
}

#[unsafe_destructor]
impl<T> Drop for CVec<T> {
    fn drop(&mut self) { unimplemented!() }
}

impl<T> CVec<T> {
    /// Create a `CVec` from a raw pointer to a buffer with a given length.
    ///
    /// Panics if the given pointer is null. The returned vector will not attempt
    /// to deallocate the vector when dropped.
    ///
    /// # Arguments
    ///
    /// * base - A raw pointer to a buffer
    /// * len - The number of elements in the buffer
    pub unsafe fn new(base: *mut T, len: uint) -> CVec<T> { unimplemented!() }

    /// Create a `CVec` from a foreign buffer, with a given length,
    /// and a function to run upon destruction.
    ///
    /// Panics if the given pointer is null.
    ///
    /// # Arguments
    ///
    /// * base - A foreign pointer to a buffer
    /// * len - The number of elements in the buffer
    /// * dtor - A proc to run when the value is destructed, useful
    ///          for freeing the buffer, etc.
    pub unsafe fn new_with_dtor(base: *mut T, len: uint,
                                dtor: proc():Send) -> CVec<T> { unimplemented!() }

    /// View the stored data as a mutable slice.
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] { unimplemented!() }

    /// Retrieves an element at a given index, returning `None` if the requested
    /// index is greater than the length of the vector.
    pub fn get<'a>(&'a self, ofs: uint) -> Option<&'a T> { unimplemented!() }

    /// Retrieves a mutable element at a given index, returning `None` if the
    /// requested index is greater than the length of the vector.
    pub fn get_mut<'a>(&'a mut self, ofs: uint) -> Option<&'a mut T> { unimplemented!() }

    /// Unwrap the pointer without running the destructor
    ///
    /// This method retrieves the underlying pointer, and in the process
    /// destroys the CVec but without running the destructor. A use case
    /// would be transferring ownership of the buffer to a C function, as
    /// in this case you would not want to run the destructor.
    ///
    /// Note that if you want to access the underlying pointer without
    /// cancelling the destructor, you can simply call `transmute` on the return
    /// value of `get(0)`.
    pub unsafe fn unwrap(mut self) -> *mut T { unimplemented!() }

    /// Returns the number of items in this vector.
    pub fn len(&self) -> uint { unimplemented!() }

    /// Returns whether this vector is empty.
    pub fn is_empty(&self) -> bool { unimplemented!() }
}

impl<T> AsSlice<T> for CVec<T> {
    /// View the stored data as a slice.
    fn as_slice<'a>(&'a self) -> &'a [T] { unimplemented!() }
}
