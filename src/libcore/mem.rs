// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Basic functions for dealing with memory
//!
//! This module contains functions for querying the size and alignment of
//! types, initializing and manipulating memory.

use intrinsics;
use ptr;

pub use intrinsics::transmute;

/// Moves a thing into the void.
///
/// The forget function will take ownership of the provided value but neglect
/// to run any required cleanup or memory management operations on it.
///
/// This function is the unsafe version of the `drop` function because it does
/// not run any destructors.
#[stable]
pub use intrinsics::forget;

/// Returns the size of a type in bytes.
#[inline]
#[stable]
pub fn size_of<T>() -> uint {
    unsafe { intrinsics::size_of::<T>() }
}

/// Returns the size of the type that `_val` points to in bytes.
#[inline]
#[stable]
pub fn size_of_val<T>(_val: &T) -> uint {
    size_of::<T>()
}

/// Returns the ABI-required minimum alignment of a type
///
/// This is the alignment used for struct fields. It may be smaller
/// than the preferred alignment.
#[inline]
#[stable]
pub fn min_align_of<T>() -> uint {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the ABI-required minimum alignment of the type of the value that
/// `_val` points to
#[inline]
#[stable]
pub fn min_align_of_val<T>(_val: &T) -> uint {
    min_align_of::<T>()
}

/// Returns the alignment in memory for a type.
///
/// This function will return the alignment, in bytes, of a type in memory. If
/// the alignment returned is adhered to, then the type is guaranteed to
/// function properly.
#[inline]
#[stable]
pub fn align_of<T>() -> uint {
    // We use the preferred alignment as the default alignment for a type. This
    // appears to be what clang migrated towards as well:
    //
    // http://lists.cs.uiuc.edu/pipermail/cfe-commits/Week-of-Mon-20110725/044411.html
    unsafe { intrinsics::pref_align_of::<T>() }
}

/// Returns the alignment of the type of the value that `_val` points to.
///
/// This is similar to `align_of`, but function will properly handle types such
/// as trait objects (in the future), returning the alignment for an arbitrary
/// value at runtime.
#[inline]
#[stable]
pub fn align_of_val<T>(_val: &T) -> uint {
    align_of::<T>()
}

/// Create a value initialized to zero.
///
/// This function is similar to allocating space for a a local variable and
/// zeroing it out (an unsafe operation).
///
/// Care must be taken when using this function, if the type `T` has a
/// destructor and the value falls out of scope (due to unwinding or returning)
/// before being initialized, then the destructor will run on zeroed
/// data, likely leading to crashes.
///
/// This is useful for FFI functions sometimes, but should generally be avoided.
#[inline]
#[stable]
pub unsafe fn zeroed<T>() -> T {
    intrinsics::init()
}

/// Create an uninitialized value.
///
/// Care must be taken when using this function, if the type `T` has a
/// destructor and the value falls out of scope (due to unwinding or returning)
/// before being initialized, then the destructor will run on uninitialized
/// data, likely leading to crashes.
///
/// This is useful for FFI functions sometimes, but should generally be avoided.
#[inline]
#[stable]
pub unsafe fn uninitialized<T>() -> T {
    intrinsics::uninit()
}

/// Swap the values at two mutable locations of the same type, without
/// deinitialising or copying either one.
#[inline]
#[stable]
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        // Give ourselves some scratch space to work with
        let mut t: T = uninitialized();

        // Perform the swap, `&mut` pointers never alias
        ptr::copy_nonoverlapping_memory(&mut t, &*x, 1);
        ptr::copy_nonoverlapping_memory(x, &*y, 1);
        ptr::copy_nonoverlapping_memory(y, &t, 1);

        // y and t now point to the same thing, but we need to completely forget `t`
        // because it's no longer relevant.
        forget(t);
    }
}

/// Replace the value at a mutable location with a new one, returning the old
/// value, without deinitialising or copying either one.
///
/// This is primarily used for transferring and swapping ownership of a value
/// in a mutable location. For example, this function allows consumption of
/// one field of a struct by replacing it with another value. The normal approach
/// doesn't always work:
///
/// ```rust,ignore
/// struct Buffer<T> { buf: Vec<T> }
///
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         // error: cannot move out of dereference of `&mut`-pointer
///         let buf = self.buf;
///         self.buf = Vec::new();
///         buf
///     }
/// }
/// ```
///
/// Note that `T` does not necessarily implement `Clone`, so it can't even
/// clone and reset `self.buf`. But `replace` can be used to disassociate
/// the original value of `self.buf` from `self`, allowing it to be returned:
///
/// ```rust
/// # struct Buffer<T> { buf: Vec<T> }
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         use std::mem::replace;
///         replace(&mut self.buf, Vec::new())
///     }
/// }
/// ```
#[inline]
#[stable]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/// Disposes of a value.
///
/// This function can be used to destroy any value by allowing `drop` to take
/// ownership of its argument.
///
/// # Example
///
/// ```
/// use std::cell::RefCell;
///
/// let x = RefCell::new(1i);
///
/// let mut mutable_borrow = x.borrow_mut();
/// *mutable_borrow = 1;
/// drop(mutable_borrow); // relinquish the mutable borrow on this slot
///
/// let borrow = x.borrow();
/// println!("{}", *borrow);
/// ```
#[inline]
#[stable]
pub fn drop<T>(_x: T) { }

/// Interprets `src` as `&U`, and then reads `src` without moving the contained
/// value.
///
/// This function will unsafely assume the pointer `src` is valid for
/// `sizeof(U)` bytes by transmuting `&T` to `&U` and then reading the `&U`. It
/// will also unsafely create a copy of the contained value instead of moving
/// out of `src`.
///
/// It is not a compile-time error if `T` and `U` have different sizes, but it
/// is highly encouraged to only invoke this function where `T` and `U` have the
/// same size. This function triggers undefined behavior if `U` is larger than
/// `T`.
#[inline]
#[stable]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    ptr::read(src as *const T as *const U)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
#[unstable = "this function may be removed in the future due to its \
              questionable utility"]
pub unsafe fn copy_lifetime<'a, S, T:'a>(_ptr: &'a S, ptr: &T) -> &'a T {
    transmute(ptr)
}

/// Transforms lifetime of the second mutable pointer to match the first.
#[inline]
#[unstable = "this function may be removed in the future due to its \
              questionable utility"]
pub unsafe fn copy_mut_lifetime<'a, S, T:'a>(_ptr: &'a mut S,
                                          ptr: &mut T) -> &'a mut T {
    transmute(ptr)
}
