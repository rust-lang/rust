// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Atomic types
//!
//! Atomic types provide primitive shared-memory communication between
//! threads, and are the building blocks of other concurrent
//! types.
//!
//! This module defines atomic versions of a select number of primitive
//! types, including `AtomicBool`, `AtomicIsize`, and `AtomicUsize`.
//! Atomic types present operations that, when used correctly, synchronize
//! updates between threads.
//!
//! Each method takes an `Ordering` which represents the strength of
//! the memory barrier for that operation. These orderings are the
//! same as [LLVM atomic orderings][1]. For more information see the [nomicon][2].
//!
//! [1]: http://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations
//! [2]: ../../../nomicon/atomics.html
//!
//! Atomic variables are safe to share between threads (they implement `Sync`)
//! but they do not themselves provide the mechanism for sharing and follow the
//! [threading model](../../../std/thread/index.html#the-threading-model) of rust.
//! The most common way to share an atomic variable is to put it into an `Arc` (an
//! atomically-reference-counted shared pointer).
//!
//! Most atomic types may be stored in static variables, initialized using
//! the provided static initializers like `ATOMIC_BOOL_INIT`. Atomic statics
//! are often used for lazy global initialization.
//!
//!
//! # Examples
//!
//! A simple spinlock:
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use std::thread;
//!
//! fn main() {
//!     let spinlock = Arc::new(AtomicUsize::new(1));
//!
//!     let spinlock_clone = spinlock.clone();
//!     let thread = thread::spawn(move|| {
//!         spinlock_clone.store(0, Ordering::SeqCst);
//!     });
//!
//!     // Wait for the other thread to release the lock
//!     while spinlock.load(Ordering::SeqCst) != 0 {}
//!
//!     if let Err(panic) = thread.join() {
//!         println!("Thread had an error: {:?}", panic);
//!     }
//! }
//! ```
//!
//! Keep a global count of live threads:
//!
//! ```
//! use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
//!
//! static GLOBAL_THREAD_COUNT: AtomicUsize = ATOMIC_USIZE_INIT;
//!
//! let old_thread_count = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
//! println!("live threads: {}", old_thread_count + 1);
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![cfg_attr(not(target_has_atomic = "8"), allow(dead_code))]
#![cfg_attr(not(target_has_atomic = "8"), allow(unused_imports))]

use self::Ordering::*;

use intrinsics;
use cell::UnsafeCell;
use fmt;

/// A boolean type which can be safely shared between threads.
///
/// This type has the same in-memory representation as a `bool`.
#[cfg(target_has_atomic = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicBool {
    v: UnsafeCell<u8>,
}

#[cfg(target_has_atomic = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
impl Default for AtomicBool {
    /// Creates an `AtomicBool` initialized to `false`.
    fn default() -> Self {
        Self::new(false)
    }
}

// Send is implicitly implemented for AtomicBool.
#[cfg(target_has_atomic = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for AtomicBool {}

/// A raw pointer type which can be safely shared between threads.
///
/// This type has the same in-memory representation as a `*mut T`.
#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicPtr<T> {
    p: UnsafeCell<*mut T>,
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for AtomicPtr<T> {
    /// Creates a null `AtomicPtr<T>`.
    fn default() -> AtomicPtr<T> {
        AtomicPtr::new(::ptr::null_mut())
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T> Send for AtomicPtr<T> {}
#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T> Sync for AtomicPtr<T> {}

/// Atomic memory orderings
///
/// Memory orderings limit the ways that both the compiler and CPU may reorder
/// instructions around atomic operations. At its most restrictive,
/// "sequentially consistent" atomics allow neither reads nor writes
/// to be moved either before or after the atomic operation; on the other end
/// "relaxed" atomics allow all reorderings.
///
/// Rust's memory orderings are [the same as
/// LLVM's](http://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations).
///
/// For more information see the [nomicon][1].
/// [1]: ../../../nomicon/atomics.html
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Debug)]
pub enum Ordering {
    /// No ordering constraints, only atomic operations. Corresponds to LLVM's
    /// `Monotonic` ordering.
    #[stable(feature = "rust1", since = "1.0.0")]
    Relaxed,
    /// When coupled with a store, all previous writes become visible
    /// to the other threads that perform a load with `Acquire` ordering
    /// on the same value.
    #[stable(feature = "rust1", since = "1.0.0")]
    Release,
    /// When coupled with a load, all subsequent loads will see data
    /// written before a store with `Release` ordering on the same value
    /// in other threads.
    #[stable(feature = "rust1", since = "1.0.0")]
    Acquire,
    /// When coupled with a load, uses `Acquire` ordering, and with a store
    /// `Release` ordering.
    #[stable(feature = "rust1", since = "1.0.0")]
    AcqRel,
    /// Like `AcqRel` with the additional guarantee that all threads see all
    /// sequentially consistent operations in the same order.
    #[stable(feature = "rust1", since = "1.0.0")]
    SeqCst,
    // Prevent exhaustive matching to allow for future extension
    #[doc(hidden)]
    #[unstable(feature = "future_atomic_orderings", issue = "0")]
    __Nonexhaustive,
}

/// An `AtomicBool` initialized to `false`.
#[cfg(target_has_atomic = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
pub const ATOMIC_BOOL_INIT: AtomicBool = AtomicBool::new(false);

#[cfg(target_has_atomic = "8")]
impl AtomicBool {
    /// Creates a new `AtomicBool`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicBool;
    ///
    /// let atomic_true  = AtomicBool::new(true);
    /// let atomic_false = AtomicBool::new(false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn new(v: bool) -> AtomicBool {
        AtomicBool { v: UnsafeCell::new(v as u8) }
    }

    /// Returns a mutable reference to the underlying `bool`.
    ///
    /// This is safe because the mutable reference guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let mut some_bool = AtomicBool::new(true);
    /// assert_eq!(*some_bool.get_mut(), true);
    /// *some_bool.get_mut() = false;
    /// assert_eq!(some_bool.load(Ordering::SeqCst), false);
    /// ```
    #[inline]
    #[stable(feature = "atomic_access", since = "1.15.0")]
    pub fn get_mut(&mut self) -> &mut bool {
        unsafe { &mut *(self.v.get() as *mut bool) }
    }

    /// Consumes the atomic and returns the contained value.
    ///
    /// This is safe because passing `self` by value guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicBool;
    ///
    /// let some_bool = AtomicBool::new(true);
    /// assert_eq!(some_bool.into_inner(), true);
    /// ```
    #[inline]
    #[stable(feature = "atomic_access", since = "1.15.0")]
    pub fn into_inner(self) -> bool {
        unsafe { self.v.into_inner() != 0 }
    }

    /// Loads a value from the bool.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation.
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Release`] or [`AcqRel`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`AcqRel`]: enum.Ordering.html#variant.Release
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let some_bool = AtomicBool::new(true);
    ///
    /// assert_eq!(some_bool.load(Ordering::Relaxed), true);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn load(&self, order: Ordering) -> bool {
        unsafe { atomic_load(self.v.get(), order) != 0 }
    }

    /// Stores a value into the bool.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation.
    ///
    /// [`Ordering`]: enum.Ordering.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let some_bool = AtomicBool::new(true);
    ///
    /// some_bool.store(false, Ordering::Relaxed);
    /// assert_eq!(some_bool.load(Ordering::Relaxed), false);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Acquire` or `AcqRel`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn store(&self, val: bool, order: Ordering) {
        unsafe {
            atomic_store(self.v.get(), val as u8, order);
        }
    }

    /// Stores a value into the bool, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation.
    ///
    /// [`Ordering`]: enum.Ordering.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let some_bool = AtomicBool::new(true);
    ///
    /// assert_eq!(some_bool.swap(false, Ordering::Relaxed), true);
    /// assert_eq!(some_bool.load(Ordering::Relaxed), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
        unsafe { atomic_swap(self.v.get(), val as u8, order) != 0 }
    }

    /// Stores a value into the `bool` if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation.
    ///
    /// [`Ordering`]: enum.Ordering.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let some_bool = AtomicBool::new(true);
    ///
    /// assert_eq!(some_bool.compare_and_swap(true, false, Ordering::Relaxed), true);
    /// assert_eq!(some_bool.load(Ordering::Relaxed), false);
    ///
    /// assert_eq!(some_bool.compare_and_swap(true, true, Ordering::Relaxed), false);
    /// assert_eq!(some_bool.load(Ordering::Relaxed), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, current: bool, new: bool, order: Ordering) -> bool {
        match self.compare_exchange(current, new, order, strongest_failure_ordering(order)) {
            Ok(x) => x,
            Err(x) => x,
        }
    }

    /// Stores a value into the `bool` if the current value is the same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was written and containing
    /// the previous value. On success this value is guaranteed to be equal to `current`.
    ///
    /// `compare_exchange` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if the
    /// operation succeeds while the second describes the required ordering when the
    /// operation fails. The failure ordering can't be [`Release`] or [`AcqRel`] and must
    /// be equivalent or weaker than the success ordering.
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`AcqRel`]: enum.Ordering.html#variant.Release
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let some_bool = AtomicBool::new(true);
    ///
    /// assert_eq!(some_bool.compare_exchange(true,
    ///                                       false,
    ///                                       Ordering::Acquire,
    ///                                       Ordering::Relaxed),
    ///            Ok(true));
    /// assert_eq!(some_bool.load(Ordering::Relaxed), false);
    ///
    /// assert_eq!(some_bool.compare_exchange(true, true,
    ///                                       Ordering::SeqCst,
    ///                                       Ordering::Acquire),
    ///            Err(false));
    /// assert_eq!(some_bool.load(Ordering::Relaxed), false);
    /// ```
    #[inline]
    #[stable(feature = "extended_compare_and_swap", since = "1.10.0")]
    pub fn compare_exchange(&self,
                            current: bool,
                            new: bool,
                            success: Ordering,
                            failure: Ordering)
                            -> Result<bool, bool> {
        match unsafe {
            atomic_compare_exchange(self.v.get(), current as u8, new as u8, success, failure)
        } {
            Ok(x) => Ok(x != 0),
            Err(x) => Err(x != 0),
        }
    }

    /// Stores a value into the `bool` if the current value is the same as the `current` value.
    ///
    /// Unlike `compare_exchange`, this function is allowed to spuriously fail even when the
    /// comparison succeeds, which can result in more efficient code on some platforms. The
    /// return value is a result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if the operation
    /// succeeds while the second describes the required ordering when the operation fails. The
    /// failure ordering can't be [`Release`] or [`AcqRel`] and must be equivalent or
    /// weaker than the success ordering.
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`AcqRel`]: enum.Ordering.html#variant.Release
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let val = AtomicBool::new(false);
    ///
    /// let new = true;
    /// let mut old = val.load(Ordering::Relaxed);
    /// loop {
    ///     match val.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
    ///         Ok(_) => break,
    ///         Err(x) => old = x,
    ///     }
    /// }
    /// ```
    #[inline]
    #[stable(feature = "extended_compare_and_swap", since = "1.10.0")]
    pub fn compare_exchange_weak(&self,
                                 current: bool,
                                 new: bool,
                                 success: Ordering,
                                 failure: Ordering)
                                 -> Result<bool, bool> {
        match unsafe {
            atomic_compare_exchange_weak(self.v.get(), current as u8, new as u8, success, failure)
        } {
            Ok(x) => Ok(x != 0),
            Err(x) => Err(x != 0),
        }
    }

    /// Logical "and" with a boolean value.
    ///
    /// Performs a logical "and" operation on the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_and(false, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_and(true, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(foo.fetch_and(false, Ordering::SeqCst), false);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_and(&self, val: bool, order: Ordering) -> bool {
        unsafe { atomic_and(self.v.get(), val as u8, order) != 0 }
    }

    /// Logical "nand" with a boolean value.
    ///
    /// Performs a logical "nand" operation on the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_nand(false, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_nand(true, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst) as usize, 0);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(foo.fetch_nand(false, Ordering::SeqCst), false);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_nand(&self, val: bool, order: Ordering) -> bool {
        // We can't use atomic_nand here because it can result in a bool with
        // an invalid value. This happens because the atomic operation is done
        // with an 8-bit integer internally, which would set the upper 7 bits.
        // So we just use a compare-exchange loop instead, which is what the
        // intrinsic actually expands to anyways on many platforms.
        let mut old = self.load(Relaxed);
        loop {
            let new = !(old && val);
            match self.compare_exchange_weak(old, new, order, Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
        old
    }

    /// Logical "or" with a boolean value.
    ///
    /// Performs a logical "or" operation on the current value and the argument `val`, and sets the
    /// new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_or(false, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_or(true, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(foo.fetch_or(false, Ordering::SeqCst), false);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_or(&self, val: bool, order: Ordering) -> bool {
        unsafe { atomic_or(self.v.get(), val as u8, order) != 0 }
    }

    /// Logical "xor" with a boolean value.
    ///
    /// Performs a logical "xor" operation on the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_xor(false, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(foo.fetch_xor(true, Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(foo.fetch_xor(false, Ordering::SeqCst), false);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_xor(&self, val: bool, order: Ordering) -> bool {
        unsafe { atomic_xor(self.v.get(), val as u8, order) != 0 }
    }
}

#[cfg(target_has_atomic = "ptr")]
impl<T> AtomicPtr<T> {
    /// Creates a new `AtomicPtr`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicPtr;
    ///
    /// let ptr = &mut 5;
    /// let atomic_ptr  = AtomicPtr::new(ptr);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn new(p: *mut T) -> AtomicPtr<T> {
        AtomicPtr { p: UnsafeCell::new(p) }
    }

    /// Returns a mutable reference to the underlying pointer.
    ///
    /// This is safe because the mutable reference guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let mut atomic_ptr = AtomicPtr::new(&mut 10);
    /// *atomic_ptr.get_mut() = &mut 5;
    /// assert_eq!(unsafe { *atomic_ptr.load(Ordering::SeqCst) }, 5);
    /// ```
    #[inline]
    #[stable(feature = "atomic_access", since = "1.15.0")]
    pub fn get_mut(&mut self) -> &mut *mut T {
        unsafe { &mut *self.p.get() }
    }

    /// Consumes the atomic and returns the contained value.
    ///
    /// This is safe because passing `self` by value guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicPtr;
    ///
    /// let atomic_ptr = AtomicPtr::new(&mut 5);
    /// assert_eq!(unsafe { *atomic_ptr.into_inner() }, 5);
    /// ```
    #[inline]
    #[stable(feature = "atomic_access", since = "1.15.0")]
    pub fn into_inner(self) -> *mut T {
        unsafe { self.p.into_inner() }
    }

    /// Loads a value from the pointer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation.
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Release`] or [`AcqRel`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let ptr = &mut 5;
    /// let some_ptr  = AtomicPtr::new(ptr);
    ///
    /// let value = some_ptr.load(Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn load(&self, order: Ordering) -> *mut T {
        unsafe { atomic_load(self.p.get() as *mut usize, order) as *mut T }
    }

    /// Stores a value into the pointer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation.
    ///
    /// [`Ordering`]: enum.Ordering.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let ptr = &mut 5;
    /// let some_ptr  = AtomicPtr::new(ptr);
    ///
    /// let other_ptr = &mut 10;
    ///
    /// some_ptr.store(other_ptr, Ordering::Relaxed);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Acquire` or `AcqRel`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn store(&self, ptr: *mut T, order: Ordering) {
        unsafe {
            atomic_store(self.p.get() as *mut usize, ptr as usize, order);
        }
    }

    /// Stores a value into the pointer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation.
    ///
    /// [`Ordering`]: enum.Ordering.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let ptr = &mut 5;
    /// let some_ptr  = AtomicPtr::new(ptr);
    ///
    /// let other_ptr = &mut 10;
    ///
    /// let value = some_ptr.swap(other_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, ptr: *mut T, order: Ordering) -> *mut T {
        unsafe { atomic_swap(self.p.get() as *mut usize, ptr as usize, order) as *mut T }
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation.
    ///
    /// [`Ordering`]: enum.Ordering.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let ptr = &mut 5;
    /// let some_ptr  = AtomicPtr::new(ptr);
    ///
    /// let other_ptr   = &mut 10;
    /// let another_ptr = &mut 10;
    ///
    /// let value = some_ptr.compare_and_swap(other_ptr, another_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, current: *mut T, new: *mut T, order: Ordering) -> *mut T {
        match self.compare_exchange(current, new, order, strongest_failure_ordering(order)) {
            Ok(x) => x,
            Err(x) => x,
        }
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was written and containing
    /// the previous value. On success this value is guaranteed to be equal to `current`.
    ///
    /// `compare_exchange` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if
    /// the operation succeeds while the second describes the required ordering when
    /// the operation fails. The failure ordering can't be [`Release`] or [`AcqRel`]
    /// and must be equivalent or weaker than the success ordering.
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let ptr = &mut 5;
    /// let some_ptr  = AtomicPtr::new(ptr);
    ///
    /// let other_ptr   = &mut 10;
    /// let another_ptr = &mut 10;
    ///
    /// let value = some_ptr.compare_exchange(other_ptr, another_ptr,
    ///                                       Ordering::SeqCst, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "extended_compare_and_swap", since = "1.10.0")]
    pub fn compare_exchange(&self,
                            current: *mut T,
                            new: *mut T,
                            success: Ordering,
                            failure: Ordering)
                            -> Result<*mut T, *mut T> {
        unsafe {
            let res = atomic_compare_exchange(self.p.get() as *mut usize,
                                              current as usize,
                                              new as usize,
                                              success,
                                              failure);
            match res {
                Ok(x) => Ok(x as *mut T),
                Err(x) => Err(x as *mut T),
            }
        }
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    ///
    /// Unlike [`compare_exchange`], this function is allowed to spuriously fail even when the
    /// comparison succeeds, which can result in more efficient code on some platforms. The
    /// return value is a result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if the operation
    /// succeeds while the second describes the required ordering when the operation fails. The
    /// failure ordering can't be [`Release`] or [`AcqRel`] and must be equivalent or
    /// weaker than the success ordering.
    ///
    /// [`compare_exchange`]: #method.compare_exchange
    /// [`Ordering`]: enum.Ordering.html
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let some_ptr = AtomicPtr::new(&mut 5);
    ///
    /// let new = &mut 10;
    /// let mut old = some_ptr.load(Ordering::Relaxed);
    /// loop {
    ///     match some_ptr.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
    ///         Ok(_) => break,
    ///         Err(x) => old = x,
    ///     }
    /// }
    /// ```
    #[inline]
    #[stable(feature = "extended_compare_and_swap", since = "1.10.0")]
    pub fn compare_exchange_weak(&self,
                                 current: *mut T,
                                 new: *mut T,
                                 success: Ordering,
                                 failure: Ordering)
                                 -> Result<*mut T, *mut T> {
        unsafe {
            let res = atomic_compare_exchange_weak(self.p.get() as *mut usize,
                                                   current as usize,
                                                   new as usize,
                                                   success,
                                                   failure);
            match res {
                Ok(x) => Ok(x as *mut T),
                Err(x) => Err(x as *mut T),
            }
        }
    }
}

macro_rules! atomic_int {
    ($stable:meta,
     $stable_cxchg:meta,
     $stable_debug:meta,
     $stable_access:meta,
     $int_type:ident $atomic_type:ident $atomic_init:ident) => {
        /// An integer type which can be safely shared between threads.
        ///
        /// This type has the same in-memory representation as the underlying integer type.
        #[$stable]
        pub struct $atomic_type {
            v: UnsafeCell<$int_type>,
        }

        /// An atomic integer initialized to `0`.
        #[$stable]
        pub const $atomic_init: $atomic_type = $atomic_type::new(0);

        #[$stable]
        impl Default for $atomic_type {
            fn default() -> Self {
                Self::new(Default::default())
            }
        }

        #[$stable_debug]
        impl fmt::Debug for $atomic_type {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($atomic_type))
                 .field(&self.load(Ordering::SeqCst))
                 .finish()
            }
        }

        // Send is implicitly implemented.
        #[$stable]
        unsafe impl Sync for $atomic_type {}

        impl $atomic_type {
            /// Creates a new atomic integer.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::AtomicIsize;
            ///
            /// let atomic_forty_two  = AtomicIsize::new(42);
            /// ```
            #[inline]
            #[$stable]
            pub const fn new(v: $int_type) -> Self {
                $atomic_type {v: UnsafeCell::new(v)}
            }

            /// Returns a mutable reference to the underlying integer.
            ///
            /// This is safe because the mutable reference guarantees that no other threads are
            /// concurrently accessing the atomic data.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let mut some_isize = AtomicIsize::new(10);
            /// assert_eq!(*some_isize.get_mut(), 10);
            /// *some_isize.get_mut() = 5;
            /// assert_eq!(some_isize.load(Ordering::SeqCst), 5);
            /// ```
            #[inline]
            #[$stable_access]
            pub fn get_mut(&mut self) -> &mut $int_type {
                unsafe { &mut *self.v.get() }
            }

            /// Consumes the atomic and returns the contained value.
            ///
            /// This is safe because passing `self` by value guarantees that no other threads are
            /// concurrently accessing the atomic data.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::AtomicIsize;
            ///
            /// let some_isize = AtomicIsize::new(5);
            /// assert_eq!(some_isize.into_inner(), 5);
            /// ```
            #[inline]
            #[$stable_access]
            pub fn into_inner(self) -> $int_type {
                unsafe { self.v.into_inner() }
            }

            /// Loads a value from the atomic integer.
            ///
            /// `load` takes an [`Ordering`] argument which describes the memory ordering of this
            /// operation.
            ///
            /// # Panics
            ///
            /// Panics if `order` is [`Release`] or [`AcqRel`].
            ///
            /// [`Ordering`]: enum.Ordering.html
            /// [`Release`]: enum.Ordering.html#variant.Release
            /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let some_isize = AtomicIsize::new(5);
            ///
            /// assert_eq!(some_isize.load(Ordering::Relaxed), 5);
            /// ```
            #[inline]
            #[$stable]
            pub fn load(&self, order: Ordering) -> $int_type {
                unsafe { atomic_load(self.v.get(), order) }
            }

            /// Stores a value into the atomic integer.
            ///
            /// `store` takes an [`Ordering`] argument which describes the memory ordering of this
            /// operation.
            ///
            /// [`Ordering`]: enum.Ordering.html
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let some_isize = AtomicIsize::new(5);
            ///
            /// some_isize.store(10, Ordering::Relaxed);
            /// assert_eq!(some_isize.load(Ordering::Relaxed), 10);
            /// ```
            ///
            /// # Panics
            ///
            /// Panics if `order` is `Acquire` or `AcqRel`.
            #[inline]
            #[$stable]
            pub fn store(&self, val: $int_type, order: Ordering) {
                unsafe { atomic_store(self.v.get(), val, order); }
            }

            /// Stores a value into the atomic integer, returning the previous value.
            ///
            /// `swap` takes an [`Ordering`] argument which describes the memory ordering of this
            /// operation.
            ///
            /// [`Ordering`]: enum.Ordering.html
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let some_isize = AtomicIsize::new(5);
            ///
            /// assert_eq!(some_isize.swap(10, Ordering::Relaxed), 5);
            /// ```
            #[inline]
            #[$stable]
            pub fn swap(&self, val: $int_type, order: Ordering) -> $int_type {
                unsafe { atomic_swap(self.v.get(), val, order) }
            }

            /// Stores a value into the atomic integer if the current value is the same as the
            /// `current` value.
            ///
            /// The return value is always the previous value. If it is equal to `current`, then the
            /// value was updated.
            ///
            /// `compare_and_swap` also takes an [`Ordering`] argument which describes the memory
            /// ordering of this operation.
            ///
            /// [`Ordering`]: enum.Ordering.html
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let some_isize = AtomicIsize::new(5);
            ///
            /// assert_eq!(some_isize.compare_and_swap(5, 10, Ordering::Relaxed), 5);
            /// assert_eq!(some_isize.load(Ordering::Relaxed), 10);
            ///
            /// assert_eq!(some_isize.compare_and_swap(6, 12, Ordering::Relaxed), 10);
            /// assert_eq!(some_isize.load(Ordering::Relaxed), 10);
            /// ```
            #[inline]
            #[$stable]
            pub fn compare_and_swap(&self,
                                    current: $int_type,
                                    new: $int_type,
                                    order: Ordering) -> $int_type {
                match self.compare_exchange(current,
                                            new,
                                            order,
                                            strongest_failure_ordering(order)) {
                    Ok(x) => x,
                    Err(x) => x,
                }
            }

            /// Stores a value into the atomic integer if the current value is the same as the
            /// `current` value.
            ///
            /// The return value is a result indicating whether the new value was written and
            /// containing the previous value. On success this value is guaranteed to be equal to
            /// `current`.
            ///
            /// `compare_exchange` takes two [`Ordering`] arguments to describe the memory
            /// ordering of this operation. The first describes the required ordering if
            /// the operation succeeds while the second describes the required ordering when
            /// the operation fails. The failure ordering can't be [`Release`] or [`AcqRel`] and
            /// must be equivalent or weaker than the success ordering.
            ///
            /// [`Ordering`]: enum.Ordering.html
            /// [`Release`]: enum.Ordering.html#variant.Release
            /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let some_isize = AtomicIsize::new(5);
            ///
            /// assert_eq!(some_isize.compare_exchange(5, 10,
            ///                                        Ordering::Acquire,
            ///                                        Ordering::Relaxed),
            ///            Ok(5));
            /// assert_eq!(some_isize.load(Ordering::Relaxed), 10);
            ///
            /// assert_eq!(some_isize.compare_exchange(6, 12,
            ///                                        Ordering::SeqCst,
            ///                                        Ordering::Acquire),
            ///            Err(10));
            /// assert_eq!(some_isize.load(Ordering::Relaxed), 10);
            /// ```
            #[inline]
            #[$stable_cxchg]
            pub fn compare_exchange(&self,
                                    current: $int_type,
                                    new: $int_type,
                                    success: Ordering,
                                    failure: Ordering) -> Result<$int_type, $int_type> {
                unsafe { atomic_compare_exchange(self.v.get(), current, new, success, failure) }
            }

            /// Stores a value into the atomic integer if the current value is the same as the
            /// `current` value.
            ///
            /// Unlike [`compare_exchange`], this function is allowed to spuriously fail even
            /// when the comparison succeeds, which can result in more efficient code on some
            /// platforms. The return value is a result indicating whether the new value was
            /// written and containing the previous value.
            ///
            /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
            /// ordering of this operation. The first describes the required ordering if the
            /// operation succeeds while the second describes the required ordering when the
            /// operation fails. The failure ordering can't be [`Release`] or [`AcqRel`] and
            /// must be equivalent or weaker than the success ordering.
            ///
            /// [`compare_exchange`]: #method.compare_exchange
            /// [`Ordering`]: enum.Ordering.html
            /// [`Release`]: enum.Ordering.html#variant.Release
            /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let val = AtomicIsize::new(4);
            ///
            /// let mut old = val.load(Ordering::Relaxed);
            /// loop {
            ///     let new = old * 2;
            ///     match val.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
            ///         Ok(_) => break,
            ///         Err(x) => old = x,
            ///     }
            /// }
            /// ```
            #[inline]
            #[$stable_cxchg]
            pub fn compare_exchange_weak(&self,
                                         current: $int_type,
                                         new: $int_type,
                                         success: Ordering,
                                         failure: Ordering) -> Result<$int_type, $int_type> {
                unsafe {
                    atomic_compare_exchange_weak(self.v.get(), current, new, success, failure)
                }
            }

            /// Adds to the current value, returning the previous value.
            ///
            /// This operation wraps around on overflow.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let foo = AtomicIsize::new(0);
            /// assert_eq!(foo.fetch_add(10, Ordering::SeqCst), 0);
            /// assert_eq!(foo.load(Ordering::SeqCst), 10);
            /// ```
            #[inline]
            #[$stable]
            pub fn fetch_add(&self, val: $int_type, order: Ordering) -> $int_type {
                unsafe { atomic_add(self.v.get(), val, order) }
            }

            /// Subtracts from the current value, returning the previous value.
            ///
            /// This operation wraps around on overflow.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let foo = AtomicIsize::new(0);
            /// assert_eq!(foo.fetch_sub(10, Ordering::SeqCst), 0);
            /// assert_eq!(foo.load(Ordering::SeqCst), -10);
            /// ```
            #[inline]
            #[$stable]
            pub fn fetch_sub(&self, val: $int_type, order: Ordering) -> $int_type {
                unsafe { atomic_sub(self.v.get(), val, order) }
            }

            /// Bitwise "and" with the current value.
            ///
            /// Performs a bitwise "and" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let foo = AtomicIsize::new(0b101101);
            /// assert_eq!(foo.fetch_and(0b110011, Ordering::SeqCst), 0b101101);
            /// assert_eq!(foo.load(Ordering::SeqCst), 0b100001);
            #[inline]
            #[$stable]
            pub fn fetch_and(&self, val: $int_type, order: Ordering) -> $int_type {
                unsafe { atomic_and(self.v.get(), val, order) }
            }

            /// Bitwise "or" with the current value.
            ///
            /// Performs a bitwise "or" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let foo = AtomicIsize::new(0b101101);
            /// assert_eq!(foo.fetch_or(0b110011, Ordering::SeqCst), 0b101101);
            /// assert_eq!(foo.load(Ordering::SeqCst), 0b111111);
            #[inline]
            #[$stable]
            pub fn fetch_or(&self, val: $int_type, order: Ordering) -> $int_type {
                unsafe { atomic_or(self.v.get(), val, order) }
            }

            /// Bitwise "xor" with the current value.
            ///
            /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// # Examples
            ///
            /// ```
            /// use std::sync::atomic::{AtomicIsize, Ordering};
            ///
            /// let foo = AtomicIsize::new(0b101101);
            /// assert_eq!(foo.fetch_xor(0b110011, Ordering::SeqCst), 0b101101);
            /// assert_eq!(foo.load(Ordering::SeqCst), 0b011110);
            #[inline]
            #[$stable]
            pub fn fetch_xor(&self, val: $int_type, order: Ordering) -> $int_type {
                unsafe { atomic_xor(self.v.get(), val, order) }
            }
        }
    }
}

#[cfg(target_has_atomic = "8")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    i8 AtomicI8 ATOMIC_I8_INIT
}
#[cfg(target_has_atomic = "8")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    u8 AtomicU8 ATOMIC_U8_INIT
}
#[cfg(target_has_atomic = "16")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    i16 AtomicI16 ATOMIC_I16_INIT
}
#[cfg(target_has_atomic = "16")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    u16 AtomicU16 ATOMIC_U16_INIT
}
#[cfg(target_has_atomic = "32")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    i32 AtomicI32 ATOMIC_I32_INIT
}
#[cfg(target_has_atomic = "32")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    u32 AtomicU32 ATOMIC_U32_INIT
}
#[cfg(target_has_atomic = "64")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    i64 AtomicI64 ATOMIC_I64_INIT
}
#[cfg(target_has_atomic = "64")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    u64 AtomicU64 ATOMIC_U64_INIT
}
#[cfg(target_has_atomic = "ptr")]
atomic_int!{
    stable(feature = "rust1", since = "1.0.0"),
    stable(feature = "extended_compare_and_swap", since = "1.10.0"),
    stable(feature = "atomic_debug", since = "1.3.0"),
    stable(feature = "atomic_access", since = "1.15.0"),
    isize AtomicIsize ATOMIC_ISIZE_INIT
}
#[cfg(target_has_atomic = "ptr")]
atomic_int!{
    stable(feature = "rust1", since = "1.0.0"),
    stable(feature = "extended_compare_and_swap", since = "1.10.0"),
    stable(feature = "atomic_debug", since = "1.3.0"),
    stable(feature = "atomic_access", since = "1.15.0"),
    usize AtomicUsize ATOMIC_USIZE_INIT
}

#[inline]
fn strongest_failure_ordering(order: Ordering) -> Ordering {
    match order {
        Release => Relaxed,
        Relaxed => Relaxed,
        SeqCst => SeqCst,
        Acquire => Acquire,
        AcqRel => Acquire,
        __Nonexhaustive => __Nonexhaustive,
    }
}

#[inline]
unsafe fn atomic_store<T>(dst: *mut T, val: T, order: Ordering) {
    match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        Relaxed => intrinsics::atomic_store_relaxed(dst, val),
        SeqCst => intrinsics::atomic_store(dst, val),
        Acquire => panic!("there is no such thing as an acquire store"),
        AcqRel => panic!("there is no such thing as an acquire/release store"),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

#[inline]
unsafe fn atomic_load<T>(dst: *const T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_load_acq(dst),
        Relaxed => intrinsics::atomic_load_relaxed(dst),
        SeqCst => intrinsics::atomic_load(dst),
        Release => panic!("there is no such thing as a release load"),
        AcqRel => panic!("there is no such thing as an acquire/release load"),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

#[inline]
unsafe fn atomic_swap<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        AcqRel => intrinsics::atomic_xchg_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xchg(dst, val),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

/// Returns the previous value (like __sync_fetch_and_add).
#[inline]
unsafe fn atomic_add<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xadd_acq(dst, val),
        Release => intrinsics::atomic_xadd_rel(dst, val),
        AcqRel => intrinsics::atomic_xadd_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xadd(dst, val),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

/// Returns the previous value (like __sync_fetch_and_sub).
#[inline]
unsafe fn atomic_sub<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xsub_acq(dst, val),
        Release => intrinsics::atomic_xsub_rel(dst, val),
        AcqRel => intrinsics::atomic_xsub_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xsub(dst, val),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

#[inline]
unsafe fn atomic_compare_exchange<T>(dst: *mut T,
                                     old: T,
                                     new: T,
                                     success: Ordering,
                                     failure: Ordering)
                                     -> Result<T, T> {
    let (val, ok) = match (success, failure) {
        (Acquire, Acquire) => intrinsics::atomic_cxchg_acq(dst, old, new),
        (Release, Relaxed) => intrinsics::atomic_cxchg_rel(dst, old, new),
        (AcqRel, Acquire) => intrinsics::atomic_cxchg_acqrel(dst, old, new),
        (Relaxed, Relaxed) => intrinsics::atomic_cxchg_relaxed(dst, old, new),
        (SeqCst, SeqCst) => intrinsics::atomic_cxchg(dst, old, new),
        (Acquire, Relaxed) => intrinsics::atomic_cxchg_acq_failrelaxed(dst, old, new),
        (AcqRel, Relaxed) => intrinsics::atomic_cxchg_acqrel_failrelaxed(dst, old, new),
        (SeqCst, Relaxed) => intrinsics::atomic_cxchg_failrelaxed(dst, old, new),
        (SeqCst, Acquire) => intrinsics::atomic_cxchg_failacq(dst, old, new),
        (__Nonexhaustive, _) => panic!("invalid memory ordering"),
        (_, __Nonexhaustive) => panic!("invalid memory ordering"),
        (_, AcqRel) => panic!("there is no such thing as an acquire/release failure ordering"),
        (_, Release) => panic!("there is no such thing as a release failure ordering"),
        _ => panic!("a failure ordering can't be stronger than a success ordering"),
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
unsafe fn atomic_compare_exchange_weak<T>(dst: *mut T,
                                          old: T,
                                          new: T,
                                          success: Ordering,
                                          failure: Ordering)
                                          -> Result<T, T> {
    let (val, ok) = match (success, failure) {
        (Acquire, Acquire) => intrinsics::atomic_cxchgweak_acq(dst, old, new),
        (Release, Relaxed) => intrinsics::atomic_cxchgweak_rel(dst, old, new),
        (AcqRel, Acquire) => intrinsics::atomic_cxchgweak_acqrel(dst, old, new),
        (Relaxed, Relaxed) => intrinsics::atomic_cxchgweak_relaxed(dst, old, new),
        (SeqCst, SeqCst) => intrinsics::atomic_cxchgweak(dst, old, new),
        (Acquire, Relaxed) => intrinsics::atomic_cxchgweak_acq_failrelaxed(dst, old, new),
        (AcqRel, Relaxed) => intrinsics::atomic_cxchgweak_acqrel_failrelaxed(dst, old, new),
        (SeqCst, Relaxed) => intrinsics::atomic_cxchgweak_failrelaxed(dst, old, new),
        (SeqCst, Acquire) => intrinsics::atomic_cxchgweak_failacq(dst, old, new),
        (__Nonexhaustive, _) => panic!("invalid memory ordering"),
        (_, __Nonexhaustive) => panic!("invalid memory ordering"),
        (_, AcqRel) => panic!("there is no such thing as an acquire/release failure ordering"),
        (_, Release) => panic!("there is no such thing as a release failure ordering"),
        _ => panic!("a failure ordering can't be stronger than a success ordering"),
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
unsafe fn atomic_and<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_and_acq(dst, val),
        Release => intrinsics::atomic_and_rel(dst, val),
        AcqRel => intrinsics::atomic_and_acqrel(dst, val),
        Relaxed => intrinsics::atomic_and_relaxed(dst, val),
        SeqCst => intrinsics::atomic_and(dst, val),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

#[inline]
unsafe fn atomic_or<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_or_acq(dst, val),
        Release => intrinsics::atomic_or_rel(dst, val),
        AcqRel => intrinsics::atomic_or_acqrel(dst, val),
        Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        SeqCst => intrinsics::atomic_or(dst, val),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

#[inline]
unsafe fn atomic_xor<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xor_acq(dst, val),
        Release => intrinsics::atomic_xor_rel(dst, val),
        AcqRel => intrinsics::atomic_xor_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xor(dst, val),
        __Nonexhaustive => panic!("invalid memory ordering"),
    }
}

/// An atomic fence.
///
/// A fence 'A' which has [`Release`] ordering semantics, synchronizes with a
/// fence 'B' with (at least) [`Acquire`] semantics, if and only if there exists
/// atomic operations X and Y, both operating on some atomic object 'M' such
/// that A is sequenced before X, Y is synchronized before B and Y observes
/// the change to M. This provides a happens-before dependence between A and B.
///
/// Atomic operations with [`Release`] or [`Acquire`] semantics can also synchronize
/// with a fence.
///
/// A fence which has [`SeqCst`] ordering, in addition to having both [`Acquire`]
/// and [`Release`] semantics, participates in the global program order of the
/// other [`SeqCst`] operations and/or fences.
///
/// Accepts [`Acquire`], [`Release`], [`AcqRel`] and [`SeqCst`] orderings.
///
/// # Panics
///
/// Panics if `order` is [`Relaxed`].
///
/// [`Ordering`]: enum.Ordering.html
/// [`Acquire`]: enum.Ordering.html#variant.Acquire
/// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
/// [`Release`]: enum.Ordering.html#variant.Release
/// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
/// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn fence(order: Ordering) {
    unsafe {
        match order {
            Acquire => intrinsics::atomic_fence_acq(),
            Release => intrinsics::atomic_fence_rel(),
            AcqRel => intrinsics::atomic_fence_acqrel(),
            SeqCst => intrinsics::atomic_fence(),
            Relaxed => panic!("there is no such thing as a relaxed fence"),
            __Nonexhaustive => panic!("invalid memory ordering"),
        }
    }
}


#[cfg(target_has_atomic = "8")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl fmt::Debug for AtomicBool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("AtomicBool").field(&self.load(Ordering::SeqCst)).finish()
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl<T> fmt::Debug for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("AtomicPtr").field(&self.load(Ordering::SeqCst)).finish()
    }
}
