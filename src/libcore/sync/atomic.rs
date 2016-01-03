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
//! same as [LLVM atomic orderings][1].
//!
//! [1]: http://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations
//!
//! Atomic variables are safe to share between threads (they implement `Sync`)
//! but they do not themselves provide the mechanism for sharing. The most
//! common way to share an atomic variable is to put it into an `Arc` (an
//! atomically-reference-counted shared pointer).
//!
//! Most atomic types may be stored in static variables, initialized using
//! the provided static initializers like `INIT_ATOMIC_BOOL`. Atomic statics
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
//!     thread::spawn(move|| {
//!         spinlock_clone.store(0, Ordering::SeqCst);
//!     });
//!
//!     // Wait for the other thread to release the lock
//!     while spinlock.load(Ordering::SeqCst) != 0 {}
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

use self::Ordering::*;

use marker::{Send, Sync};

use intrinsics;
use cell::UnsafeCell;

use default::Default;
use fmt;

/// A boolean type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicBool {
    v: UnsafeCell<usize>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for AtomicBool {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

// Send is implicitly implemented for AtomicBool.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for AtomicBool {}

/// A signed integer type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicIsize {
    v: UnsafeCell<isize>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for AtomicIsize {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

// Send is implicitly implemented for AtomicIsize.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for AtomicIsize {}

/// An unsigned integer type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicUsize {
    v: UnsafeCell<usize>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for AtomicUsize {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

// Send is implicitly implemented for AtomicUsize.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for AtomicUsize {}

/// A raw pointer type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicPtr<T> {
    p: UnsafeCell<*mut T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for AtomicPtr<T> {
    fn default() -> AtomicPtr<T> {
        AtomicPtr::new(::ptr::null_mut())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T> Send for AtomicPtr<T> {}
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
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone)]
pub enum Ordering {
    /// No ordering constraints, only atomic operations. Corresponds to LLVM's
    /// `Monotonic` ordering.
    #[stable(feature = "rust1", since = "1.0.0")]
    Relaxed,
    /// When coupled with a store, all previous writes become visible
    /// to another thread that performs a load with `Acquire` ordering
    /// on the same value.
    #[stable(feature = "rust1", since = "1.0.0")]
    Release,
    /// When coupled with a load, all subsequent loads will see data
    /// written before a store with `Release` ordering on the same value
    /// in another thread.
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
}

/// An `AtomicBool` initialized to `false`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const ATOMIC_BOOL_INIT: AtomicBool = AtomicBool::new(false);
/// An `AtomicIsize` initialized to `0`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const ATOMIC_ISIZE_INIT: AtomicIsize = AtomicIsize::new(0);
/// An `AtomicUsize` initialized to `0`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const ATOMIC_USIZE_INIT: AtomicUsize = AtomicUsize::new(0);

// NB: Needs to be -1 (0b11111111...) to make fetch_nand work correctly
const UINT_TRUE: usize = !0;

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
        AtomicBool { v: UnsafeCell::new(-(v as isize) as usize) }
    }

    /// Loads a value from the bool.
    ///
    /// `load` takes an `Ordering` argument which describes the memory ordering of this operation.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Release` or `AcqRel`.
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
        unsafe { atomic_load(self.v.get(), order) > 0 }
    }

    /// Stores a value into the bool.
    ///
    /// `store` takes an `Ordering` argument which describes the memory ordering of this operation.
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
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_store(self.v.get(), val, order); }
    }

    /// Stores a value into the bool, returning the old value.
    ///
    /// `swap` takes an `Ordering` argument which describes the memory ordering of this operation.
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
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_swap(self.v.get(), val, order) > 0 }
    }

    /// Stores a value into the `bool` if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an `Ordering` argument which describes the memory ordering of
    /// this operation.
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
        let current = if current { UINT_TRUE } else { 0 };
        let new = if new { UINT_TRUE } else { 0 };

        unsafe { atomic_compare_and_swap(self.v.get(), current, new, order) > 0 }
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
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_and(self.v.get(), val, order) > 0 }
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
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_nand(self.v.get(), val, order) > 0 }
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
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_or(self.v.get(), val, order) > 0 }
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
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_xor(self.v.get(), val, order) > 0 }
    }
}

impl AtomicIsize {
    /// Creates a new `AtomicIsize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicIsize;
    ///
    /// let atomic_forty_two  = AtomicIsize::new(42);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn new(v: isize) -> AtomicIsize {
        AtomicIsize {v: UnsafeCell::new(v)}
    }

    /// Loads a value from the isize.
    ///
    /// `load` takes an `Ordering` argument which describes the memory ordering of this operation.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Release` or `AcqRel`.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn load(&self, order: Ordering) -> isize {
        unsafe { atomic_load(self.v.get(), order) }
    }

    /// Stores a value into the isize.
    ///
    /// `store` takes an `Ordering` argument which describes the memory ordering of this operation.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn store(&self, val: isize, order: Ordering) {
        unsafe { atomic_store(self.v.get(), val, order); }
    }

    /// Stores a value into the isize, returning the old value.
    ///
    /// `swap` takes an `Ordering` argument which describes the memory ordering of this operation.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    /// Stores a value into the `isize` if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an `Ordering` argument which describes the memory ordering of
    /// this operation.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, current: isize, new: isize, order: Ordering) -> isize {
        unsafe { atomic_compare_and_swap(self.v.get(), current, new, order) }
    }

    /// Add an isize to the current value, returning the previous value.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_add(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_add(self.v.get(), val, order) }
    }

    /// Subtract an isize from the current value, returning the previous value.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_sub(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_sub(self.v.get(), val, order) }
    }

    /// Bitwise and with the current isize, returning the previous value.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_and(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_and(self.v.get(), val, order) }
    }

    /// Bitwise or with the current isize, returning the previous value.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_or(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_or(self.v.get(), val, order) }
    }

    /// Bitwise xor with the current isize, returning the previous value.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_xor(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}

impl AtomicUsize {
    /// Creates a new `AtomicUsize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicUsize;
    ///
    /// let atomic_forty_two = AtomicUsize::new(42);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn new(v: usize) -> AtomicUsize {
        AtomicUsize { v: UnsafeCell::new(v) }
    }

    /// Loads a value from the usize.
    ///
    /// `load` takes an `Ordering` argument which describes the memory ordering of this operation.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Release` or `AcqRel`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let some_usize = AtomicUsize::new(5);
    ///
    /// assert_eq!(some_usize.load(Ordering::Relaxed), 5);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn load(&self, order: Ordering) -> usize {
        unsafe { atomic_load(self.v.get(), order) }
    }

    /// Stores a value into the usize.
    ///
    /// `store` takes an `Ordering` argument which describes the memory ordering of this operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let some_usize = AtomicUsize::new(5);
    ///
    /// some_usize.store(10, Ordering::Relaxed);
    /// assert_eq!(some_usize.load(Ordering::Relaxed), 10);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Acquire` or `AcqRel`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn store(&self, val: usize, order: Ordering) {
        unsafe { atomic_store(self.v.get(), val, order); }
    }

    /// Stores a value into the usize, returning the old value.
    ///
    /// `swap` takes an `Ordering` argument which describes the memory ordering of this operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let some_usize= AtomicUsize::new(5);
    ///
    /// assert_eq!(some_usize.swap(10, Ordering::Relaxed), 5);
    /// assert_eq!(some_usize.load(Ordering::Relaxed), 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    /// Stores a value into the `usize` if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an `Ordering` argument which describes the memory ordering of
    /// this operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let some_usize = AtomicUsize::new(5);
    ///
    /// assert_eq!(some_usize.compare_and_swap(5, 10, Ordering::Relaxed), 5);
    /// assert_eq!(some_usize.load(Ordering::Relaxed), 10);
    ///
    /// assert_eq!(some_usize.compare_and_swap(6, 12, Ordering::Relaxed), 10);
    /// assert_eq!(some_usize.load(Ordering::Relaxed), 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, current: usize, new: usize, order: Ordering) -> usize {
        unsafe { atomic_compare_and_swap(self.v.get(), current, new, order) }
    }

    /// Add to the current usize, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let foo = AtomicUsize::new(0);
    /// assert_eq!(foo.fetch_add(10, Ordering::SeqCst), 0);
    /// assert_eq!(foo.load(Ordering::SeqCst), 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_add(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_add(self.v.get(), val, order) }
    }

    /// Subtract from the current usize, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let foo = AtomicUsize::new(10);
    /// assert_eq!(foo.fetch_sub(10, Ordering::SeqCst), 10);
    /// assert_eq!(foo.load(Ordering::SeqCst), 0);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_sub(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_sub(self.v.get(), val, order) }
    }

    /// Bitwise and with the current usize, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let foo = AtomicUsize::new(0b101101);
    /// assert_eq!(foo.fetch_and(0b110011, Ordering::SeqCst), 0b101101);
    /// assert_eq!(foo.load(Ordering::SeqCst), 0b100001);
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_and(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_and(self.v.get(), val, order) }
    }

    /// Bitwise or with the current usize, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let foo = AtomicUsize::new(0b101101);
    /// assert_eq!(foo.fetch_or(0b110011, Ordering::SeqCst), 0b101101);
    /// assert_eq!(foo.load(Ordering::SeqCst), 0b111111);
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_or(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_or(self.v.get(), val, order) }
    }

    /// Bitwise xor with the current usize, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let foo = AtomicUsize::new(0b101101);
    /// assert_eq!(foo.fetch_xor(0b110011, Ordering::SeqCst), 0b101101);
    /// assert_eq!(foo.load(Ordering::SeqCst), 0b011110);
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_xor(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}

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

    /// Loads a value from the pointer.
    ///
    /// `load` takes an `Ordering` argument which describes the memory ordering of this operation.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Release` or `AcqRel`.
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
        unsafe {
            atomic_load(self.p.get() as *mut usize, order) as *mut T
        }
    }

    /// Stores a value into the pointer.
    ///
    /// `store` takes an `Ordering` argument which describes the memory ordering of this operation.
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
        unsafe { atomic_store(self.p.get() as *mut usize, ptr as usize, order); }
    }

    /// Stores a value into the pointer, returning the old value.
    ///
    /// `swap` takes an `Ordering` argument which describes the memory ordering of this operation.
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

    /// Stores a value into the pointer if the current pointer *address* value is the same
    /// as the `current` pointer *address* value.
    ///
    /// The return value is always the previous value contained in the pointer. If it is equal
    /// to the value pointed by `current`, then the value was updated.
    ///
    /// `compare_and_swap` also takes an `Ordering` argument which describes the memory
    ///  ordering of this operation.
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
    /// // here `some_ptr` will not change as `other_ptr` has a different address from `ptr`
    /// some_ptr.compare_and_swap(other_ptr, another_ptr, Ordering::Relaxed);
    /// assert_eq!( unsafe { *some_ptr.load(Ordering::Relaxed) }, 5);
    /// // but, in this instance `some_ptr` has `ptr` address, so the value is changed.
    /// some_ptr.compare_and_swap(ptr, another_ptr, Ordering::Relaxed);
    /// assert_eq!(unsafe { *some_ptr.load(Ordering::Relaxed) }, 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, current: *mut T, new: *mut T, order: Ordering) -> *mut T {
        unsafe {
            atomic_compare_and_swap(self.p.get() as *mut usize, current as usize,
                                    new as usize, order) as *mut T
        }
    }
}

#[inline]
unsafe fn atomic_store<T>(dst: *mut T, val: T, order:Ordering) {
    match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        Relaxed => intrinsics::atomic_store_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_store(dst, val),
        Acquire => panic!("there is no such thing as an acquire store"),
        AcqRel  => panic!("there is no such thing as an acquire/release store"),
    }
}

#[inline]
unsafe fn atomic_load<T>(dst: *const T, order:Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_load_acq(dst),
        Relaxed => intrinsics::atomic_load_relaxed(dst),
        SeqCst  => intrinsics::atomic_load(dst),
        Release => panic!("there is no such thing as a release load"),
        AcqRel  => panic!("there is no such thing as an acquire/release load"),
    }
}

#[inline]
unsafe fn atomic_swap<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        AcqRel  => intrinsics::atomic_xchg_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_xchg(dst, val)
    }
}

/// Returns the old value (like __sync_fetch_and_add).
#[inline]
unsafe fn atomic_add<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xadd_acq(dst, val),
        Release => intrinsics::atomic_xadd_rel(dst, val),
        AcqRel  => intrinsics::atomic_xadd_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_xadd(dst, val)
    }
}

/// Returns the old value (like __sync_fetch_and_sub).
#[inline]
unsafe fn atomic_sub<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xsub_acq(dst, val),
        Release => intrinsics::atomic_xsub_rel(dst, val),
        AcqRel  => intrinsics::atomic_xsub_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_xsub(dst, val)
    }
}

#[inline]
unsafe fn atomic_compare_and_swap<T>(dst: *mut T, old:T, new:T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_cxchg_acq(dst, old, new),
        Release => intrinsics::atomic_cxchg_rel(dst, old, new),
        AcqRel  => intrinsics::atomic_cxchg_acqrel(dst, old, new),
        Relaxed => intrinsics::atomic_cxchg_relaxed(dst, old, new),
        SeqCst  => intrinsics::atomic_cxchg(dst, old, new),
    }
}

#[inline]
unsafe fn atomic_and<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_and_acq(dst, val),
        Release => intrinsics::atomic_and_rel(dst, val),
        AcqRel  => intrinsics::atomic_and_acqrel(dst, val),
        Relaxed => intrinsics::atomic_and_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_and(dst, val)
    }
}

#[inline]
unsafe fn atomic_nand<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_nand_acq(dst, val),
        Release => intrinsics::atomic_nand_rel(dst, val),
        AcqRel  => intrinsics::atomic_nand_acqrel(dst, val),
        Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_nand(dst, val)
    }
}


#[inline]
unsafe fn atomic_or<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_or_acq(dst, val),
        Release => intrinsics::atomic_or_rel(dst, val),
        AcqRel  => intrinsics::atomic_or_acqrel(dst, val),
        Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_or(dst, val)
    }
}


#[inline]
unsafe fn atomic_xor<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xor_acq(dst, val),
        Release => intrinsics::atomic_xor_rel(dst, val),
        AcqRel  => intrinsics::atomic_xor_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        SeqCst  => intrinsics::atomic_xor(dst, val)
    }
}


/// An atomic fence.
///
/// A fence 'A' which has `Release` ordering semantics, synchronizes with a
/// fence 'B' with (at least) `Acquire` semantics, if and only if there exists
/// atomic operations X and Y, both operating on some atomic object 'M' such
/// that A is sequenced before X, Y is synchronized before B and Y observes
/// the change to M. This provides a happens-before dependence between A and B.
///
/// Atomic operations with `Release` or `Acquire` semantics can also synchronize
/// with a fence.
///
/// A fence which has `SeqCst` ordering, in addition to having both `Acquire`
/// and `Release` semantics, participates in the global program order of the
/// other `SeqCst` operations and/or fences.
///
/// Accepts `Acquire`, `Release`, `AcqRel` and `SeqCst` orderings.
///
/// # Panics
///
/// Panics if `order` is `Relaxed`.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn fence(order: Ordering) {
    unsafe {
        match order {
            Acquire => intrinsics::atomic_fence_acq(),
            Release => intrinsics::atomic_fence_rel(),
            AcqRel  => intrinsics::atomic_fence_acqrel(),
            SeqCst  => intrinsics::atomic_fence(),
            Relaxed => panic!("there is no such thing as a relaxed fence")
        }
    }
}

macro_rules! impl_Debug {
    ($($t:ident)*) => ($(
        #[stable(feature = "atomic_debug", since = "1.3.0")]
        impl fmt::Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($t)).field(&self.load(Ordering::SeqCst)).finish()
            }
        }
    )*);
}

impl_Debug!{ AtomicUsize AtomicIsize AtomicBool }

#[stable(feature = "atomic_debug", since = "1.3.0")]
impl<T> fmt::Debug for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("AtomicPtr").field(&self.load(Ordering::SeqCst)).finish()
    }
}
