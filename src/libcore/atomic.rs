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
//! types, including `AtomicBool`, `AtomicIsize`, `AtomicUsize`, and `AtomicOption`.
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
//!     // Wait for the other task to release the lock
//!     while spinlock.load(Ordering::SeqCst) != 0 {}
//! }
//! ```
//!
//! Keep a global count of live tasks:
//!
//! ```
//! use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
//!
//! static GLOBAL_TASK_COUNT: AtomicUsize = ATOMIC_USIZE_INIT;
//!
//! let old_task_count = GLOBAL_TASK_COUNT.fetch_add(1, Ordering::SeqCst);
//! println!("live tasks: {}", old_task_count + 1);
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

use self::Ordering::*;

use marker::Sync;

use intrinsics;
use cell::UnsafeCell;
use marker::PhantomData;

/// A boolean type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicBool {
    v: UnsafeCell<usize>,
}

unsafe impl Sync for AtomicBool {}

/// A signed integer type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicIsize {
    v: UnsafeCell<isize>,
}

unsafe impl Sync for AtomicIsize {}

/// An unsigned integer type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicUsize {
    v: UnsafeCell<usize>,
}

unsafe impl Sync for AtomicUsize {}

/// A raw pointer type which can be safely shared between threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct AtomicPtr<T> {
    p: UnsafeCell<usize>,
    _marker: PhantomData<*mut T>,
}

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
/// C++'s](http://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync).
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy)]
pub enum Ordering {
    /// No ordering constraints, only atomic operations.
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
pub const ATOMIC_BOOL_INIT: AtomicBool =
        AtomicBool { v: UnsafeCell { value: 0 } };
/// An `AtomicIsize` initialized to `0`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const ATOMIC_ISIZE_INIT: AtomicIsize =
        AtomicIsize { v: UnsafeCell { value: 0 } };
/// An `AtomicUsize` initialized to `0`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const ATOMIC_USIZE_INIT: AtomicUsize =
        AtomicUsize { v: UnsafeCell { value: 0, } };

// NB: Needs to be -1 (0b11111111...) to make fetch_nand work correctly
const UINT_TRUE: usize = -1;

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
    pub fn new(v: bool) -> AtomicBool {
        let val = if v { UINT_TRUE } else { 0 };
        AtomicBool { v: UnsafeCell::new(val) }
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
    /// let value = some_bool.load(Ordering::Relaxed);
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
    /// let value = some_bool.swap(false, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_swap(self.v.get(), val, order) > 0 }
    }

    /// Stores a value into the bool if the current value is the same as the expected value.
    ///
    /// If the return value is equal to `old` then the value was updated.
    ///
    /// `swap` also takes an `Ordering` argument which describes the memory ordering of this
    /// operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let some_bool = AtomicBool::new(true);
    ///
    /// let value = some_bool.store(false, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, old: bool, new: bool, order: Ordering) -> bool {
        let old = if old { UINT_TRUE } else { 0 };
        let new = if new { UINT_TRUE } else { 0 };

        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) > 0 }
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
    /// assert_eq!(true, foo.fetch_and(false, Ordering::SeqCst));
    /// assert_eq!(false, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_and(true, Ordering::SeqCst));
    /// assert_eq!(true, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_and(false, Ordering::SeqCst));
    /// assert_eq!(false, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(true, foo.fetch_nand(false, Ordering::SeqCst));
    /// assert_eq!(true, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_nand(true, Ordering::SeqCst));
    /// assert_eq!(0, foo.load(Ordering::SeqCst) as usize);
    /// assert_eq!(false, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_nand(false, Ordering::SeqCst));
    /// assert_eq!(true, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(true, foo.fetch_or(false, Ordering::SeqCst));
    /// assert_eq!(true, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_or(true, Ordering::SeqCst));
    /// assert_eq!(true, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_or(false, Ordering::SeqCst));
    /// assert_eq!(false, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(true, foo.fetch_xor(false, Ordering::SeqCst));
    /// assert_eq!(true, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_xor(true, Ordering::SeqCst));
    /// assert_eq!(false, foo.load(Ordering::SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_xor(false, Ordering::SeqCst));
    /// assert_eq!(false, foo.load(Ordering::SeqCst));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_xor(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_xor(self.v.get(), val, order) > 0 }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
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
    pub fn new(v: isize) -> AtomicIsize {
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
    /// let value = some_isize.load(Ordering::Relaxed);
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
    /// let value = some_isize.swap(10, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    /// Stores a value into the isize if the current value is the same as the expected value.
    ///
    /// If the return value is equal to `old` then the value was updated.
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
    /// let value = some_isize.compare_and_swap(5, 10, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, old: isize, new: isize, order: Ordering) -> isize {
        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) }
    }

    /// Add an isize to the current value, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicIsize, Ordering};
    ///
    /// let foo = AtomicIsize::new(0);
    /// assert_eq!(0, foo.fetch_add(10, Ordering::SeqCst));
    /// assert_eq!(10, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0, foo.fetch_sub(10, Ordering::SeqCst));
    /// assert_eq!(-10, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0b101101, foo.fetch_and(0b110011, Ordering::SeqCst));
    /// assert_eq!(0b100001, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0b101101, foo.fetch_or(0b110011, Ordering::SeqCst));
    /// assert_eq!(0b111111, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0b101101, foo.fetch_xor(0b110011, Ordering::SeqCst));
    /// assert_eq!(0b011110, foo.load(Ordering::SeqCst));
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn fetch_xor(&self, val: isize, order: Ordering) -> isize {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
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
    pub fn new(v: usize) -> AtomicUsize {
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
    /// let value = some_usize.load(Ordering::Relaxed);
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
    /// let value = some_usize.swap(10, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&self, val: usize, order: Ordering) -> usize {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    /// Stores a value into the usize if the current value is the same as the expected value.
    ///
    /// If the return value is equal to `old` then the value was updated.
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
    /// let value = some_usize.compare_and_swap(5, 10, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn compare_and_swap(&self, old: usize, new: usize, order: Ordering) -> usize {
        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) }
    }

    /// Add to the current usize, returning the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let foo = AtomicUsize::new(0);
    /// assert_eq!(0, foo.fetch_add(10, Ordering::SeqCst));
    /// assert_eq!(10, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(10, foo.fetch_sub(10, Ordering::SeqCst));
    /// assert_eq!(0, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0b101101, foo.fetch_and(0b110011, Ordering::SeqCst));
    /// assert_eq!(0b100001, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0b101101, foo.fetch_or(0b110011, Ordering::SeqCst));
    /// assert_eq!(0b111111, foo.load(Ordering::SeqCst));
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
    /// assert_eq!(0b101101, foo.fetch_xor(0b110011, Ordering::SeqCst));
    /// assert_eq!(0b011110, foo.load(Ordering::SeqCst));
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
    pub fn new(p: *mut T) -> AtomicPtr<T> {
        AtomicPtr { p: UnsafeCell::new(p as usize),
                    _marker: PhantomData }
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
            atomic_load(self.p.get(), order) as *mut T
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
        unsafe { atomic_store(self.p.get(), ptr as usize, order); }
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
        unsafe { atomic_swap(self.p.get(), ptr as usize, order) as *mut T }
    }

    /// Stores a value into the pointer if the current value is the same as the expected value.
    ///
    /// If the return value is equal to `old` then the value was updated.
    ///
    /// `compare_and_swap` also takes an `Ordering` argument which describes the memory ordering of
    /// this operation.
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
    pub fn compare_and_swap(&self, old: *mut T, new: *mut T, order: Ordering) -> *mut T {
        unsafe {
            atomic_compare_and_swap(self.p.get(), old as usize,
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
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

#[unstable(feature = "core")]
#[deprecated(since = "1.0.0",
             reason = "renamed to AtomicIsize")]
#[allow(missing_docs)]
pub struct AtomicInt {
    v: UnsafeCell<int>,
}

unsafe impl Sync for AtomicInt {}

#[unstable(feature = "core")]
#[deprecated(since = "1.0.0",
             reason = "renamed to AtomicUsize")]
#[allow(missing_docs)]
pub struct AtomicUint {
    v: UnsafeCell<uint>,
}

unsafe impl Sync for AtomicUint {}

#[unstable(feature = "core")]
#[deprecated(since = "1.0.0",
             reason = "use ATOMIC_ISIZE_INIT instead")]
#[allow(missing_docs, deprecated)]
pub const ATOMIC_INT_INIT: AtomicInt =
        AtomicInt { v: UnsafeCell { value: 0 } };
#[unstable(feature = "core")]
#[deprecated(since = "1.0.0",
             reason = "use ATOMIC_USIZE_INIT instead")]
#[allow(missing_docs, deprecated)]
pub const ATOMIC_UINT_INIT: AtomicUint =
        AtomicUint { v: UnsafeCell { value: 0, } };

#[allow(missing_docs, deprecated)]
impl AtomicInt {
    #[inline]
    pub fn new(v: int) -> AtomicInt {
        AtomicInt {v: UnsafeCell::new(v)}
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> int {
        unsafe { atomic_load(self.v.get(), order) }
    }

    #[inline]
    pub fn store(&self, val: int, order: Ordering) {
        unsafe { atomic_store(self.v.get(), val, order); }
    }

    #[inline]
    pub fn swap(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    #[inline]
    pub fn compare_and_swap(&self, old: int, new: int, order: Ordering) -> int {
        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) }
    }

    #[inline]
    pub fn fetch_add(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_add(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_sub(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_sub(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_and(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_and(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_or(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_or(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_xor(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}

#[allow(missing_docs, deprecated)]
impl AtomicUint {
    #[inline]
    pub fn new(v: uint) -> AtomicUint {
        AtomicUint { v: UnsafeCell::new(v) }
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> uint {
        unsafe { atomic_load(self.v.get(), order) }
    }

    #[inline]
    pub fn store(&self, val: uint, order: Ordering) {
        unsafe { atomic_store(self.v.get(), val, order); }
    }

    #[inline]
    pub fn swap(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    #[inline]
    pub fn compare_and_swap(&self, old: uint, new: uint, order: Ordering) -> uint {
        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) }
    }

    #[inline]
    pub fn fetch_add(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_add(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_sub(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_sub(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_and(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_and(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_or(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_or(self.v.get(), val, order) }
    }

    #[inline]
    pub fn fetch_xor(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}
