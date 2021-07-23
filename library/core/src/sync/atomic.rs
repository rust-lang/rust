//! Atomic types
//!
//! Atomic types provide primitive shared-memory communication between
//! threads, and are the building blocks of other concurrent
//! types.
//!
//! This module defines atomic versions of a select number of primitive
//! types, including [`AtomicBool`], [`AtomicIsize`], [`AtomicUsize`],
//! [`AtomicI8`], [`AtomicU16`], etc.
//! Atomic types present operations that, when used correctly, synchronize
//! updates between threads.
//!
//! Each method takes an [`Ordering`] which represents the strength of
//! the memory barrier for that operation. These orderings are the
//! same as the [C++20 atomic orderings][1]. For more information see the [nomicon][2].
//!
//! [1]: https://en.cppreference.com/w/cpp/atomic/memory_order
//! [2]: ../../../nomicon/atomics.html
//!
//! Atomic variables are safe to share between threads (they implement [`Sync`])
//! but they do not themselves provide the mechanism for sharing and follow the
//! [threading model](../../../std/thread/index.html#the-threading-model) of Rust.
//! The most common way to share an atomic variable is to put it into an [`Arc`][arc] (an
//! atomically-reference-counted shared pointer).
//!
//! [arc]: ../../../std/sync/struct.Arc.html
//!
//! Atomic types may be stored in static variables, initialized using
//! the constant initializers like [`AtomicBool::new`]. Atomic statics
//! are often used for lazy global initialization.
//!
//! # Portability
//!
//! All atomic types in this module are guaranteed to be [lock-free] if they're
//! available. This means they don't internally acquire a global mutex. Atomic
//! types and operations are not guaranteed to be wait-free. This means that
//! operations like `fetch_or` may be implemented with a compare-and-swap loop.
//!
//! Atomic operations may be implemented at the instruction layer with
//! larger-size atomics. For example some platforms use 4-byte atomic
//! instructions to implement `AtomicI8`. Note that this emulation should not
//! have an impact on correctness of code, it's just something to be aware of.
//!
//! The atomic types in this module might not be available on all platforms. The
//! atomic types here are all widely available, however, and can generally be
//! relied upon existing. Some notable exceptions are:
//!
//! * PowerPC and MIPS platforms with 32-bit pointers do not have `AtomicU64` or
//!   `AtomicI64` types.
//! * ARM platforms like `armv5te` that aren't for Linux only provide `load`
//!   and `store` operations, and do not support Compare and Swap (CAS)
//!   operations, such as `swap`, `fetch_add`, etc. Additionally on Linux,
//!   these CAS operations are implemented via [operating system support], which
//!   may come with a performance penalty.
//! * ARM targets with `thumbv6m` only provide `load` and `store` operations,
//!   and do not support Compare and Swap (CAS) operations, such as `swap`,
//!   `fetch_add`, etc.
//!
//! [operating system support]: https://www.kernel.org/doc/Documentation/arm/kernel_user_helpers.txt
//!
//! Note that future platforms may be added that also do not have support for
//! some atomic operations. Maximally portable code will want to be careful
//! about which atomic types are used. `AtomicUsize` and `AtomicIsize` are
//! generally the most portable, but even then they're not available everywhere.
//! For reference, the `std` library requires pointer-sized atomics, although
//! `core` does not.
//!
//! Currently you'll need to use `#[cfg(target_arch)]` primarily to
//! conditionally compile in code with atomics. There is an unstable
//! `#[cfg(target_has_atomic)]` as well which may be stabilized in the future.
//!
//! [lock-free]: https://en.wikipedia.org/wiki/Non-blocking_algorithm
//!
//! # Examples
//!
//! A simple spinlock:
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use std::{hint, thread};
//!
//! fn main() {
//!     let spinlock = Arc::new(AtomicUsize::new(1));
//!
//!     let spinlock_clone = Arc::clone(&spinlock);
//!     let thread = thread::spawn(move|| {
//!         spinlock_clone.store(0, Ordering::SeqCst);
//!     });
//!
//!     // Wait for the other thread to release the lock
//!     while spinlock.load(Ordering::SeqCst) != 0 {
//!         hint::spin_loop();
//!     }
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
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);
//!
//! let old_thread_count = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
//! println!("live threads: {}", old_thread_count + 1);
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![cfg_attr(not(target_has_atomic_load_store = "8"), allow(dead_code))]
#![cfg_attr(not(target_has_atomic_load_store = "8"), allow(unused_imports))]

use self::Ordering::*;

use crate::cell::UnsafeCell;
use crate::fmt;
use crate::intrinsics;

use crate::hint::spin_loop;

/// A boolean type which can be safely shared between threads.
///
/// This type has the same in-memory representation as a [`bool`].
///
/// **Note**: This type is only available on platforms that support atomic
/// loads and stores of `u8`.
#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
#[repr(C, align(1))]
pub struct AtomicBool {
    v: UnsafeCell<u8>,
}

#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
impl Default for AtomicBool {
    /// Creates an `AtomicBool` initialized to `false`.
    #[inline]
    fn default() -> Self {
        Self::new(false)
    }
}

// Send is implicitly implemented for AtomicBool.
#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for AtomicBool {}

/// A raw pointer type which can be safely shared between threads.
///
/// This type has the same in-memory representation as a `*mut T`.
///
/// **Note**: This type is only available on platforms that support atomic
/// loads and stores of pointers. Its size depends on the target pointer's size.
#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(target_pointer_width = "16", repr(C, align(2)))]
#[cfg_attr(target_pointer_width = "32", repr(C, align(4)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(8)))]
pub struct AtomicPtr<T> {
    p: UnsafeCell<*mut T>,
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for AtomicPtr<T> {
    /// Creates a null `AtomicPtr<T>`.
    fn default() -> AtomicPtr<T> {
        AtomicPtr::new(crate::ptr::null_mut())
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T> Send for AtomicPtr<T> {}
#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T> Sync for AtomicPtr<T> {}

/// Atomic memory orderings
///
/// Memory orderings specify the way atomic operations synchronize memory.
/// In its weakest [`Ordering::Relaxed`], only the memory directly touched by the
/// operation is synchronized. On the other hand, a store-load pair of [`Ordering::SeqCst`]
/// operations synchronize other memory while additionally preserving a total order of such
/// operations across all threads.
///
/// Rust's memory orderings are [the same as those of
/// C++20](https://en.cppreference.com/w/cpp/atomic/memory_order).
///
/// For more information see the [nomicon].
///
/// [nomicon]: ../../../nomicon/atomics.html
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Ordering {
    /// No ordering constraints, only atomic operations.
    ///
    /// Corresponds to [`memory_order_relaxed`] in C++20.
    ///
    /// [`memory_order_relaxed`]: https://en.cppreference.com/w/cpp/atomic/memory_order#Relaxed_ordering
    #[stable(feature = "rust1", since = "1.0.0")]
    Relaxed,
    /// When coupled with a store, all previous operations become ordered
    /// before any load of this value with [`Acquire`] (or stronger) ordering.
    /// In particular, all previous writes become visible to all threads
    /// that perform an [`Acquire`] (or stronger) load of this value.
    ///
    /// Notice that using this ordering for an operation that combines loads
    /// and stores leads to a [`Relaxed`] load operation!
    ///
    /// This ordering is only applicable for operations that can perform a store.
    ///
    /// Corresponds to [`memory_order_release`] in C++20.
    ///
    /// [`memory_order_release`]: https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
    #[stable(feature = "rust1", since = "1.0.0")]
    Release,
    /// When coupled with a load, if the loaded value was written by a store operation with
    /// [`Release`] (or stronger) ordering, then all subsequent operations
    /// become ordered after that store. In particular, all subsequent loads will see data
    /// written before the store.
    ///
    /// Notice that using this ordering for an operation that combines loads
    /// and stores leads to a [`Relaxed`] store operation!
    ///
    /// This ordering is only applicable for operations that can perform a load.
    ///
    /// Corresponds to [`memory_order_acquire`] in C++20.
    ///
    /// [`memory_order_acquire`]: https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
    #[stable(feature = "rust1", since = "1.0.0")]
    Acquire,
    /// Has the effects of both [`Acquire`] and [`Release`] together:
    /// For loads it uses [`Acquire`] ordering. For stores it uses the [`Release`] ordering.
    ///
    /// Notice that in the case of `compare_and_swap`, it is possible that the operation ends up
    /// not performing any store and hence it has just [`Acquire`] ordering. However,
    /// `AcqRel` will never perform [`Relaxed`] accesses.
    ///
    /// This ordering is only applicable for operations that combine both loads and stores.
    ///
    /// Corresponds to [`memory_order_acq_rel`] in C++20.
    ///
    /// [`memory_order_acq_rel`]: https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
    #[stable(feature = "rust1", since = "1.0.0")]
    AcqRel,
    /// Like [`Acquire`]/[`Release`]/[`AcqRel`] (for load, store, and load-with-store
    /// operations, respectively) with the additional guarantee that all threads see all
    /// sequentially consistent operations in the same order.
    ///
    /// Corresponds to [`memory_order_seq_cst`] in C++20.
    ///
    /// [`memory_order_seq_cst`]: https://en.cppreference.com/w/cpp/atomic/memory_order#Sequentially-consistent_ordering
    #[stable(feature = "rust1", since = "1.0.0")]
    SeqCst,
}

/// An [`AtomicBool`] initialized to `false`.
#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(
    since = "1.34.0",
    reason = "the `new` function is now preferred",
    suggestion = "AtomicBool::new(false)"
)]
pub const ATOMIC_BOOL_INIT: AtomicBool = AtomicBool::new(false);

#[cfg(target_has_atomic_load_store = "8")]
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
    #[rustc_const_stable(feature = "const_atomic_new", since = "1.24.0")]
    pub const fn new(v: bool) -> AtomicBool {
        AtomicBool { v: UnsafeCell::new(v as u8) }
    }

    /// Returns a mutable reference to the underlying [`bool`].
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
        // SAFETY: the mutable reference guarantees unique ownership.
        unsafe { &mut *(self.v.get() as *mut bool) }
    }

    /// Get atomic access to a `&mut bool`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(atomic_from_mut)]
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let mut some_bool = true;
    /// let a = AtomicBool::from_mut(&mut some_bool);
    /// a.store(false, Ordering::Relaxed);
    /// assert_eq!(some_bool, false);
    /// ```
    #[inline]
    #[cfg(target_has_atomic_equal_alignment = "8")]
    #[unstable(feature = "atomic_from_mut", issue = "76314")]
    pub fn from_mut(v: &mut bool) -> &Self {
        // SAFETY: the mutable reference guarantees unique ownership, and
        // alignment of both `bool` and `Self` is 1.
        unsafe { &*(v as *mut bool as *mut Self) }
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
    #[rustc_const_unstable(feature = "const_cell_into_inner", issue = "78729")]
    pub const fn into_inner(self) -> bool {
        self.v.into_inner() != 0
    }

    /// Loads a value from the bool.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. Possible values are [`SeqCst`], [`Acquire`] and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Release`] or [`AcqRel`].
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
        // SAFETY: any data races are prevented by atomic intrinsics and the raw
        // pointer passed in is valid because we got it from a reference.
        unsafe { atomic_load(self.v.get(), order) != 0 }
    }

    /// Stores a value into the bool.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Acquire`] or [`AcqRel`].
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn store(&self, val: bool, order: Ordering) {
        // SAFETY: any data races are prevented by atomic intrinsics and the raw
        // pointer passed in is valid because we got it from a reference.
        unsafe {
            atomic_store(self.v.get(), val as u8, order);
        }
    }

    /// Stores a value into the bool, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[cfg(target_has_atomic = "8")]
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_swap(self.v.get(), val as u8, order) != 0 }
    }

    /// Stores a value into the [`bool`] if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. Notice that even when using [`AcqRel`], the operation
    /// might fail and hence just perform an `Acquire` load, but not have `Release` semantics.
    /// Using [`Acquire`] makes the store part of this operation [`Relaxed`] if it
    /// happens, and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
    ///
    /// # Migrating to `compare_exchange` and `compare_exchange_weak`
    ///
    /// `compare_and_swap` is equivalent to `compare_exchange` with the following mapping for
    /// memory orderings:
    ///
    /// Original | Success | Failure
    /// -------- | ------- | -------
    /// Relaxed  | Relaxed | Relaxed
    /// Acquire  | Acquire | Acquire
    /// Release  | Release | Relaxed
    /// AcqRel   | AcqRel  | Acquire
    /// SeqCst   | SeqCst  | SeqCst
    ///
    /// `compare_exchange_weak` is allowed to fail spuriously even when the comparison succeeds,
    /// which allows the compiler to generate better assembly code when the compare and swap
    /// is used in a loop.
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
    #[rustc_deprecated(
        since = "1.50.0",
        reason = "Use `compare_exchange` or `compare_exchange_weak` instead"
    )]
    #[cfg(target_has_atomic = "8")]
    pub fn compare_and_swap(&self, current: bool, new: bool, order: Ordering) -> bool {
        match self.compare_exchange(current, new, order, strongest_failure_ordering(order)) {
            Ok(x) => x,
            Err(x) => x,
        }
    }

    /// Stores a value into the [`bool`] if the current value is the same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was written and containing
    /// the previous value. On success this value is guaranteed to be equal to `current`.
    ///
    /// `compare_exchange` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. `success` describes the required ordering for the
    /// read-modify-write operation that takes place if the comparison with `current` succeeds.
    /// `failure` describes the required ordering for the load operation that takes place when
    /// the comparison fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[doc(alias = "compare_and_swap")]
    #[cfg(target_has_atomic = "8")]
    pub fn compare_exchange(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        // SAFETY: data races are prevented by atomic intrinsics.
        match unsafe {
            atomic_compare_exchange(self.v.get(), current as u8, new as u8, success, failure)
        } {
            Ok(x) => Ok(x != 0),
            Err(x) => Err(x != 0),
        }
    }

    /// Stores a value into the [`bool`] if the current value is the same as the `current` value.
    ///
    /// Unlike [`AtomicBool::compare_exchange`], this function is allowed to spuriously fail even when the
    /// comparison succeeds, which can result in more efficient code on some platforms. The
    /// return value is a result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. `success` describes the required ordering for the
    /// read-modify-write operation that takes place if the comparison with `current` succeeds.
    /// `failure` describes the required ordering for the load operation that takes place when
    /// the comparison fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[doc(alias = "compare_and_swap")]
    #[cfg(target_has_atomic = "8")]
    pub fn compare_exchange_weak(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        // SAFETY: data races are prevented by atomic intrinsics.
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
    /// `fetch_and` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[cfg(target_has_atomic = "8")]
    pub fn fetch_and(&self, val: bool, order: Ordering) -> bool {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_and(self.v.get(), val as u8, order) != 0 }
    }

    /// Logical "nand" with a boolean value.
    ///
    /// Performs a logical "nand" operation on the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// `fetch_nand` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[cfg(target_has_atomic = "8")]
    pub fn fetch_nand(&self, val: bool, order: Ordering) -> bool {
        // We can't use atomic_nand here because it can result in a bool with
        // an invalid value. This happens because the atomic operation is done
        // with an 8-bit integer internally, which would set the upper 7 bits.
        // So we just use fetch_xor or swap instead.
        if val {
            // !(x & true) == !x
            // We must invert the bool.
            self.fetch_xor(true, order)
        } else {
            // !(x & false) == true
            // We must set the bool to true.
            self.swap(true, order)
        }
    }

    /// Logical "or" with a boolean value.
    ///
    /// Performs a logical "or" operation on the current value and the argument `val`, and sets the
    /// new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// `fetch_or` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[cfg(target_has_atomic = "8")]
    pub fn fetch_or(&self, val: bool, order: Ordering) -> bool {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_or(self.v.get(), val as u8, order) != 0 }
    }

    /// Logical "xor" with a boolean value.
    ///
    /// Performs a logical "xor" operation on the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// `fetch_xor` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
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
    #[cfg(target_has_atomic = "8")]
    pub fn fetch_xor(&self, val: bool, order: Ordering) -> bool {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_xor(self.v.get(), val as u8, order) != 0 }
    }

    /// Returns a mutable pointer to the underlying [`bool`].
    ///
    /// Doing non-atomic reads and writes on the resulting integer can be a data race.
    /// This method is mostly useful for FFI, where the function signature may use
    /// `*mut bool` instead of `&AtomicBool`.
    ///
    /// Returning an `*mut` pointer from a shared reference to this atomic is safe because the
    /// atomic types work with interior mutability. All modifications of an atomic change the value
    /// through a shared reference, and can do so safely as long as they use atomic operations. Any
    /// use of the returned raw pointer requires an `unsafe` block and still has to uphold the same
    /// restriction: operations on it must be atomic.
    ///
    /// # Examples
    ///
    /// ```ignore (extern-declaration)
    /// # fn main() {
    /// use std::sync::atomic::AtomicBool;
    /// extern "C" {
    ///     fn my_atomic_op(arg: *mut bool);
    /// }
    ///
    /// let mut atomic = AtomicBool::new(true);
    /// unsafe {
    ///     my_atomic_op(atomic.as_mut_ptr());
    /// }
    /// # }
    /// ```
    #[inline]
    #[unstable(feature = "atomic_mut_ptr", reason = "recently added", issue = "66893")]
    pub fn as_mut_ptr(&self) -> *mut bool {
        self.v.get() as *mut bool
    }

    /// Fetches the value, and applies a function to it that returns an optional
    /// new value. Returns a `Result` of `Ok(previous_value)` if the function
    /// returned `Some(_)`, else `Err(previous_value)`.
    ///
    /// Note: This may call the function multiple times if the value has been
    /// changed from other threads in the meantime, as long as the function
    /// returns `Some(_)`, but the function will have been applied only once to
    /// the stored value.
    ///
    /// `fetch_update` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering for
    /// when the operation finally succeeds while the second describes the
    /// required ordering for loads. These correspond to the success and failure
    /// orderings of [`AtomicBool::compare_exchange`] respectively.
    ///
    /// Using [`Acquire`] as success ordering makes the store part of this
    /// operation [`Relaxed`], and using [`Release`] makes the final successful
    /// load [`Relaxed`]. The (failed) load ordering can only be [`SeqCst`],
    /// [`Acquire`] or [`Relaxed`] and must be equivalent to or weaker than the
    /// success ordering.
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let x = AtomicBool::new(false);
    /// assert_eq!(x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |_| None), Err(false));
    /// assert_eq!(x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(!x)), Ok(false));
    /// assert_eq!(x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(!x)), Ok(true));
    /// assert_eq!(x.load(Ordering::SeqCst), false);
    /// ```
    #[inline]
    #[stable(feature = "atomic_fetch_update", since = "1.53.0")]
    #[cfg(target_has_atomic = "8")]
    pub fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
    ) -> Result<bool, bool>
    where
        F: FnMut(bool) -> Option<bool>,
    {
        let mut prev = self.load(fetch_order);
        while let Some(next) = f(prev) {
            match self.compare_exchange_weak(prev, next, set_order, fetch_order) {
                x @ Ok(_) => return x,
                Err(next_prev) => prev = next_prev,
            }
        }
        Err(prev)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
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
    #[rustc_const_stable(feature = "const_atomic_new", since = "1.24.0")]
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
    /// let mut data = 10;
    /// let mut atomic_ptr = AtomicPtr::new(&mut data);
    /// let mut other_data = 5;
    /// *atomic_ptr.get_mut() = &mut other_data;
    /// assert_eq!(unsafe { *atomic_ptr.load(Ordering::SeqCst) }, 5);
    /// ```
    #[inline]
    #[stable(feature = "atomic_access", since = "1.15.0")]
    pub fn get_mut(&mut self) -> &mut *mut T {
        self.p.get_mut()
    }

    /// Get atomic access to a pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(atomic_from_mut)]
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let mut data = 123;
    /// let mut some_ptr = &mut data as *mut i32;
    /// let a = AtomicPtr::from_mut(&mut some_ptr);
    /// let mut other_data = 456;
    /// a.store(&mut other_data, Ordering::Relaxed);
    /// assert_eq!(unsafe { *some_ptr }, 456);
    /// ```
    #[inline]
    #[cfg(target_has_atomic_equal_alignment = "ptr")]
    #[unstable(feature = "atomic_from_mut", issue = "76314")]
    pub fn from_mut(v: &mut *mut T) -> &Self {
        use crate::mem::align_of;
        let [] = [(); align_of::<AtomicPtr<()>>() - align_of::<*mut ()>()];
        // SAFETY:
        //  - the mutable reference guarantees unique ownership.
        //  - the alignment of `*mut T` and `Self` is the same on all platforms
        //    supported by rust, as verified above.
        unsafe { &*(v as *mut *mut T as *mut Self) }
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
    /// let mut data = 5;
    /// let atomic_ptr = AtomicPtr::new(&mut data);
    /// assert_eq!(unsafe { *atomic_ptr.into_inner() }, 5);
    /// ```
    #[inline]
    #[stable(feature = "atomic_access", since = "1.15.0")]
    #[rustc_const_unstable(feature = "const_cell_into_inner", issue = "78729")]
    pub const fn into_inner(self) -> *mut T {
        self.p.into_inner()
    }

    /// Loads a value from the pointer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. Possible values are [`SeqCst`], [`Acquire`] and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Release`] or [`AcqRel`].
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
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_load(self.p.get(), order) }
    }

    /// Stores a value into the pointer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Acquire`] or [`AcqRel`].
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn store(&self, ptr: *mut T, order: Ordering) {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe {
            atomic_store(self.p.get(), ptr, order);
        }
    }

    /// Stores a value into the pointer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on pointers.
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
    #[cfg(target_has_atomic = "ptr")]
    pub fn swap(&self, ptr: *mut T, order: Ordering) -> *mut T {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_swap(self.p.get(), ptr, order) }
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to `current`, then the value
    /// was updated.
    ///
    /// `compare_and_swap` also takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. Notice that even when using [`AcqRel`], the operation
    /// might fail and hence just perform an `Acquire` load, but not have `Release` semantics.
    /// Using [`Acquire`] makes the store part of this operation [`Relaxed`] if it
    /// happens, and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on pointers.
    ///
    /// # Migrating to `compare_exchange` and `compare_exchange_weak`
    ///
    /// `compare_and_swap` is equivalent to `compare_exchange` with the following mapping for
    /// memory orderings:
    ///
    /// Original | Success | Failure
    /// -------- | ------- | -------
    /// Relaxed  | Relaxed | Relaxed
    /// Acquire  | Acquire | Acquire
    /// Release  | Release | Relaxed
    /// AcqRel   | AcqRel  | Acquire
    /// SeqCst   | SeqCst  | SeqCst
    ///
    /// `compare_exchange_weak` is allowed to fail spuriously even when the comparison succeeds,
    /// which allows the compiler to generate better assembly code when the compare and swap
    /// is used in a loop.
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
    ///
    /// let value = some_ptr.compare_and_swap(ptr, other_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(
        since = "1.50.0",
        reason = "Use `compare_exchange` or `compare_exchange_weak` instead"
    )]
    #[cfg(target_has_atomic = "ptr")]
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
    /// ordering of this operation. `success` describes the required ordering for the
    /// read-modify-write operation that takes place if the comparison with `current` succeeds.
    /// `failure` describes the required ordering for the load operation that takes place when
    /// the comparison fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on pointers.
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
    ///
    /// let value = some_ptr.compare_exchange(ptr, other_ptr,
    ///                                       Ordering::SeqCst, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "extended_compare_and_swap", since = "1.10.0")]
    #[cfg(target_has_atomic = "ptr")]
    pub fn compare_exchange(
        &self,
        current: *mut T,
        new: *mut T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<*mut T, *mut T> {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_compare_exchange(self.p.get(), current, new, success, failure) }
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    ///
    /// Unlike [`AtomicPtr::compare_exchange`], this function is allowed to spuriously fail even when the
    /// comparison succeeds, which can result in more efficient code on some platforms. The
    /// return value is a result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. `success` describes the required ordering for the
    /// read-modify-write operation that takes place if the comparison with `current` succeeds.
    /// `failure` describes the required ordering for the load operation that takes place when
    /// the comparison fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on pointers.
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
    #[cfg(target_has_atomic = "ptr")]
    pub fn compare_exchange_weak(
        &self,
        current: *mut T,
        new: *mut T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<*mut T, *mut T> {
        // SAFETY: This intrinsic is unsafe because it operates on a raw pointer
        // but we know for sure that the pointer is valid (we just got it from
        // an `UnsafeCell` that we have by reference) and the atomic operation
        // itself allows us to safely mutate the `UnsafeCell` contents.
        unsafe { atomic_compare_exchange_weak(self.p.get(), current, new, success, failure) }
    }

    /// Fetches the value, and applies a function to it that returns an optional
    /// new value. Returns a `Result` of `Ok(previous_value)` if the function
    /// returned `Some(_)`, else `Err(previous_value)`.
    ///
    /// Note: This may call the function multiple times if the value has been
    /// changed from other threads in the meantime, as long as the function
    /// returns `Some(_)`, but the function will have been applied only once to
    /// the stored value.
    ///
    /// `fetch_update` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering for
    /// when the operation finally succeeds while the second describes the
    /// required ordering for loads. These correspond to the success and failure
    /// orderings of [`AtomicPtr::compare_exchange`] respectively.
    ///
    /// Using [`Acquire`] as success ordering makes the store part of this
    /// operation [`Relaxed`], and using [`Release`] makes the final successful
    /// load [`Relaxed`]. The (failed) load ordering can only be [`SeqCst`],
    /// [`Acquire`] or [`Relaxed`] and must be equivalent to or weaker than the
    /// success ordering.
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on pointers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let ptr: *mut _ = &mut 5;
    /// let some_ptr = AtomicPtr::new(ptr);
    ///
    /// let new: *mut _ = &mut 10;
    /// assert_eq!(some_ptr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |_| None), Err(ptr));
    /// let result = some_ptr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
    ///     if x == ptr {
    ///         Some(new)
    ///     } else {
    ///         None
    ///     }
    /// });
    /// assert_eq!(result, Ok(ptr));
    /// assert_eq!(some_ptr.load(Ordering::SeqCst), new);
    /// ```
    #[inline]
    #[stable(feature = "atomic_fetch_update", since = "1.53.0")]
    #[cfg(target_has_atomic = "ptr")]
    pub fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
    ) -> Result<*mut T, *mut T>
    where
        F: FnMut(*mut T) -> Option<*mut T>,
    {
        let mut prev = self.load(fetch_order);
        while let Some(next) = f(prev) {
            match self.compare_exchange_weak(prev, next, set_order, fetch_order) {
                x @ Ok(_) => return x,
                Err(next_prev) => prev = next_prev,
            }
        }
        Err(prev)
    }
}

#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "atomic_bool_from", since = "1.24.0")]
impl From<bool> for AtomicBool {
    /// Converts a `bool` into an `AtomicBool`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::AtomicBool;
    /// let atomic_bool = AtomicBool::from(true);
    /// assert_eq!(format!("{:?}", atomic_bool), "true")
    /// ```
    #[inline]
    fn from(b: bool) -> Self {
        Self::new(b)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "atomic_from", since = "1.23.0")]
impl<T> From<*mut T> for AtomicPtr<T> {
    #[inline]
    fn from(p: *mut T) -> Self {
        Self::new(p)
    }
}

#[allow(unused_macros)] // This macro ends up being unused on some architectures.
macro_rules! if_not_8_bit {
    (u8, $($tt:tt)*) => { "" };
    (i8, $($tt:tt)*) => { "" };
    ($_:ident, $($tt:tt)*) => { $($tt)* };
}

#[cfg(target_has_atomic_load_store = "8")]
macro_rules! atomic_int {
    ($cfg_cas:meta,
     $cfg_align:meta,
     $stable:meta,
     $stable_cxchg:meta,
     $stable_debug:meta,
     $stable_access:meta,
     $stable_from:meta,
     $stable_nand:meta,
     $const_stable:meta,
     $stable_init_const:meta,
     $s_int_type:literal,
     $extra_feature:expr,
     $min_fn:ident, $max_fn:ident,
     $align:expr,
     $atomic_new:expr,
     $int_type:ident $atomic_type:ident $atomic_init:ident) => {
        /// An integer type which can be safely shared between threads.
        ///
        /// This type has the same in-memory representation as the underlying
        /// integer type, [`
        #[doc = $s_int_type]
        /// `]. For more about the differences between atomic types and
        /// non-atomic types as well as information about the portability of
        /// this type, please see the [module-level documentation].
        ///
        /// **Note:** This type is only available on platforms that support
        /// atomic loads and stores of [`
        #[doc = $s_int_type]
        /// `].
        ///
        /// [module-level documentation]: crate::sync::atomic
        #[$stable]
        #[repr(C, align($align))]
        pub struct $atomic_type {
            v: UnsafeCell<$int_type>,
        }

        /// An atomic integer initialized to `0`.
        #[$stable_init_const]
        #[rustc_deprecated(
            since = "1.34.0",
            reason = "the `new` function is now preferred",
            suggestion = $atomic_new,
        )]
        pub const $atomic_init: $atomic_type = $atomic_type::new(0);

        #[$stable]
        impl Default for $atomic_type {
            #[inline]
            fn default() -> Self {
                Self::new(Default::default())
            }
        }

        #[$stable_from]
        impl From<$int_type> for $atomic_type {
            #[doc = concat!("Converts an `", stringify!($int_type), "` into an `", stringify!($atomic_type), "`.")]
            #[inline]
            fn from(v: $int_type) -> Self { Self::new(v) }
        }

        #[$stable_debug]
        impl fmt::Debug for $atomic_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.load(Ordering::SeqCst), f)
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
            #[doc = concat!($extra_feature, "use std::sync::atomic::", stringify!($atomic_type), ";")]
            ///
            #[doc = concat!("let atomic_forty_two = ", stringify!($atomic_type), "::new(42);")]
            /// ```
            #[inline]
            #[$stable]
            #[$const_stable]
            pub const fn new(v: $int_type) -> Self {
                Self {v: UnsafeCell::new(v)}
            }

            /// Returns a mutable reference to the underlying integer.
            ///
            /// This is safe because the mutable reference guarantees that no other threads are
            /// concurrently accessing the atomic data.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let mut some_var = ", stringify!($atomic_type), "::new(10);")]
            /// assert_eq!(*some_var.get_mut(), 10);
            /// *some_var.get_mut() = 5;
            /// assert_eq!(some_var.load(Ordering::SeqCst), 5);
            /// ```
            #[inline]
            #[$stable_access]
            pub fn get_mut(&mut self) -> &mut $int_type {
                self.v.get_mut()
            }

            #[doc = concat!("Get atomic access to a `&mut ", stringify!($int_type), "`.")]
            ///
            #[doc = if_not_8_bit! {
                $int_type,
                concat!(
                    "**Note:** This function is only available on targets where `",
                    stringify!($int_type), "` has an alignment of ", $align, " bytes."
                )
            }]
            ///
            /// # Examples
            ///
            /// ```
            /// #![feature(atomic_from_mut)]
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            /// let mut some_int = 123;
            #[doc = concat!("let a = ", stringify!($atomic_type), "::from_mut(&mut some_int);")]
            /// a.store(100, Ordering::Relaxed);
            /// assert_eq!(some_int, 100);
            /// ```
            ///
            #[inline]
            #[$cfg_align]
            #[unstable(feature = "atomic_from_mut", issue = "76314")]
            pub fn from_mut(v: &mut $int_type) -> &Self {
                use crate::mem::align_of;
                let [] = [(); align_of::<Self>() - align_of::<$int_type>()];
                // SAFETY:
                //  - the mutable reference guarantees unique ownership.
                //  - the alignment of `$int_type` and `Self` is the
                //    same, as promised by $cfg_align and verified above.
                unsafe { &*(v as *mut $int_type as *mut Self) }
            }

            /// Consumes the atomic and returns the contained value.
            ///
            /// This is safe because passing `self` by value guarantees that no other threads are
            /// concurrently accessing the atomic data.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::", stringify!($atomic_type), ";")]
            ///
            #[doc = concat!("let some_var = ", stringify!($atomic_type), "::new(5);")]
            /// assert_eq!(some_var.into_inner(), 5);
            /// ```
            #[inline]
            #[$stable_access]
            #[rustc_const_unstable(feature = "const_cell_into_inner", issue = "78729")]
            pub const fn into_inner(self) -> $int_type {
                self.v.into_inner()
            }

            /// Loads a value from the atomic integer.
            ///
            /// `load` takes an [`Ordering`] argument which describes the memory ordering of this operation.
            /// Possible values are [`SeqCst`], [`Acquire`] and [`Relaxed`].
            ///
            /// # Panics
            ///
            /// Panics if `order` is [`Release`] or [`AcqRel`].
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let some_var = ", stringify!($atomic_type), "::new(5);")]
            ///
            /// assert_eq!(some_var.load(Ordering::Relaxed), 5);
            /// ```
            #[inline]
            #[$stable]
            pub fn load(&self, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_load(self.v.get(), order) }
            }

            /// Stores a value into the atomic integer.
            ///
            /// `store` takes an [`Ordering`] argument which describes the memory ordering of this operation.
            ///  Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].
            ///
            /// # Panics
            ///
            /// Panics if `order` is [`Acquire`] or [`AcqRel`].
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let some_var = ", stringify!($atomic_type), "::new(5);")]
            ///
            /// some_var.store(10, Ordering::Relaxed);
            /// assert_eq!(some_var.load(Ordering::Relaxed), 10);
            /// ```
            #[inline]
            #[$stable]
            pub fn store(&self, val: $int_type, order: Ordering) {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_store(self.v.get(), val, order); }
            }

            /// Stores a value into the atomic integer, returning the previous value.
            ///
            /// `swap` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let some_var = ", stringify!($atomic_type), "::new(5);")]
            ///
            /// assert_eq!(some_var.swap(10, Ordering::Relaxed), 5);
            /// ```
            #[inline]
            #[$stable]
            #[$cfg_cas]
            pub fn swap(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_swap(self.v.get(), val, order) }
            }

            /// Stores a value into the atomic integer if the current value is the same as
            /// the `current` value.
            ///
            /// The return value is always the previous value. If it is equal to `current`, then the
            /// value was updated.
            ///
            /// `compare_and_swap` also takes an [`Ordering`] argument which describes the memory
            /// ordering of this operation. Notice that even when using [`AcqRel`], the operation
            /// might fail and hence just perform an `Acquire` load, but not have `Release` semantics.
            /// Using [`Acquire`] makes the store part of this operation [`Relaxed`] if it
            /// happens, and using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Migrating to `compare_exchange` and `compare_exchange_weak`
            ///
            /// `compare_and_swap` is equivalent to `compare_exchange` with the following mapping for
            /// memory orderings:
            ///
            /// Original | Success | Failure
            /// -------- | ------- | -------
            /// Relaxed  | Relaxed | Relaxed
            /// Acquire  | Acquire | Acquire
            /// Release  | Release | Relaxed
            /// AcqRel   | AcqRel  | Acquire
            /// SeqCst   | SeqCst  | SeqCst
            ///
            /// `compare_exchange_weak` is allowed to fail spuriously even when the comparison succeeds,
            /// which allows the compiler to generate better assembly code when the compare and swap
            /// is used in a loop.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let some_var = ", stringify!($atomic_type), "::new(5);")]
            ///
            /// assert_eq!(some_var.compare_and_swap(5, 10, Ordering::Relaxed), 5);
            /// assert_eq!(some_var.load(Ordering::Relaxed), 10);
            ///
            /// assert_eq!(some_var.compare_and_swap(6, 12, Ordering::Relaxed), 10);
            /// assert_eq!(some_var.load(Ordering::Relaxed), 10);
            /// ```
            #[inline]
            #[$stable]
            #[rustc_deprecated(
                since = "1.50.0",
                reason = "Use `compare_exchange` or `compare_exchange_weak` instead")
            ]
            #[$cfg_cas]
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

            /// Stores a value into the atomic integer if the current value is the same as
            /// the `current` value.
            ///
            /// The return value is a result indicating whether the new value was written and
            /// containing the previous value. On success this value is guaranteed to be equal to
            /// `current`.
            ///
            /// `compare_exchange` takes two [`Ordering`] arguments to describe the memory
            /// ordering of this operation. `success` describes the required ordering for the
            /// read-modify-write operation that takes place if the comparison with `current` succeeds.
            /// `failure` describes the required ordering for the load operation that takes place when
            /// the comparison fails. Using [`Acquire`] as success ordering makes the store part
            /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
            /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
            /// and must be equivalent to or weaker than the success ordering.
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let some_var = ", stringify!($atomic_type), "::new(5);")]
            ///
            /// assert_eq!(some_var.compare_exchange(5, 10,
            ///                                      Ordering::Acquire,
            ///                                      Ordering::Relaxed),
            ///            Ok(5));
            /// assert_eq!(some_var.load(Ordering::Relaxed), 10);
            ///
            /// assert_eq!(some_var.compare_exchange(6, 12,
            ///                                      Ordering::SeqCst,
            ///                                      Ordering::Acquire),
            ///            Err(10));
            /// assert_eq!(some_var.load(Ordering::Relaxed), 10);
            /// ```
            #[inline]
            #[$stable_cxchg]
            #[$cfg_cas]
            pub fn compare_exchange(&self,
                                    current: $int_type,
                                    new: $int_type,
                                    success: Ordering,
                                    failure: Ordering) -> Result<$int_type, $int_type> {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_compare_exchange(self.v.get(), current, new, success, failure) }
            }

            /// Stores a value into the atomic integer if the current value is the same as
            /// the `current` value.
            ///
            #[doc = concat!("Unlike [`", stringify!($atomic_type), "::compare_exchange`],")]
            /// this function is allowed to spuriously fail even
            /// when the comparison succeeds, which can result in more efficient code on some
            /// platforms. The return value is a result indicating whether the new value was
            /// written and containing the previous value.
            ///
            /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
            /// ordering of this operation. `success` describes the required ordering for the
            /// read-modify-write operation that takes place if the comparison with `current` succeeds.
            /// `failure` describes the required ordering for the load operation that takes place when
            /// the comparison fails. Using [`Acquire`] as success ordering makes the store part
            /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
            /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
            /// and must be equivalent to or weaker than the success ordering.
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let val = ", stringify!($atomic_type), "::new(4);")]
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
            #[$cfg_cas]
            pub fn compare_exchange_weak(&self,
                                         current: $int_type,
                                         new: $int_type,
                                         success: Ordering,
                                         failure: Ordering) -> Result<$int_type, $int_type> {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe {
                    atomic_compare_exchange_weak(self.v.get(), current, new, success, failure)
                }
            }

            /// Adds to the current value, returning the previous value.
            ///
            /// This operation wraps around on overflow.
            ///
            /// `fetch_add` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(0);")]
            /// assert_eq!(foo.fetch_add(10, Ordering::SeqCst), 0);
            /// assert_eq!(foo.load(Ordering::SeqCst), 10);
            /// ```
            #[inline]
            #[$stable]
            #[$cfg_cas]
            pub fn fetch_add(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_add(self.v.get(), val, order) }
            }

            /// Subtracts from the current value, returning the previous value.
            ///
            /// This operation wraps around on overflow.
            ///
            /// `fetch_sub` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(20);")]
            /// assert_eq!(foo.fetch_sub(10, Ordering::SeqCst), 20);
            /// assert_eq!(foo.load(Ordering::SeqCst), 10);
            /// ```
            #[inline]
            #[$stable]
            #[$cfg_cas]
            pub fn fetch_sub(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_sub(self.v.get(), val, order) }
            }

            /// Bitwise "and" with the current value.
            ///
            /// Performs a bitwise "and" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// `fetch_and` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(0b101101);")]
            /// assert_eq!(foo.fetch_and(0b110011, Ordering::SeqCst), 0b101101);
            /// assert_eq!(foo.load(Ordering::SeqCst), 0b100001);
            /// ```
            #[inline]
            #[$stable]
            #[$cfg_cas]
            pub fn fetch_and(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_and(self.v.get(), val, order) }
            }

            /// Bitwise "nand" with the current value.
            ///
            /// Performs a bitwise "nand" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// `fetch_nand` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(0x13);")]
            /// assert_eq!(foo.fetch_nand(0x31, Ordering::SeqCst), 0x13);
            /// assert_eq!(foo.load(Ordering::SeqCst), !(0x13 & 0x31));
            /// ```
            #[inline]
            #[$stable_nand]
            #[$cfg_cas]
            pub fn fetch_nand(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_nand(self.v.get(), val, order) }
            }

            /// Bitwise "or" with the current value.
            ///
            /// Performs a bitwise "or" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// `fetch_or` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(0b101101);")]
            /// assert_eq!(foo.fetch_or(0b110011, Ordering::SeqCst), 0b101101);
            /// assert_eq!(foo.load(Ordering::SeqCst), 0b111111);
            /// ```
            #[inline]
            #[$stable]
            #[$cfg_cas]
            pub fn fetch_or(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_or(self.v.get(), val, order) }
            }

            /// Bitwise "xor" with the current value.
            ///
            /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// `fetch_xor` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(0b101101);")]
            /// assert_eq!(foo.fetch_xor(0b110011, Ordering::SeqCst), 0b101101);
            /// assert_eq!(foo.load(Ordering::SeqCst), 0b011110);
            /// ```
            #[inline]
            #[$stable]
            #[$cfg_cas]
            pub fn fetch_xor(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { atomic_xor(self.v.get(), val, order) }
            }

            /// Fetches the value, and applies a function to it that returns an optional
            /// new value. Returns a `Result` of `Ok(previous_value)` if the function returned `Some(_)`, else
            /// `Err(previous_value)`.
            ///
            /// Note: This may call the function multiple times if the value has been changed from other threads in
            /// the meantime, as long as the function returns `Some(_)`, but the function will have been applied
            /// only once to the stored value.
            ///
            /// `fetch_update` takes two [`Ordering`] arguments to describe the memory ordering of this operation.
            /// The first describes the required ordering for when the operation finally succeeds while the second
            /// describes the required ordering for loads. These correspond to the success and failure orderings of
            #[doc = concat!("[`", stringify!($atomic_type), "::compare_exchange`]")]
            /// respectively.
            ///
            /// Using [`Acquire`] as success ordering makes the store part
            /// of this operation [`Relaxed`], and using [`Release`] makes the final successful load
            /// [`Relaxed`]. The (failed) load ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
            /// and must be equivalent to or weaker than the success ordering.
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let x = ", stringify!($atomic_type), "::new(7);")]
            /// assert_eq!(x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |_| None), Err(7));
            /// assert_eq!(x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(x + 1)), Ok(7));
            /// assert_eq!(x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(x + 1)), Ok(8));
            /// assert_eq!(x.load(Ordering::SeqCst), 9);
            /// ```
            #[inline]
            #[stable(feature = "no_more_cas", since = "1.45.0")]
            #[$cfg_cas]
            pub fn fetch_update<F>(&self,
                                   set_order: Ordering,
                                   fetch_order: Ordering,
                                   mut f: F) -> Result<$int_type, $int_type>
            where F: FnMut($int_type) -> Option<$int_type> {
                let mut prev = self.load(fetch_order);
                while let Some(next) = f(prev) {
                    match self.compare_exchange_weak(prev, next, set_order, fetch_order) {
                        x @ Ok(_) => return x,
                        Err(next_prev) => prev = next_prev
                    }
                }
                Err(prev)
            }

            /// Maximum with the current value.
            ///
            /// Finds the maximum of the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// `fetch_max` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(23);")]
            /// assert_eq!(foo.fetch_max(42, Ordering::SeqCst), 23);
            /// assert_eq!(foo.load(Ordering::SeqCst), 42);
            /// ```
            ///
            /// If you want to obtain the maximum value in one step, you can use the following:
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(23);")]
            /// let bar = 42;
            /// let max_foo = foo.fetch_max(bar, Ordering::SeqCst).max(bar);
            /// assert!(max_foo == 42);
            /// ```
            #[inline]
            #[stable(feature = "atomic_min_max", since = "1.45.0")]
            #[$cfg_cas]
            pub fn fetch_max(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { $max_fn(self.v.get(), val, order) }
            }

            /// Minimum with the current value.
            ///
            /// Finds the minimum of the current value and the argument `val`, and
            /// sets the new value to the result.
            ///
            /// Returns the previous value.
            ///
            /// `fetch_min` takes an [`Ordering`] argument which describes the memory ordering
            /// of this operation. All ordering modes are possible. Note that using
            /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
            /// using [`Release`] makes the load part [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(23);")]
            /// assert_eq!(foo.fetch_min(42, Ordering::Relaxed), 23);
            /// assert_eq!(foo.load(Ordering::Relaxed), 23);
            /// assert_eq!(foo.fetch_min(22, Ordering::Relaxed), 23);
            /// assert_eq!(foo.load(Ordering::Relaxed), 22);
            /// ```
            ///
            /// If you want to obtain the minimum value in one step, you can use the following:
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let foo = ", stringify!($atomic_type), "::new(23);")]
            /// let bar = 12;
            /// let min_foo = foo.fetch_min(bar, Ordering::SeqCst).min(bar);
            /// assert_eq!(min_foo, 12);
            /// ```
            #[inline]
            #[stable(feature = "atomic_min_max", since = "1.45.0")]
            #[$cfg_cas]
            pub fn fetch_min(&self, val: $int_type, order: Ordering) -> $int_type {
                // SAFETY: data races are prevented by atomic intrinsics.
                unsafe { $min_fn(self.v.get(), val, order) }
            }

            /// Returns a mutable pointer to the underlying integer.
            ///
            /// Doing non-atomic reads and writes on the resulting integer can be a data race.
            /// This method is mostly useful for FFI, where the function signature may use
            #[doc = concat!("`*mut ", stringify!($int_type), "` instead of `&", stringify!($atomic_type), "`.")]
            ///
            /// Returning an `*mut` pointer from a shared reference to this atomic is safe because the
            /// atomic types work with interior mutability. All modifications of an atomic change the value
            /// through a shared reference, and can do so safely as long as they use atomic operations. Any
            /// use of the returned raw pointer requires an `unsafe` block and still has to uphold the same
            /// restriction: operations on it must be atomic.
            ///
            /// # Examples
            ///
            /// ```ignore (extern-declaration)
            /// # fn main() {
            #[doc = concat!($extra_feature, "use std::sync::atomic::", stringify!($atomic_type), ";")]
            ///
            /// extern "C" {
            #[doc = concat!("    fn my_atomic_op(arg: *mut ", stringify!($int_type), ");")]
            /// }
            ///
            #[doc = concat!("let mut atomic = ", stringify!($atomic_type), "::new(1);")]
            ///
            // SAFETY: Safe as long as `my_atomic_op` is atomic.
            /// unsafe {
            ///     my_atomic_op(atomic.as_mut_ptr());
            /// }
            /// # }
            /// ```
            #[inline]
            #[unstable(feature = "atomic_mut_ptr",
                   reason = "recently added",
                   issue = "66893")]
            pub fn as_mut_ptr(&self) -> *mut $int_type {
                self.v.get()
            }
        }
    }
}

#[cfg(target_has_atomic_load_store = "8")]
atomic_int! {
    cfg(target_has_atomic = "8"),
    cfg(target_has_atomic_equal_alignment = "8"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i8",
    "",
    atomic_min, atomic_max,
    1,
    "AtomicI8::new(0)",
    i8 AtomicI8 ATOMIC_I8_INIT
}
#[cfg(target_has_atomic_load_store = "8")]
atomic_int! {
    cfg(target_has_atomic = "8"),
    cfg(target_has_atomic_equal_alignment = "8"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u8",
    "",
    atomic_umin, atomic_umax,
    1,
    "AtomicU8::new(0)",
    u8 AtomicU8 ATOMIC_U8_INIT
}
#[cfg(target_has_atomic_load_store = "16")]
atomic_int! {
    cfg(target_has_atomic = "16"),
    cfg(target_has_atomic_equal_alignment = "16"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i16",
    "",
    atomic_min, atomic_max,
    2,
    "AtomicI16::new(0)",
    i16 AtomicI16 ATOMIC_I16_INIT
}
#[cfg(target_has_atomic_load_store = "16")]
atomic_int! {
    cfg(target_has_atomic = "16"),
    cfg(target_has_atomic_equal_alignment = "16"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u16",
    "",
    atomic_umin, atomic_umax,
    2,
    "AtomicU16::new(0)",
    u16 AtomicU16 ATOMIC_U16_INIT
}
#[cfg(target_has_atomic_load_store = "32")]
atomic_int! {
    cfg(target_has_atomic = "32"),
    cfg(target_has_atomic_equal_alignment = "32"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i32",
    "",
    atomic_min, atomic_max,
    4,
    "AtomicI32::new(0)",
    i32 AtomicI32 ATOMIC_I32_INIT
}
#[cfg(target_has_atomic_load_store = "32")]
atomic_int! {
    cfg(target_has_atomic = "32"),
    cfg(target_has_atomic_equal_alignment = "32"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u32",
    "",
    atomic_umin, atomic_umax,
    4,
    "AtomicU32::new(0)",
    u32 AtomicU32 ATOMIC_U32_INIT
}
#[cfg(target_has_atomic_load_store = "64")]
atomic_int! {
    cfg(target_has_atomic = "64"),
    cfg(target_has_atomic_equal_alignment = "64"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i64",
    "",
    atomic_min, atomic_max,
    8,
    "AtomicI64::new(0)",
    i64 AtomicI64 ATOMIC_I64_INIT
}
#[cfg(target_has_atomic_load_store = "64")]
atomic_int! {
    cfg(target_has_atomic = "64"),
    cfg(target_has_atomic_equal_alignment = "64"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u64",
    "",
    atomic_umin, atomic_umax,
    8,
    "AtomicU64::new(0)",
    u64 AtomicU64 ATOMIC_U64_INIT
}
#[cfg(target_has_atomic_load_store = "128")]
atomic_int! {
    cfg(target_has_atomic = "128"),
    cfg(target_has_atomic_equal_alignment = "128"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i128",
    "#![feature(integer_atomics)]\n\n",
    atomic_min, atomic_max,
    16,
    "AtomicI128::new(0)",
    i128 AtomicI128 ATOMIC_I128_INIT
}
#[cfg(target_has_atomic_load_store = "128")]
atomic_int! {
    cfg(target_has_atomic = "128"),
    cfg(target_has_atomic_equal_alignment = "128"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    rustc_const_stable(feature = "const_integer_atomics", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u128",
    "#![feature(integer_atomics)]\n\n",
    atomic_umin, atomic_umax,
    16,
    "AtomicU128::new(0)",
    u128 AtomicU128 ATOMIC_U128_INIT
}

macro_rules! atomic_int_ptr_sized {
    ( $($target_pointer_width:literal $align:literal)* ) => { $(
        #[cfg(target_has_atomic_load_store = "ptr")]
        #[cfg(target_pointer_width = $target_pointer_width)]
        atomic_int! {
            cfg(target_has_atomic = "ptr"),
            cfg(target_has_atomic_equal_alignment = "ptr"),
            stable(feature = "rust1", since = "1.0.0"),
            stable(feature = "extended_compare_and_swap", since = "1.10.0"),
            stable(feature = "atomic_debug", since = "1.3.0"),
            stable(feature = "atomic_access", since = "1.15.0"),
            stable(feature = "atomic_from", since = "1.23.0"),
            stable(feature = "atomic_nand", since = "1.27.0"),
            rustc_const_stable(feature = "const_integer_atomics", since = "1.24.0"),
            stable(feature = "rust1", since = "1.0.0"),
            "isize",
            "",
            atomic_min, atomic_max,
            $align,
            "AtomicIsize::new(0)",
            isize AtomicIsize ATOMIC_ISIZE_INIT
        }
        #[cfg(target_has_atomic_load_store = "ptr")]
        #[cfg(target_pointer_width = $target_pointer_width)]
        atomic_int! {
            cfg(target_has_atomic = "ptr"),
            cfg(target_has_atomic_equal_alignment = "ptr"),
            stable(feature = "rust1", since = "1.0.0"),
            stable(feature = "extended_compare_and_swap", since = "1.10.0"),
            stable(feature = "atomic_debug", since = "1.3.0"),
            stable(feature = "atomic_access", since = "1.15.0"),
            stable(feature = "atomic_from", since = "1.23.0"),
            stable(feature = "atomic_nand", since = "1.27.0"),
            rustc_const_stable(feature = "const_integer_atomics", since = "1.24.0"),
            stable(feature = "rust1", since = "1.0.0"),
            "usize",
            "",
            atomic_umin, atomic_umax,
            $align,
            "AtomicUsize::new(0)",
            usize AtomicUsize ATOMIC_USIZE_INIT
        }
    )* };
}

atomic_int_ptr_sized! {
    "16" 2
    "32" 4
    "64" 8
}

#[inline]
#[cfg(target_has_atomic = "8")]
fn strongest_failure_ordering(order: Ordering) -> Ordering {
    match order {
        Release => Relaxed,
        Relaxed => Relaxed,
        SeqCst => SeqCst,
        Acquire => Acquire,
        AcqRel => Acquire,
    }
}

#[inline]
unsafe fn atomic_store<T: Copy>(dst: *mut T, val: T, order: Ordering) {
    // SAFETY: the caller must uphold the safety contract for `atomic_store`.
    unsafe {
        match order {
            Release => intrinsics::atomic_store_rel(dst, val),
            Relaxed => intrinsics::atomic_store_relaxed(dst, val),
            SeqCst => intrinsics::atomic_store(dst, val),
            Acquire => panic!("there is no such thing as an acquire store"),
            AcqRel => panic!("there is no such thing as an acquire/release store"),
        }
    }
}

#[inline]
unsafe fn atomic_load<T: Copy>(dst: *const T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_load`.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_load_acq(dst),
            Relaxed => intrinsics::atomic_load_relaxed(dst),
            SeqCst => intrinsics::atomic_load(dst),
            Release => panic!("there is no such thing as a release load"),
            AcqRel => panic!("there is no such thing as an acquire/release load"),
        }
    }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_swap<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_swap`.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_xchg_acq(dst, val),
            Release => intrinsics::atomic_xchg_rel(dst, val),
            AcqRel => intrinsics::atomic_xchg_acqrel(dst, val),
            Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
            SeqCst => intrinsics::atomic_xchg(dst, val),
        }
    }
}

/// Returns the previous value (like __sync_fetch_and_add).
#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_add<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_add`.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_xadd_acq(dst, val),
            Release => intrinsics::atomic_xadd_rel(dst, val),
            AcqRel => intrinsics::atomic_xadd_acqrel(dst, val),
            Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
            SeqCst => intrinsics::atomic_xadd(dst, val),
        }
    }
}

/// Returns the previous value (like __sync_fetch_and_sub).
#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_sub<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_sub`.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_xsub_acq(dst, val),
            Release => intrinsics::atomic_xsub_rel(dst, val),
            AcqRel => intrinsics::atomic_xsub_acqrel(dst, val),
            Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
            SeqCst => intrinsics::atomic_xsub(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_compare_exchange<T: Copy>(
    dst: *mut T,
    old: T,
    new: T,
    success: Ordering,
    failure: Ordering,
) -> Result<T, T> {
    // SAFETY: the caller must uphold the safety contract for `atomic_compare_exchange`.
    let (val, ok) = unsafe {
        match (success, failure) {
            (Acquire, Acquire) => intrinsics::atomic_cxchg_acq(dst, old, new),
            (Release, Relaxed) => intrinsics::atomic_cxchg_rel(dst, old, new),
            (AcqRel, Acquire) => intrinsics::atomic_cxchg_acqrel(dst, old, new),
            (Relaxed, Relaxed) => intrinsics::atomic_cxchg_relaxed(dst, old, new),
            (SeqCst, SeqCst) => intrinsics::atomic_cxchg(dst, old, new),
            (Acquire, Relaxed) => intrinsics::atomic_cxchg_acq_failrelaxed(dst, old, new),
            (AcqRel, Relaxed) => intrinsics::atomic_cxchg_acqrel_failrelaxed(dst, old, new),
            (SeqCst, Relaxed) => intrinsics::atomic_cxchg_failrelaxed(dst, old, new),
            (SeqCst, Acquire) => intrinsics::atomic_cxchg_failacq(dst, old, new),
            (_, AcqRel) => panic!("there is no such thing as an acquire/release failure ordering"),
            (_, Release) => panic!("there is no such thing as a release failure ordering"),
            _ => panic!("a failure ordering can't be stronger than a success ordering"),
        }
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_compare_exchange_weak<T: Copy>(
    dst: *mut T,
    old: T,
    new: T,
    success: Ordering,
    failure: Ordering,
) -> Result<T, T> {
    // SAFETY: the caller must uphold the safety contract for `atomic_compare_exchange_weak`.
    let (val, ok) = unsafe {
        match (success, failure) {
            (Acquire, Acquire) => intrinsics::atomic_cxchgweak_acq(dst, old, new),
            (Release, Relaxed) => intrinsics::atomic_cxchgweak_rel(dst, old, new),
            (AcqRel, Acquire) => intrinsics::atomic_cxchgweak_acqrel(dst, old, new),
            (Relaxed, Relaxed) => intrinsics::atomic_cxchgweak_relaxed(dst, old, new),
            (SeqCst, SeqCst) => intrinsics::atomic_cxchgweak(dst, old, new),
            (Acquire, Relaxed) => intrinsics::atomic_cxchgweak_acq_failrelaxed(dst, old, new),
            (AcqRel, Relaxed) => intrinsics::atomic_cxchgweak_acqrel_failrelaxed(dst, old, new),
            (SeqCst, Relaxed) => intrinsics::atomic_cxchgweak_failrelaxed(dst, old, new),
            (SeqCst, Acquire) => intrinsics::atomic_cxchgweak_failacq(dst, old, new),
            (_, AcqRel) => panic!("there is no such thing as an acquire/release failure ordering"),
            (_, Release) => panic!("there is no such thing as a release failure ordering"),
            _ => panic!("a failure ordering can't be stronger than a success ordering"),
        }
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_and<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_and`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_and_acq(dst, val),
            Release => intrinsics::atomic_and_rel(dst, val),
            AcqRel => intrinsics::atomic_and_acqrel(dst, val),
            Relaxed => intrinsics::atomic_and_relaxed(dst, val),
            SeqCst => intrinsics::atomic_and(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_nand<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_nand`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_nand_acq(dst, val),
            Release => intrinsics::atomic_nand_rel(dst, val),
            AcqRel => intrinsics::atomic_nand_acqrel(dst, val),
            Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
            SeqCst => intrinsics::atomic_nand(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_or<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_or`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_or_acq(dst, val),
            Release => intrinsics::atomic_or_rel(dst, val),
            AcqRel => intrinsics::atomic_or_acqrel(dst, val),
            Relaxed => intrinsics::atomic_or_relaxed(dst, val),
            SeqCst => intrinsics::atomic_or(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_xor<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_xor`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_xor_acq(dst, val),
            Release => intrinsics::atomic_xor_rel(dst, val),
            AcqRel => intrinsics::atomic_xor_acqrel(dst, val),
            Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
            SeqCst => intrinsics::atomic_xor(dst, val),
        }
    }
}

/// returns the max value (signed comparison)
#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_max<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_max`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_max_acq(dst, val),
            Release => intrinsics::atomic_max_rel(dst, val),
            AcqRel => intrinsics::atomic_max_acqrel(dst, val),
            Relaxed => intrinsics::atomic_max_relaxed(dst, val),
            SeqCst => intrinsics::atomic_max(dst, val),
        }
    }
}

/// returns the min value (signed comparison)
#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_min<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_min`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_min_acq(dst, val),
            Release => intrinsics::atomic_min_rel(dst, val),
            AcqRel => intrinsics::atomic_min_acqrel(dst, val),
            Relaxed => intrinsics::atomic_min_relaxed(dst, val),
            SeqCst => intrinsics::atomic_min(dst, val),
        }
    }
}

/// returns the max value (unsigned comparison)
#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_umax<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_umax`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_umax_acq(dst, val),
            Release => intrinsics::atomic_umax_rel(dst, val),
            AcqRel => intrinsics::atomic_umax_acqrel(dst, val),
            Relaxed => intrinsics::atomic_umax_relaxed(dst, val),
            SeqCst => intrinsics::atomic_umax(dst, val),
        }
    }
}

/// returns the min value (unsigned comparison)
#[inline]
#[cfg(target_has_atomic = "8")]
unsafe fn atomic_umin<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_umin`
    unsafe {
        match order {
            Acquire => intrinsics::atomic_umin_acq(dst, val),
            Release => intrinsics::atomic_umin_rel(dst, val),
            AcqRel => intrinsics::atomic_umin_acqrel(dst, val),
            Relaxed => intrinsics::atomic_umin_relaxed(dst, val),
            SeqCst => intrinsics::atomic_umin(dst, val),
        }
    }
}

/// An atomic fence.
///
/// Depending on the specified order, a fence prevents the compiler and CPU from
/// reordering certain types of memory operations around it.
/// That creates synchronizes-with relationships between it and atomic operations
/// or fences in other threads.
///
/// A fence 'A' which has (at least) [`Release`] ordering semantics, synchronizes
/// with a fence 'B' with (at least) [`Acquire`] semantics, if and only if there
/// exist operations X and Y, both operating on some atomic object 'M' such
/// that A is sequenced before X, Y is synchronized before B and Y observes
/// the change to M. This provides a happens-before dependence between A and B.
///
/// ```text
///     Thread 1                                          Thread 2
///
/// fence(Release);      A --------------
/// x.store(3, Relaxed); X ---------    |
///                                |    |
///                                |    |
///                                -------------> Y  if x.load(Relaxed) == 3 {
///                                     |-------> B      fence(Acquire);
///                                                      ...
///                                                  }
/// ```
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
/// # Examples
///
/// ```
/// use std::sync::atomic::AtomicBool;
/// use std::sync::atomic::fence;
/// use std::sync::atomic::Ordering;
///
/// // A mutual exclusion primitive based on spinlock.
/// pub struct Mutex {
///     flag: AtomicBool,
/// }
///
/// impl Mutex {
///     pub fn new() -> Mutex {
///         Mutex {
///             flag: AtomicBool::new(false),
///         }
///     }
///
///     pub fn lock(&self) {
///         // Wait until the old value is `false`.
///         while self
///             .flag
///             .compare_exchange_weak(false, true, Ordering::Relaxed, Ordering::Relaxed)
///             .is_err()
///         {}
///         // This fence synchronizes-with store in `unlock`.
///         fence(Ordering::Acquire);
///     }
///
///     pub fn unlock(&self) {
///         self.flag.store(false, Ordering::Release);
///     }
/// }
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn fence(order: Ordering) {
    // SAFETY: using an atomic fence is safe.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_fence_acq(),
            Release => intrinsics::atomic_fence_rel(),
            AcqRel => intrinsics::atomic_fence_acqrel(),
            SeqCst => intrinsics::atomic_fence(),
            Relaxed => panic!("there is no such thing as a relaxed fence"),
        }
    }
}

/// A compiler memory fence.
///
/// `compiler_fence` does not emit any machine code, but restricts the kinds
/// of memory re-ordering the compiler is allowed to do. Specifically, depending on
/// the given [`Ordering`] semantics, the compiler may be disallowed from moving reads
/// or writes from before or after the call to the other side of the call to
/// `compiler_fence`. Note that it does **not** prevent the *hardware*
/// from doing such re-ordering. This is not a problem in a single-threaded,
/// execution context, but when other threads may modify memory at the same
/// time, stronger synchronization primitives such as [`fence`] are required.
///
/// The re-ordering prevented by the different ordering semantics are:
///
///  - with [`SeqCst`], no re-ordering of reads and writes across this point is allowed.
///  - with [`Release`], preceding reads and writes cannot be moved past subsequent writes.
///  - with [`Acquire`], subsequent reads and writes cannot be moved ahead of preceding reads.
///  - with [`AcqRel`], both of the above rules are enforced.
///
/// `compiler_fence` is generally only useful for preventing a thread from
/// racing *with itself*. That is, if a given thread is executing one piece
/// of code, and is then interrupted, and starts executing code elsewhere
/// (while still in the same thread, and conceptually still on the same
/// core). In traditional programs, this can only occur when a signal
/// handler is registered. In more low-level code, such situations can also
/// arise when handling interrupts, when implementing green threads with
/// pre-emption, etc. Curious readers are encouraged to read the Linux kernel's
/// discussion of [memory barriers].
///
/// # Panics
///
/// Panics if `order` is [`Relaxed`].
///
/// # Examples
///
/// Without `compiler_fence`, the `assert_eq!` in following code
/// is *not* guaranteed to succeed, despite everything happening in a single thread.
/// To see why, remember that the compiler is free to swap the stores to
/// `IMPORTANT_VARIABLE` and `IS_READY` since they are both
/// `Ordering::Relaxed`. If it does, and the signal handler is invoked right
/// after `IS_READY` is updated, then the signal handler will see
/// `IS_READY=1`, but `IMPORTANT_VARIABLE=0`.
/// Using a `compiler_fence` remedies this situation.
///
/// ```
/// use std::sync::atomic::{AtomicBool, AtomicUsize};
/// use std::sync::atomic::Ordering;
/// use std::sync::atomic::compiler_fence;
///
/// static IMPORTANT_VARIABLE: AtomicUsize = AtomicUsize::new(0);
/// static IS_READY: AtomicBool = AtomicBool::new(false);
///
/// fn main() {
///     IMPORTANT_VARIABLE.store(42, Ordering::Relaxed);
///     // prevent earlier writes from being moved beyond this point
///     compiler_fence(Ordering::Release);
///     IS_READY.store(true, Ordering::Relaxed);
/// }
///
/// fn signal_handler() {
///     if IS_READY.load(Ordering::Relaxed) {
///         assert_eq!(IMPORTANT_VARIABLE.load(Ordering::Relaxed), 42);
///     }
/// }
/// ```
///
/// [memory barriers]: https://www.kernel.org/doc/Documentation/memory-barriers.txt
#[inline]
#[stable(feature = "compiler_fences", since = "1.21.0")]
pub fn compiler_fence(order: Ordering) {
    // SAFETY: using an atomic fence is safe.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_singlethreadfence_acq(),
            Release => intrinsics::atomic_singlethreadfence_rel(),
            AcqRel => intrinsics::atomic_singlethreadfence_acqrel(),
            SeqCst => intrinsics::atomic_singlethreadfence(),
            Relaxed => panic!("there is no such thing as a relaxed compiler fence"),
        }
    }
}

#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl fmt::Debug for AtomicBool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::SeqCst), f)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl<T> fmt::Debug for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::SeqCst), f)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "atomic_pointer", since = "1.24.0")]
impl<T> fmt::Pointer for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.load(Ordering::SeqCst), f)
    }
}

/// Signals the processor that it is inside a busy-wait spin-loop ("spin lock").
///
/// This function is deprecated in favor of [`hint::spin_loop`].
///
/// [`hint::spin_loop`]: crate::hint::spin_loop
#[inline]
#[stable(feature = "spin_loop_hint", since = "1.24.0")]
#[rustc_deprecated(since = "1.51.0", reason = "use hint::spin_loop instead")]
pub fn spin_loop_hint() {
    spin_loop()
}
