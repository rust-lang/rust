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
//! [`AtomicBool`]: struct.AtomicBool.html
//! [`AtomicIsize`]: struct.AtomicIsize.html
//! [`AtomicUsize`]: struct.AtomicUsize.html
//! [`AtomicI8`]: struct.AtomicI8.html
//! [`AtomicU16`]: struct.AtomicU16.html
//!
//! Each method takes an [`Ordering`] which represents the strength of
//! the memory barrier for that operation. These orderings are the
//! same as [LLVM atomic orderings][1]. For more information see the [nomicon][2].
//!
//! [`Ordering`]: enum.Ordering.html
//!
//! [1]: https://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations
//! [2]: ../../../nomicon/atomics.html
//!
//! Atomic variables are safe to share between threads (they implement [`Sync`])
//! but they do not themselves provide the mechanism for sharing and follow the
//! [threading model](../../../std/thread/index.html#the-threading-model) of rust.
//! The most common way to share an atomic variable is to put it into an [`Arc`][arc] (an
//! atomically-reference-counted shared pointer).
//!
//! [`Sync`]: ../../marker/trait.Sync.html
//! [arc]: ../../../std/sync/struct.Arc.html
//!
//! Atomic types may be stored in static variables, initialized using
//! the constant initializers like [`AtomicBool::new`]. Atomic statics
//! are often used for lazy global initialization.
//!
//! [`AtomicBool::new`]: struct.AtomicBool.html#method.new
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
//! The atomic types in this module may not be available on all platforms. The
//! atomic types here are all widely available, however, and can generally be
//! relied upon existing. Some notable exceptions are:
//!
//! * PowerPC and MIPS platforms with 32-bit pointers do not have `AtomicU64` or
//!   `AtomicI64` types.
//! * ARM platforms like `armv5te` that aren't for Linux do not have any atomics
//!   at all.
//! * ARM targets with `thumbv6m` do not have atomic operations at all.
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
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);
//!
//! let old_thread_count = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
//! println!("live threads: {}", old_thread_count + 1);
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![cfg_attr(not(target_has_atomic = "8"), allow(dead_code))]
#![cfg_attr(not(target_has_atomic = "8"), allow(unused_imports))]

use self::Ordering::*;

use crate::intrinsics;
use crate::cell::UnsafeCell;
use crate::fmt;

use crate::hint::spin_loop;

/// Signals the processor that it is entering a busy-wait spin-loop.
///
/// Upon receiving spin-loop signal the processor can optimize its behavior by, for example, saving
/// power or switching hyper-threads.
///
/// This function is different than [`std::thread::yield_now`] which directly yields to the
/// system's scheduler, whereas `spin_loop_hint` only signals the processor that it is entering a
/// busy-wait spin-loop without yielding control to the system's scheduler.
///
/// Using a busy-wait spin-loop with `spin_loop_hint` is ideally used in situations where a
/// contended lock is held by another thread executed on a different CPU and where the waiting
/// times are relatively small. Because entering busy-wait spin-loop does not trigger the system's
/// scheduler, no overhead for switching threads occurs. However, if the thread holding the
/// contended lock is running on the same CPU, the spin-loop is likely to occupy an entire CPU slice
/// before switching to the thread that holds the lock. If the contending lock is held by a thread
/// on the same CPU or if the waiting times for acquiring the lock are longer, it is often better to
/// use [`std::thread::yield_now`].
///
/// **Note**: On platforms that do not support receiving spin-loop hints this function does not
/// do anything at all.
///
/// [`std::thread::yield_now`]: ../../../std/thread/fn.yield_now.html
#[inline]
#[stable(feature = "spin_loop_hint", since = "1.24.0")]
pub fn spin_loop_hint() {
    spin_loop()
}

/// A boolean type which can be safely shared between threads.
///
/// This type has the same in-memory representation as a [`bool`].
///
/// [`bool`]: ../../../std/primitive.bool.html
#[cfg(target_has_atomic = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
#[repr(C, align(1))]
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
#[cfg_attr(target_pointer_width = "16", repr(C, align(2)))]
#[cfg_attr(target_pointer_width = "32", repr(C, align(4)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(8)))]
pub struct AtomicPtr<T> {
    p: UnsafeCell<*mut T>,
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for AtomicPtr<T> {
    /// Creates a null `AtomicPtr<T>`.
    fn default() -> AtomicPtr<T> {
        AtomicPtr::new(crate::ptr::null_mut())
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
/// Memory orderings specify the way atomic operations synchronize memory.
/// In its weakest [`Relaxed`][Ordering::Relaxed], only the memory directly touched by the
/// operation is synchronized. On the other hand, a store-load pair of [`SeqCst`][Ordering::SeqCst]
/// operations synchronize other memory while additionally preserving a total order of such
/// operations across all threads.
///
/// Rust's memory orderings are [the same as
/// LLVM's](https://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations).
///
/// For more information see the [nomicon].
///
/// [nomicon]: ../../../nomicon/atomics.html
/// [Ordering::Relaxed]: #variant.Relaxed
/// [Ordering::SeqCst]: #variant.SeqCst
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Ordering {
    /// No ordering constraints, only atomic operations.
    ///
    /// Corresponds to LLVM's [`Monotonic`] ordering.
    ///
    /// [`Monotonic`]: https://llvm.org/docs/Atomics.html#monotonic
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
    /// Corresponds to LLVM's [`Release`] ordering.
    ///
    /// [`Release`]: https://llvm.org/docs/Atomics.html#release
    /// [`Acquire`]: https://llvm.org/docs/Atomics.html#acquire
    /// [`Relaxed`]: https://llvm.org/docs/Atomics.html#monotonic
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
    /// Corresponds to LLVM's [`Acquire`] ordering.
    ///
    /// [`Acquire`]: https://llvm.org/docs/Atomics.html#acquire
    /// [`Release`]: https://llvm.org/docs/Atomics.html#release
    /// [`Relaxed`]: https://llvm.org/docs/Atomics.html#monotonic
    #[stable(feature = "rust1", since = "1.0.0")]
    Acquire,
    /// Has the effects of both [`Acquire`] and [`Release`] together:
    /// For loads it uses [`Acquire`] ordering. For stores it uses the [`Release`] ordering.
    ///
    /// Notice that in the case of `compare_and_swap`, it is possible that the operation ends up
    /// not performing any store and hence it has just [`Acquire`] ordering. However,
    /// [`AcqRel`][`AcquireRelease`] will never perform [`Relaxed`] accesses.
    ///
    /// This ordering is only applicable for operations that combine both loads and stores.
    ///
    /// Corresponds to LLVM's [`AcquireRelease`] ordering.
    ///
    /// [`AcquireRelease`]: https://llvm.org/docs/Atomics.html#acquirerelease
    /// [`Acquire`]: https://llvm.org/docs/Atomics.html#acquire
    /// [`Release`]: https://llvm.org/docs/Atomics.html#release
    /// [`Relaxed`]: https://llvm.org/docs/Atomics.html#monotonic
    #[stable(feature = "rust1", since = "1.0.0")]
    AcqRel,
    /// Like [`Acquire`]/[`Release`]/[`AcqRel`] (for load, store, and load-with-store
    /// operations, respectively) with the additional guarantee that all threads see all
    /// sequentially consistent operations in the same order.
    ///
    /// Corresponds to LLVM's [`SequentiallyConsistent`] ordering.
    ///
    /// [`SequentiallyConsistent`]: https://llvm.org/docs/Atomics.html#sequentiallyconsistent
    /// [`Acquire`]: https://llvm.org/docs/Atomics.html#acquire
    /// [`Release`]: https://llvm.org/docs/Atomics.html#release
    /// [`AcqRel`]: https://llvm.org/docs/Atomics.html#acquirerelease
    #[stable(feature = "rust1", since = "1.0.0")]
    SeqCst,
}

/// An [`AtomicBool`] initialized to `false`.
///
/// [`AtomicBool`]: struct.AtomicBool.html
#[cfg(target_has_atomic = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(
    since = "1.34.0",
    reason = "the `new` function is now preferred",
    suggestion = "AtomicBool::new(false)",
)]
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

    /// Returns a mutable reference to the underlying [`bool`].
    ///
    /// This is safe because the mutable reference guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// [`bool`]: ../../../std/primitive.bool.html
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
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    /// of this operation. Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Acquire`] or [`AcqRel`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    #[cfg(target_has_atomic = "cas")]
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
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
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    /// [`bool`]: ../../../std/primitive.bool.html
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
    #[cfg(target_has_atomic = "cas")]
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
    /// ordering of this operation. The first describes the required ordering if the
    /// operation succeeds while the second describes the required ordering when the
    /// operation fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    ///
    /// [`bool`]: ../../../std/primitive.bool.html
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    #[cfg(target_has_atomic = "cas")]
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

    /// Stores a value into the [`bool`] if the current value is the same as the `current` value.
    ///
    /// Unlike [`compare_exchange`], this function is allowed to spuriously fail even when the
    /// comparison succeeds, which can result in more efficient code on some platforms. The
    /// return value is a result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if the
    /// operation succeeds while the second describes the required ordering when the
    /// operation fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// [`bool`]: ../../../std/primitive.bool.html
    /// [`compare_exchange`]: #method.compare_exchange
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    #[cfg(target_has_atomic = "cas")]
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
    /// `fetch_and` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    #[cfg(target_has_atomic = "cas")]
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
    /// `fetch_nand` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    #[cfg(target_has_atomic = "cas")]
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
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    #[cfg(target_has_atomic = "cas")]
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
    /// `fetch_xor` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    #[cfg(target_has_atomic = "cas")]
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
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    /// of this operation. Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Acquire`] or [`AcqRel`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
        unsafe {
            atomic_store(self.p.get() as *mut usize, ptr as usize, order);
        }
    }

    /// Stores a value into the pointer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    #[cfg(target_has_atomic = "cas")]
    pub fn swap(&self, ptr: *mut T, order: Ordering) -> *mut T {
        unsafe { atomic_swap(self.p.get() as *mut usize, ptr as usize, order) as *mut T }
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
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
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
    /// let value = some_ptr.compare_and_swap(other_ptr, another_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg(target_has_atomic = "cas")]
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
    /// ordering of this operation. The first describes the required ordering if the
    /// operation succeeds while the second describes the required ordering when the
    /// operation fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    #[cfg(target_has_atomic = "cas")]
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
    /// ordering of this operation. The first describes the required ordering if the
    /// operation succeeds while the second describes the required ordering when the
    /// operation fails. Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the successful load
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// [`compare_exchange`]: #method.compare_exchange
    /// [`Ordering`]: enum.Ordering.html
    /// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
    /// [`Release`]: enum.Ordering.html#variant.Release
    /// [`Acquire`]: enum.Ordering.html#variant.Acquire
    /// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
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
    #[cfg(target_has_atomic = "cas")]
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

#[cfg(target_has_atomic = "8")]
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
    fn from(b: bool) -> Self { Self::new(b) }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "atomic_from", since = "1.23.0")]
impl<T> From<*mut T> for AtomicPtr<T> {
    #[inline]
    fn from(p: *mut T) -> Self { Self::new(p) }
}

#[cfg(target_has_atomic = "ptr")]
macro_rules! atomic_int {
    ($stable:meta,
     $stable_cxchg:meta,
     $stable_debug:meta,
     $stable_access:meta,
     $stable_from:meta,
     $stable_nand:meta,
     $stable_init_const:meta,
     $s_int_type:expr, $int_ref:expr,
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
        /// `](
        #[doc = $int_ref]
        /// ). For more about the differences between atomic types and
        /// non-atomic types as well as information about the portability of
        /// this type, please see the [module-level documentation].
        ///
        /// [module-level documentation]: index.html
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
            fn default() -> Self {
                Self::new(Default::default())
            }
        }

        #[$stable_from]
        impl From<$int_type> for $atomic_type {
            doc_comment! {
                concat!(
"Converts an `", stringify!($int_type), "` into an `", stringify!($atomic_type), "`."),
                #[inline]
                fn from(v: $int_type) -> Self { Self::new(v) }
            }
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
            doc_comment! {
                concat!("Creates a new atomic integer.

# Examples

```
", $extra_feature, "use std::sync::atomic::", stringify!($atomic_type), ";

let atomic_forty_two = ", stringify!($atomic_type), "::new(42);
```"),
                #[inline]
                #[$stable]
                pub const fn new(v: $int_type) -> Self {
                    $atomic_type {v: UnsafeCell::new(v)}
                }
            }

            doc_comment! {
                concat!("Returns a mutable reference to the underlying integer.

This is safe because the mutable reference guarantees that no other threads are
concurrently accessing the atomic data.

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let mut some_var = ", stringify!($atomic_type), "::new(10);
assert_eq!(*some_var.get_mut(), 10);
*some_var.get_mut() = 5;
assert_eq!(some_var.load(Ordering::SeqCst), 5);
```"),
                #[inline]
                #[$stable_access]
                pub fn get_mut(&mut self) -> &mut $int_type {
                    unsafe { &mut *self.v.get() }
                }
            }

            doc_comment! {
                concat!("Consumes the atomic and returns the contained value.

This is safe because passing `self` by value guarantees that no other threads are
concurrently accessing the atomic data.

# Examples

```
", $extra_feature, "use std::sync::atomic::", stringify!($atomic_type), ";

let some_var = ", stringify!($atomic_type), "::new(5);
assert_eq!(some_var.into_inner(), 5);
```"),
                #[inline]
                #[$stable_access]
                pub fn into_inner(self) -> $int_type {
                    self.v.into_inner()
                }
            }

            doc_comment! {
                concat!("Loads a value from the atomic integer.

`load` takes an [`Ordering`] argument which describes the memory ordering of this operation.
Possible values are [`SeqCst`], [`Acquire`] and [`Relaxed`].

# Panics

Panics if `order` is [`Release`] or [`AcqRel`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire
[`AcqRel`]: enum.Ordering.html#variant.AcqRel
[`SeqCst`]: enum.Ordering.html#variant.SeqCst

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let some_var = ", stringify!($atomic_type), "::new(5);

assert_eq!(some_var.load(Ordering::Relaxed), 5);
```"),
                #[inline]
                #[$stable]
                pub fn load(&self, order: Ordering) -> $int_type {
                    unsafe { atomic_load(self.v.get(), order) }
                }
            }

            doc_comment! {
                concat!("Stores a value into the atomic integer.

`store` takes an [`Ordering`] argument which describes the memory ordering of this operation.
 Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].

# Panics

Panics if `order` is [`Acquire`] or [`AcqRel`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire
[`AcqRel`]: enum.Ordering.html#variant.AcqRel
[`SeqCst`]: enum.Ordering.html#variant.SeqCst

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let some_var = ", stringify!($atomic_type), "::new(5);

some_var.store(10, Ordering::Relaxed);
assert_eq!(some_var.load(Ordering::Relaxed), 10);
```"),
                #[inline]
                #[$stable]
                pub fn store(&self, val: $int_type, order: Ordering) {
                    unsafe { atomic_store(self.v.get(), val, order); }
                }
            }

            doc_comment! {
                concat!("Stores a value into the atomic integer, returning the previous value.

`swap` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let some_var = ", stringify!($atomic_type), "::new(5);

assert_eq!(some_var.swap(10, Ordering::Relaxed), 5);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
                pub fn swap(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_swap(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Stores a value into the atomic integer if the current value is the same as
the `current` value.

The return value is always the previous value. If it is equal to `current`, then the
value was updated.

`compare_and_swap` also takes an [`Ordering`] argument which describes the memory
ordering of this operation. Notice that even when using [`AcqRel`], the operation
might fail and hence just perform an `Acquire` load, but not have `Release` semantics.
Using [`Acquire`] makes the store part of this operation [`Relaxed`] if it
happens, and using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire
[`AcqRel`]: enum.Ordering.html#variant.AcqRel

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let some_var = ", stringify!($atomic_type), "::new(5);

assert_eq!(some_var.compare_and_swap(5, 10, Ordering::Relaxed), 5);
assert_eq!(some_var.load(Ordering::Relaxed), 10);

assert_eq!(some_var.compare_and_swap(6, 12, Ordering::Relaxed), 10);
assert_eq!(some_var.load(Ordering::Relaxed), 10);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
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
            }

            doc_comment! {
                concat!("Stores a value into the atomic integer if the current value is the same as
the `current` value.

The return value is a result indicating whether the new value was written and
containing the previous value. On success this value is guaranteed to be equal to
`current`.

`compare_exchange` takes two [`Ordering`] arguments to describe the memory
ordering of this operation. The first describes the required ordering if the
operation succeeds while the second describes the required ordering when the
operation fails. Using [`Acquire`] as success ordering makes the store part
of this operation [`Relaxed`], and using [`Release`] makes the successful load
[`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
and must be equivalent to or weaker than the success ordering.

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire
[`SeqCst`]: enum.Ordering.html#variant.SeqCst

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let some_var = ", stringify!($atomic_type), "::new(5);

assert_eq!(some_var.compare_exchange(5, 10,
                                     Ordering::Acquire,
                                     Ordering::Relaxed),
           Ok(5));
assert_eq!(some_var.load(Ordering::Relaxed), 10);

assert_eq!(some_var.compare_exchange(6, 12,
                                     Ordering::SeqCst,
                                     Ordering::Acquire),
           Err(10));
assert_eq!(some_var.load(Ordering::Relaxed), 10);
```"),
                #[inline]
                #[$stable_cxchg]
                #[cfg(target_has_atomic = "cas")]
                pub fn compare_exchange(&self,
                                        current: $int_type,
                                        new: $int_type,
                                        success: Ordering,
                                        failure: Ordering) -> Result<$int_type, $int_type> {
                    unsafe { atomic_compare_exchange(self.v.get(), current, new, success, failure) }
                }
            }

            doc_comment! {
                concat!("Stores a value into the atomic integer if the current value is the same as
the `current` value.

Unlike [`compare_exchange`], this function is allowed to spuriously fail even
when the comparison succeeds, which can result in more efficient code on some
platforms. The return value is a result indicating whether the new value was
written and containing the previous value.

`compare_exchange_weak` takes two [`Ordering`] arguments to describe the memory
ordering of this operation. The first describes the required ordering if the
operation succeeds while the second describes the required ordering when the
operation fails. Using [`Acquire`] as success ordering makes the store part
of this operation [`Relaxed`], and using [`Release`] makes the successful load
[`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
and must be equivalent to or weaker than the success ordering.

[`compare_exchange`]: #method.compare_exchange
[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire
[`SeqCst`]: enum.Ordering.html#variant.SeqCst

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let val = ", stringify!($atomic_type), "::new(4);

let mut old = val.load(Ordering::Relaxed);
loop {
    let new = old * 2;
    match val.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
        Ok(_) => break,
        Err(x) => old = x,
    }
}
```"),
                #[inline]
                #[$stable_cxchg]
                #[cfg(target_has_atomic = "cas")]
                pub fn compare_exchange_weak(&self,
                                             current: $int_type,
                                             new: $int_type,
                                             success: Ordering,
                                             failure: Ordering) -> Result<$int_type, $int_type> {
                    unsafe {
                        atomic_compare_exchange_weak(self.v.get(), current, new, success, failure)
                    }
                }
            }

            doc_comment! {
                concat!("Adds to the current value, returning the previous value.

This operation wraps around on overflow.

`fetch_add` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(0);
assert_eq!(foo.fetch_add(10, Ordering::SeqCst), 0);
assert_eq!(foo.load(Ordering::SeqCst), 10);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_add(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_add(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Subtracts from the current value, returning the previous value.

This operation wraps around on overflow.

`fetch_sub` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(20);
assert_eq!(foo.fetch_sub(10, Ordering::SeqCst), 20);
assert_eq!(foo.load(Ordering::SeqCst), 10);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_sub(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_sub(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Bitwise \"and\" with the current value.

Performs a bitwise \"and\" operation on the current value and the argument `val`, and
sets the new value to the result.

Returns the previous value.

`fetch_and` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(0b101101);
assert_eq!(foo.fetch_and(0b110011, Ordering::SeqCst), 0b101101);
assert_eq!(foo.load(Ordering::SeqCst), 0b100001);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_and(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_and(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Bitwise \"nand\" with the current value.

Performs a bitwise \"nand\" operation on the current value and the argument `val`, and
sets the new value to the result.

Returns the previous value.

`fetch_nand` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "
use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(0x13);
assert_eq!(foo.fetch_nand(0x31, Ordering::SeqCst), 0x13);
assert_eq!(foo.load(Ordering::SeqCst), !(0x13 & 0x31));
```"),
                #[inline]
                #[$stable_nand]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_nand(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_nand(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Bitwise \"or\" with the current value.

Performs a bitwise \"or\" operation on the current value and the argument `val`, and
sets the new value to the result.

Returns the previous value.

`fetch_or` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(0b101101);
assert_eq!(foo.fetch_or(0b110011, Ordering::SeqCst), 0b101101);
assert_eq!(foo.load(Ordering::SeqCst), 0b111111);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_or(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_or(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Bitwise \"xor\" with the current value.

Performs a bitwise \"xor\" operation on the current value and the argument `val`, and
sets the new value to the result.

Returns the previous value.

`fetch_xor` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(0b101101);
assert_eq!(foo.fetch_xor(0b110011, Ordering::SeqCst), 0b101101);
assert_eq!(foo.load(Ordering::SeqCst), 0b011110);
```"),
                #[inline]
                #[$stable]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_xor(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { atomic_xor(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Fetches the value, and applies a function to it that returns an optional
new value. Returns a `Result` of `Ok(previous_value)` if the function returned `Some(_)`, else
`Err(previous_value)`.

Note: This may call the function multiple times if the value has been changed from other threads in
the meantime, as long as the function returns `Some(_)`, but the function will have been applied
but once to the stored value.

`fetch_update` takes two [`Ordering`] arguments to describe the memory
ordering of this operation. The first describes the required ordering for loads
and failed updates while the second describes the required ordering when the
operation finally succeeds. Beware that this is different from the two
modes in [`compare_exchange`]!

Using [`Acquire`] as success ordering makes the store part
of this operation [`Relaxed`], and using [`Release`] makes the final successful load
[`Relaxed`]. The (failed) load ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
and must be equivalent to or weaker than the success ordering.

[`bool`]: ../../../std/primitive.bool.html
[`compare_exchange`]: #method.compare_exchange
[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire
[`SeqCst`]: enum.Ordering.html#variant.SeqCst

# Examples

```rust
#![feature(no_more_cas)]
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let x = ", stringify!($atomic_type), "::new(7);
assert_eq!(x.fetch_update(|_| None, Ordering::SeqCst, Ordering::SeqCst), Err(7));
assert_eq!(x.fetch_update(|x| Some(x + 1), Ordering::SeqCst, Ordering::SeqCst), Ok(7));
assert_eq!(x.fetch_update(|x| Some(x + 1), Ordering::SeqCst, Ordering::SeqCst), Ok(8));
assert_eq!(x.load(Ordering::SeqCst), 9);
```"),
                #[inline]
                #[unstable(feature = "no_more_cas",
                       reason = "no more CAS loops in user code",
                       issue = "48655")]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_update<F>(&self,
                                       mut f: F,
                                       fetch_order: Ordering,
                                       set_order: Ordering) -> Result<$int_type, $int_type>
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
            }

            doc_comment! {
                concat!("Maximum with the current value.

Finds the maximum of the current value and the argument `val`, and
sets the new value to the result.

Returns the previous value.

`fetch_max` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
#![feature(atomic_min_max)]
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(23);
assert_eq!(foo.fetch_max(42, Ordering::SeqCst), 23);
assert_eq!(foo.load(Ordering::SeqCst), 42);
```

If you want to obtain the maximum value in one step, you can use the following:

```
#![feature(atomic_min_max)]
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(23);
let bar = 42;
let max_foo = foo.fetch_max(bar, Ordering::SeqCst).max(bar);
assert!(max_foo == 42);
```"),
                #[inline]
                #[unstable(feature = "atomic_min_max",
                       reason = "easier and faster min/max than writing manual CAS loop",
                       issue = "48655")]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_max(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { $max_fn(self.v.get(), val, order) }
                }
            }

            doc_comment! {
                concat!("Minimum with the current value.

Finds the minimum of the current value and the argument `val`, and
sets the new value to the result.

Returns the previous value.

`fetch_min` takes an [`Ordering`] argument which describes the memory ordering
of this operation. All ordering modes are possible. Note that using
[`Acquire`] makes the store part of this operation [`Relaxed`], and
using [`Release`] makes the load part [`Relaxed`].

[`Ordering`]: enum.Ordering.html
[`Relaxed`]: enum.Ordering.html#variant.Relaxed
[`Release`]: enum.Ordering.html#variant.Release
[`Acquire`]: enum.Ordering.html#variant.Acquire

# Examples

```
#![feature(atomic_min_max)]
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(23);
assert_eq!(foo.fetch_min(42, Ordering::Relaxed), 23);
assert_eq!(foo.load(Ordering::Relaxed), 23);
assert_eq!(foo.fetch_min(22, Ordering::Relaxed), 23);
assert_eq!(foo.load(Ordering::Relaxed), 22);
```

If you want to obtain the minimum value in one step, you can use the following:

```
#![feature(atomic_min_max)]
", $extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};

let foo = ", stringify!($atomic_type), "::new(23);
let bar = 12;
let min_foo = foo.fetch_min(bar, Ordering::SeqCst).min(bar);
assert_eq!(min_foo, 12);
```"),
                #[inline]
                #[unstable(feature = "atomic_min_max",
                       reason = "easier and faster min/max than writing manual CAS loop",
                       issue = "48655")]
                #[cfg(target_has_atomic = "cas")]
                pub fn fetch_min(&self, val: $int_type, order: Ordering) -> $int_type {
                    unsafe { $min_fn(self.v.get(), val, order) }
                }
            }

        }
    }
}

#[cfg(target_has_atomic = "8")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i8", "../../../std/primitive.i8.html",
    "",
    atomic_min, atomic_max,
    1,
    "AtomicI8::new(0)",
    i8 AtomicI8 ATOMIC_I8_INIT
}
#[cfg(target_has_atomic = "8")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u8", "../../../std/primitive.u8.html",
    "",
    atomic_umin, atomic_umax,
    1,
    "AtomicU8::new(0)",
    u8 AtomicU8 ATOMIC_U8_INIT
}
#[cfg(target_has_atomic = "16")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i16", "../../../std/primitive.i16.html",
    "",
    atomic_min, atomic_max,
    2,
    "AtomicI16::new(0)",
    i16 AtomicI16 ATOMIC_I16_INIT
}
#[cfg(target_has_atomic = "16")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u16", "../../../std/primitive.u16.html",
    "",
    atomic_umin, atomic_umax,
    2,
    "AtomicU16::new(0)",
    u16 AtomicU16 ATOMIC_U16_INIT
}
#[cfg(target_has_atomic = "32")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i32", "../../../std/primitive.i32.html",
    "",
    atomic_min, atomic_max,
    4,
    "AtomicI32::new(0)",
    i32 AtomicI32 ATOMIC_I32_INIT
}
#[cfg(target_has_atomic = "32")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u32", "../../../std/primitive.u32.html",
    "",
    atomic_umin, atomic_umax,
    4,
    "AtomicU32::new(0)",
    u32 AtomicU32 ATOMIC_U32_INIT
}
#[cfg(target_has_atomic = "64")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i64", "../../../std/primitive.i64.html",
    "",
    atomic_min, atomic_max,
    8,
    "AtomicI64::new(0)",
    i64 AtomicI64 ATOMIC_I64_INIT
}
#[cfg(target_has_atomic = "64")]
atomic_int! {
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    stable(feature = "integer_atomics_stable", since = "1.34.0"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u64", "../../../std/primitive.u64.html",
    "",
    atomic_umin, atomic_umax,
    8,
    "AtomicU64::new(0)",
    u64 AtomicU64 ATOMIC_U64_INIT
}
#[cfg(target_has_atomic = "128")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "i128", "../../../std/primitive.i128.html",
    "#![feature(integer_atomics)]\n\n",
    atomic_min, atomic_max,
    16,
    "AtomicI128::new(0)",
    i128 AtomicI128 ATOMIC_I128_INIT
}
#[cfg(target_has_atomic = "128")]
atomic_int! {
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    unstable(feature = "integer_atomics", issue = "32976"),
    "u128", "../../../std/primitive.u128.html",
    "#![feature(integer_atomics)]\n\n",
    atomic_umin, atomic_umax,
    16,
    "AtomicU128::new(0)",
    u128 AtomicU128 ATOMIC_U128_INIT
}
#[cfg(target_pointer_width = "16")]
macro_rules! ptr_width {
    () => { 2 }
}
#[cfg(target_pointer_width = "32")]
macro_rules! ptr_width {
    () => { 4 }
}
#[cfg(target_pointer_width = "64")]
macro_rules! ptr_width {
    () => { 8 }
}
#[cfg(target_has_atomic = "ptr")]
atomic_int!{
    stable(feature = "rust1", since = "1.0.0"),
    stable(feature = "extended_compare_and_swap", since = "1.10.0"),
    stable(feature = "atomic_debug", since = "1.3.0"),
    stable(feature = "atomic_access", since = "1.15.0"),
    stable(feature = "atomic_from", since = "1.23.0"),
    stable(feature = "atomic_nand", since = "1.27.0"),
    stable(feature = "rust1", since = "1.0.0"),
    "isize", "../../../std/primitive.isize.html",
    "",
    atomic_min, atomic_max,
    ptr_width!(),
    "AtomicIsize::new(0)",
    isize AtomicIsize ATOMIC_ISIZE_INIT
}
#[cfg(target_has_atomic = "ptr")]
atomic_int!{
    stable(feature = "rust1", since = "1.0.0"),
    stable(feature = "extended_compare_and_swap", since = "1.10.0"),
    stable(feature = "atomic_debug", since = "1.3.0"),
    stable(feature = "atomic_access", since = "1.15.0"),
    stable(feature = "atomic_from", since = "1.23.0"),
    stable(feature = "atomic_nand", since = "1.27.0"),
    stable(feature = "rust1", since = "1.0.0"),
    "usize", "../../../std/primitive.usize.html",
    "",
    atomic_umin, atomic_umax,
    ptr_width!(),
    "AtomicUsize::new(0)",
    usize AtomicUsize ATOMIC_USIZE_INIT
}

#[inline]
#[cfg(target_has_atomic = "cas")]
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
unsafe fn atomic_store<T>(dst: *mut T, val: T, order: Ordering) {
    match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        Relaxed => intrinsics::atomic_store_relaxed(dst, val),
        SeqCst => intrinsics::atomic_store(dst, val),
        Acquire => panic!("there is no such thing as an acquire store"),
        AcqRel => panic!("there is no such thing as an acquire/release store"),
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
    }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_swap<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        AcqRel => intrinsics::atomic_xchg_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xchg(dst, val),
    }
}

/// Returns the previous value (like __sync_fetch_and_add).
#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_add<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xadd_acq(dst, val),
        Release => intrinsics::atomic_xadd_rel(dst, val),
        AcqRel => intrinsics::atomic_xadd_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xadd(dst, val),
    }
}

/// Returns the previous value (like __sync_fetch_and_sub).
#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_sub<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xsub_acq(dst, val),
        Release => intrinsics::atomic_xsub_rel(dst, val),
        AcqRel => intrinsics::atomic_xsub_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xsub(dst, val),
    }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
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
        (_, AcqRel) => panic!("there is no such thing as an acquire/release failure ordering"),
        (_, Release) => panic!("there is no such thing as a release failure ordering"),
        _ => panic!("a failure ordering can't be stronger than a success ordering"),
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
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
        (_, AcqRel) => panic!("there is no such thing as an acquire/release failure ordering"),
        (_, Release) => panic!("there is no such thing as a release failure ordering"),
        _ => panic!("a failure ordering can't be stronger than a success ordering"),
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_and<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_and_acq(dst, val),
        Release => intrinsics::atomic_and_rel(dst, val),
        AcqRel => intrinsics::atomic_and_acqrel(dst, val),
        Relaxed => intrinsics::atomic_and_relaxed(dst, val),
        SeqCst => intrinsics::atomic_and(dst, val),
    }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_nand<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_nand_acq(dst, val),
        Release => intrinsics::atomic_nand_rel(dst, val),
        AcqRel => intrinsics::atomic_nand_acqrel(dst, val),
        Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
        SeqCst => intrinsics::atomic_nand(dst, val),
    }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_or<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_or_acq(dst, val),
        Release => intrinsics::atomic_or_rel(dst, val),
        AcqRel => intrinsics::atomic_or_acqrel(dst, val),
        Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        SeqCst => intrinsics::atomic_or(dst, val),
    }
}

#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_xor<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xor_acq(dst, val),
        Release => intrinsics::atomic_xor_rel(dst, val),
        AcqRel => intrinsics::atomic_xor_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        SeqCst => intrinsics::atomic_xor(dst, val),
    }
}

/// returns the max value (signed comparison)
#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_max<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_max_acq(dst, val),
        Release => intrinsics::atomic_max_rel(dst, val),
        AcqRel => intrinsics::atomic_max_acqrel(dst, val),
        Relaxed => intrinsics::atomic_max_relaxed(dst, val),
        SeqCst => intrinsics::atomic_max(dst, val),
    }
}

/// returns the min value (signed comparison)
#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_min<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_min_acq(dst, val),
        Release => intrinsics::atomic_min_rel(dst, val),
        AcqRel => intrinsics::atomic_min_acqrel(dst, val),
        Relaxed => intrinsics::atomic_min_relaxed(dst, val),
        SeqCst => intrinsics::atomic_min(dst, val),
    }
}

/// returns the max value (signed comparison)
#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_umax<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_umax_acq(dst, val),
        Release => intrinsics::atomic_umax_rel(dst, val),
        AcqRel => intrinsics::atomic_umax_acqrel(dst, val),
        Relaxed => intrinsics::atomic_umax_relaxed(dst, val),
        SeqCst => intrinsics::atomic_umax(dst, val),
    }
}

/// returns the min value (signed comparison)
#[inline]
#[cfg(target_has_atomic = "cas")]
unsafe fn atomic_umin<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_umin_acq(dst, val),
        Release => intrinsics::atomic_umin_rel(dst, val),
        AcqRel => intrinsics::atomic_umin_acqrel(dst, val),
        Relaxed => intrinsics::atomic_umin_relaxed(dst, val),
        SeqCst => intrinsics::atomic_umin(dst, val),
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
///         while !self.flag.compare_and_swap(false, true, Ordering::Relaxed) {}
///         // This fence synchronizes-with store in `unlock`.
///         fence(Ordering::Acquire);
///     }
///
///     pub fn unlock(&self) {
///         self.flag.store(false, Ordering::Release);
///     }
/// }
/// ```
///
/// [`Ordering`]: enum.Ordering.html
/// [`Acquire`]: enum.Ordering.html#variant.Acquire
/// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
/// [`Release`]: enum.Ordering.html#variant.Release
/// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
/// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
pub fn fence(order: Ordering) {
    // On wasm32 it looks like fences aren't implemented in LLVM yet in that
    // they will cause LLVM to abort. The wasm instruction set doesn't have
    // fences right now. There's discussion online about the best way for tools
    // to conventionally implement fences at
    // https://github.com/WebAssembly/tool-conventions/issues/59. We should
    // follow that discussion and implement a solution when one comes about!
    #[cfg(not(target_arch = "wasm32"))]
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
/// `IMPORTANT_VARIABLE` and `IS_READ` since they are both
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
/// [`fence`]: fn.fence.html
/// [`Ordering`]: enum.Ordering.html
/// [`Acquire`]: enum.Ordering.html#variant.Acquire
/// [`SeqCst`]: enum.Ordering.html#variant.SeqCst
/// [`Release`]: enum.Ordering.html#variant.Release
/// [`AcqRel`]: enum.Ordering.html#variant.AcqRel
/// [`Relaxed`]: enum.Ordering.html#variant.Relaxed
/// [memory barriers]: https://www.kernel.org/doc/Documentation/memory-barriers.txt
#[inline]
#[stable(feature = "compiler_fences", since = "1.21.0")]
pub fn compiler_fence(order: Ordering) {
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


#[cfg(target_has_atomic = "8")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl fmt::Debug for AtomicBool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::SeqCst), f)
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl<T> fmt::Debug for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::SeqCst), f)
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "atomic_pointer", since = "1.24.0")]
impl<T> fmt::Pointer for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.load(Ordering::SeqCst), f)
    }
}
