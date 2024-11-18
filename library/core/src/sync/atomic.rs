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
//! ## Memory model for atomic accesses
//!
//! Rust atomics currently follow the same rules as [C++20 atomics][cpp], specifically the rules
//! from the [`intro.races`][cpp-intro.races] section, without the "consume" memory ordering. Since
//! C++ uses an object-based memory model whereas Rust is access-based, a bit of translation work
//! has to be done to apply the C++ rules to Rust: whenever C++ talks about "the value of an
//! object", we understand that to mean the resulting bytes obtained when doing a read. When the C++
//! standard talks about "the value of an atomic object", this refers to the result of doing an
//! atomic load (via the operations provided in this module). A "modification of an atomic object"
//! refers to an atomic store.
//!
//! The end result is *almost* equivalent to saying that creating a *shared reference* to one of the
//! Rust atomic types corresponds to creating an `atomic_ref` in C++, with the `atomic_ref` being
//! destroyed when the lifetime of the shared reference ends. The main difference is that Rust
//! permits concurrent atomic and non-atomic reads to the same memory as those cause no issue in the
//! C++ memory model, they are just forbidden in C++ because memory is partitioned into "atomic
//! objects" and "non-atomic objects" (with `atomic_ref` temporarily converting a non-atomic object
//! into an atomic object).
//!
//! The most important aspect of this model is that *data races* are undefined behavior. A data race
//! is defined as conflicting non-synchronized accesses where at least one of the accesses is
//! non-atomic. Here, accesses are *conflicting* if they affect overlapping regions of memory and at
//! least one of them is a write. They are *non-synchronized* if neither of them *happens-before*
//! the other, according to the happens-before order of the memory model.
//!
//! The other possible cause of undefined behavior in the memory model are mixed-size accesses: Rust
//! inherits the C++ limitation that non-synchronized conflicting atomic accesses may not partially
//! overlap. In other words, every pair of non-synchronized atomic accesses must be either disjoint,
//! access the exact same memory (including using the same access size), or both be reads.
//!
//! Each atomic access takes an [`Ordering`] which defines how the operation interacts with the
//! happens-before order. These orderings behave the same as the corresponding [C++20 atomic
//! orderings][cpp_memory_order]. For more information, see the [nomicon].
//!
//! [cpp]: https://en.cppreference.com/w/cpp/atomic
//! [cpp-intro.races]: https://timsong-cpp.github.io/cppwp/n4868/intro.multithread#intro.races
//! [cpp_memory_order]: https://en.cppreference.com/w/cpp/atomic/memory_order
//! [nomicon]: ../../../nomicon/atomics.html
//!
//! ```rust,no_run undefined_behavior
//! use std::sync::atomic::{AtomicU16, AtomicU8, Ordering};
//! use std::mem::transmute;
//! use std::thread;
//!
//! let atomic = AtomicU16::new(0);
//!
//! thread::scope(|s| {
//!     // This is UB: conflicting non-synchronized accesses, at least one of which is non-atomic.
//!     s.spawn(|| atomic.store(1, Ordering::Relaxed)); // atomic store
//!     s.spawn(|| unsafe { atomic.as_ptr().write(2) }); // non-atomic write
//! });
//!
//! thread::scope(|s| {
//!     // This is fine: the accesses do not conflict (as none of them performs any modification).
//!     // In C++ this would be disallowed since creating an `atomic_ref` precludes
//!     // further non-atomic accesses, but Rust does not have that limitation.
//!     s.spawn(|| atomic.load(Ordering::Relaxed)); // atomic load
//!     s.spawn(|| unsafe { atomic.as_ptr().read() }); // non-atomic read
//! });
//!
//! thread::scope(|s| {
//!     // This is fine: `join` synchronizes the code in a way such that the atomic
//!     // store happens-before the non-atomic write.
//!     let handle = s.spawn(|| atomic.store(1, Ordering::Relaxed)); // atomic store
//!     handle.join().unwrap(); // synchronize
//!     s.spawn(|| unsafe { atomic.as_ptr().write(2) }); // non-atomic write
//! });
//!
//! thread::scope(|s| {
//!     // This is UB: non-synchronized conflicting differently-sized atomic accesses.
//!     s.spawn(|| atomic.store(1, Ordering::Relaxed));
//!     s.spawn(|| unsafe {
//!         let differently_sized = transmute::<&AtomicU16, &AtomicU8>(&atomic);
//!         differently_sized.store(2, Ordering::Relaxed);
//!     });
//! });
//!
//! thread::scope(|s| {
//!     // This is fine: `join` synchronizes the code in a way such that
//!     // the 1-byte store happens-before the 2-byte store.
//!     let handle = s.spawn(|| atomic.store(1, Ordering::Relaxed));
//!     handle.join().unwrap();
//!     s.spawn(|| unsafe {
//!         let differently_sized = transmute::<&AtomicU16, &AtomicU8>(&atomic);
//!         differently_sized.store(2, Ordering::Relaxed);
//!     });
//! });
//! ```
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
//! For reference, the `std` library requires `AtomicBool`s and pointer-sized atomics, although
//! `core` does not.
//!
//! The `#[cfg(target_has_atomic)]` attribute can be used to conditionally
//! compile based on the target's supported bit widths. It is a key-value
//! option set for each supported size, with values "8", "16", "32", "64",
//! "128", and "ptr" for pointer-sized atomics.
//!
//! [lock-free]: https://en.wikipedia.org/wiki/Non-blocking_algorithm
//!
//! # Atomic accesses to read-only memory
//!
//! In general, *all* atomic accesses on read-only memory are undefined behavior. For instance, attempting
//! to do a `compare_exchange` that will definitely fail (making it conceptually a read-only
//! operation) can still cause a segmentation fault if the underlying memory page is mapped read-only. Since
//! atomic `load`s might be implemented using compare-exchange operations, even a `load` can fault
//! on read-only memory.
//!
//! For the purpose of this section, "read-only memory" is defined as memory that is read-only in
//! the underlying target, i.e., the pages are mapped with a read-only flag and any attempt to write
//! will cause a page fault. In particular, an `&u128` reference that points to memory that is
//! read-write mapped is *not* considered to point to "read-only memory". In Rust, almost all memory
//! is read-write; the only exceptions are memory created by `const` items or `static` items without
//! interior mutability, and memory that was specifically marked as read-only by the operating
//! system via platform-specific APIs.
//!
//! As an exception from the general rule stated above, "sufficiently small" atomic loads with
//! `Ordering::Relaxed` are implemented in a way that works on read-only memory, and are hence not
//! undefined behavior. The exact size limit for what makes a load "sufficiently small" varies
//! depending on the target:
//!
//! | `target_arch` | Size limit |
//! |---------------|---------|
//! | `x86`, `arm`, `mips`, `mips32r6`, `powerpc`, `riscv32`, `sparc`, `hexagon` | 4 bytes |
//! | `x86_64`, `aarch64`, `loongarch64`, `mips64`, `mips64r6`, `powerpc64`, `riscv64`, `sparc64`, `s390x` | 8 bytes |
//!
//! Atomics loads that are larger than this limit as well as atomic loads with ordering other
//! than `Relaxed`, as well as *all* atomic loads on targets not listed in the table, might still be
//! read-only under certain conditions, but that is not a stable guarantee and should not be relied
//! upon.
//!
//! If you need to do an acquire load on read-only memory, you can do a relaxed load followed by an
//! acquire fence instead.
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
//!
//!     let thread = thread::spawn(move || {
//!         spinlock_clone.store(0, Ordering::Release);
//!     });
//!
//!     // Wait for the other thread to release the lock
//!     while spinlock.load(Ordering::Acquire) != 0 {
//!         hint::spin_loop();
//!     }
//!
//!     if let Err(panic) = thread.join() {
//!         println!("Thread had an error: {panic:?}");
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
//! // Note that Relaxed ordering doesn't synchronize anything
//! // except the global thread counter itself.
//! let old_thread_count = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::Relaxed);
//! // Note that this number may not be true at the moment of printing
//! // because some other thread may have changed static value already.
//! println!("live threads: {}", old_thread_count + 1);
//! ```

#![stable(feature = "rust1", since = "1.0.0")]
#![cfg_attr(not(target_has_atomic_load_store = "8"), allow(dead_code))]
#![cfg_attr(not(target_has_atomic_load_store = "8"), allow(unused_imports))]
#![rustc_diagnostic_item = "atomic_mod"]
// Clippy complains about the pattern of "safe function calling unsafe function taking pointers".
// This happens with AtomicPtr intrinsics but is fine, as the pointers clippy is concerned about
// are just normal values that get loaded/stored, but not dereferenced.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use self::Ordering::*;
use crate::cell::UnsafeCell;
use crate::hint::spin_loop;
use crate::{fmt, intrinsics};

// Some architectures don't have byte-sized atomics, which results in LLVM
// emulating them using a LL/SC loop. However for AtomicBool we can take
// advantage of the fact that it only ever contains 0 or 1 and use atomic OR/AND
// instead, which LLVM can emulate using a larger atomic OR/AND operation.
//
// This list should only contain architectures which have word-sized atomic-or/
// atomic-and instructions but don't natively support byte-sized atomics.
#[cfg(target_has_atomic = "8")]
const EMULATE_ATOMIC_BOOL: bool =
    cfg!(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "loongarch64"));

/// A boolean type which can be safely shared between threads.
///
/// This type has the same size, alignment, and bit validity as a [`bool`].
///
/// **Note**: This type is only available on platforms that support atomic
/// loads and stores of `u8`.
#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "AtomicBool"]
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
/// This type has the same size and bit validity as a `*mut T`.
///
/// **Note**: This type is only available on platforms that support atomic
/// loads and stores of pointers. Its size depends on the target pointer's size.
#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "AtomicPtr")]
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
#[rustc_diagnostic_item = "Ordering"]
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
#[deprecated(
    since = "1.34.0",
    note = "the `new` function is now preferred",
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
    /// let atomic_true = AtomicBool::new(true);
    /// let atomic_false = AtomicBool::new(false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_atomic_new", since = "1.24.0")]
    #[must_use]
    pub const fn new(v: bool) -> AtomicBool {
        AtomicBool { v: UnsafeCell::new(v as u8) }
    }

    /// Creates a new `AtomicBool` from a pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{self, AtomicBool};
    ///
    /// // Get a pointer to an allocated value
    /// let ptr: *mut bool = Box::into_raw(Box::new(false));
    ///
    /// assert!(ptr.cast::<AtomicBool>().is_aligned());
    ///
    /// {
    ///     // Create an atomic view of the allocated value
    ///     let atomic = unsafe { AtomicBool::from_ptr(ptr) };
    ///
    ///     // Use `atomic` for atomic operations, possibly share it with other threads
    ///     atomic.store(true, atomic::Ordering::Relaxed);
    /// }
    ///
    /// // It's ok to non-atomically access the value behind `ptr`,
    /// // since the reference to the atomic ended its lifetime in the block above
    /// assert_eq!(unsafe { *ptr }, true);
    ///
    /// // Deallocate the value
    /// unsafe { drop(Box::from_raw(ptr)) }
    /// ```
    ///
    /// # Safety
    ///
    /// * `ptr` must be aligned to `align_of::<AtomicBool>()` (note that this is always true, since
    ///   `align_of::<AtomicBool>() == 1`).
    /// * `ptr` must be [valid] for both reads and writes for the whole lifetime `'a`.
    /// * You must adhere to the [Memory model for atomic accesses]. In particular, it is not
    ///   allowed to mix atomic and non-atomic accesses, or atomic accesses of different sizes,
    ///   without synchronization.
    ///
    /// [valid]: crate::ptr#safety
    /// [Memory model for atomic accesses]: self#memory-model-for-atomic-accesses
    #[stable(feature = "atomic_from_ptr", since = "1.75.0")]
    #[rustc_const_stable(feature = "const_atomic_from_ptr", since = "CURRENT_RUSTC_VERSION")]
    pub const unsafe fn from_ptr<'a>(ptr: *mut bool) -> &'a AtomicBool {
        // SAFETY: guaranteed by the caller
        unsafe { &*ptr.cast() }
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

    /// Gets atomic access to a `&mut bool`.
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
    pub fn from_mut(v: &mut bool) -> &mut Self {
        // SAFETY: the mutable reference guarantees unique ownership, and
        // alignment of both `bool` and `Self` is 1.
        unsafe { &mut *(v as *mut bool as *mut Self) }
    }

    /// Gets non-atomic access to a `&mut [AtomicBool]` slice.
    ///
    /// This is safe because the mutable reference guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(atomic_from_mut)]
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let mut some_bools = [const { AtomicBool::new(false) }; 10];
    ///
    /// let view: &mut [bool] = AtomicBool::get_mut_slice(&mut some_bools);
    /// assert_eq!(view, [false; 10]);
    /// view[..5].copy_from_slice(&[true; 5]);
    ///
    /// std::thread::scope(|s| {
    ///     for t in &some_bools[..5] {
    ///         s.spawn(move || assert_eq!(t.load(Ordering::Relaxed), true));
    ///     }
    ///
    ///     for f in &some_bools[5..] {
    ///         s.spawn(move || assert_eq!(f.load(Ordering::Relaxed), false));
    ///     }
    /// });
    /// ```
    #[inline]
    #[unstable(feature = "atomic_from_mut", issue = "76314")]
    pub fn get_mut_slice(this: &mut [Self]) -> &mut [bool] {
        // SAFETY: the mutable reference guarantees unique ownership.
        unsafe { &mut *(this as *mut [Self] as *mut [bool]) }
    }

    /// Gets atomic access to a `&mut [bool]` slice.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(atomic_from_mut)]
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let mut some_bools = [false; 10];
    /// let a = &*AtomicBool::from_mut_slice(&mut some_bools);
    /// std::thread::scope(|s| {
    ///     for i in 0..a.len() {
    ///         s.spawn(move || a[i].store(true, Ordering::Relaxed));
    ///     }
    /// });
    /// assert_eq!(some_bools, [true; 10]);
    /// ```
    #[inline]
    #[cfg(target_has_atomic_equal_alignment = "8")]
    #[unstable(feature = "atomic_from_mut", issue = "76314")]
    pub fn from_mut_slice(v: &mut [bool]) -> &mut [Self] {
        // SAFETY: the mutable reference guarantees unique ownership, and
        // alignment of both `bool` and `Self` is 1.
        unsafe { &mut *(v as *mut [bool] as *mut [Self]) }
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
    #[rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0")]
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
        if EMULATE_ATOMIC_BOOL {
            if val { self.fetch_or(true, order) } else { self.fetch_and(false, order) }
        } else {
            // SAFETY: data races are prevented by atomic intrinsics.
            unsafe { atomic_swap(self.v.get(), val as u8, order) != 0 }
        }
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
    #[deprecated(
        since = "1.50.0",
        note = "Use `compare_exchange` or `compare_exchange_weak` instead"
    )]
    #[cfg(target_has_atomic = "8")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn compare_exchange(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        if EMULATE_ATOMIC_BOOL {
            // Pick the strongest ordering from success and failure.
            let order = match (success, failure) {
                (SeqCst, _) => SeqCst,
                (_, SeqCst) => SeqCst,
                (AcqRel, _) => AcqRel,
                (_, AcqRel) => {
                    panic!("there is no such thing as an acquire-release failure ordering")
                }
                (Release, Acquire) => AcqRel,
                (Acquire, _) => Acquire,
                (_, Acquire) => Acquire,
                (Release, Relaxed) => Release,
                (_, Release) => panic!("there is no such thing as a release failure ordering"),
                (Relaxed, Relaxed) => Relaxed,
            };
            let old = if current == new {
                // This is a no-op, but we still need to perform the operation
                // for memory ordering reasons.
                self.fetch_or(false, order)
            } else {
                // This sets the value to the new one and returns the old one.
                self.swap(new, order)
            };
            if old == current { Ok(old) } else { Err(old) }
        } else {
            // SAFETY: data races are prevented by atomic intrinsics.
            match unsafe {
                atomic_compare_exchange(self.v.get(), current as u8, new as u8, success, failure)
            } {
                Ok(x) => Ok(x != 0),
                Err(x) => Err(x != 0),
            }
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
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn compare_exchange_weak(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        if EMULATE_ATOMIC_BOOL {
            return self.compare_exchange(current, new, success, failure);
        }

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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_xor(&self, val: bool, order: Ordering) -> bool {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_xor(self.v.get(), val as u8, order) != 0 }
    }

    /// Logical "not" with a boolean value.
    ///
    /// Performs a logical "not" operation on the current value, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// `fetch_not` takes an [`Ordering`] argument which describes the memory ordering
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
    /// assert_eq!(foo.fetch_not(Ordering::SeqCst), true);
    /// assert_eq!(foo.load(Ordering::SeqCst), false);
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(foo.fetch_not(Ordering::SeqCst), false);
    /// assert_eq!(foo.load(Ordering::SeqCst), true);
    /// ```
    #[inline]
    #[stable(feature = "atomic_bool_fetch_not", since = "1.81.0")]
    #[cfg(target_has_atomic = "8")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_not(&self, order: Ordering) -> bool {
        self.fetch_xor(true, order)
    }

    /// Returns a mutable pointer to the underlying [`bool`].
    ///
    /// Doing non-atomic reads and writes on the resulting boolean can be a data race.
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
    ///
    /// extern "C" {
    ///     fn my_atomic_op(arg: *mut bool);
    /// }
    ///
    /// let mut atomic = AtomicBool::new(true);
    /// unsafe {
    ///     my_atomic_op(atomic.as_ptr());
    /// }
    /// # }
    /// ```
    #[inline]
    #[stable(feature = "atomic_as_ptr", since = "1.70.0")]
    #[rustc_const_stable(feature = "atomic_as_ptr", since = "1.70.0")]
    #[rustc_never_returns_null_ptr]
    pub const fn as_ptr(&self) -> *mut bool {
        self.v.get().cast()
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
    /// [`Acquire`] or [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on `u8`.
    ///
    /// # Considerations
    ///
    /// This method is not magic;  it is not provided by the hardware.
    /// It is implemented in terms of [`AtomicBool::compare_exchange_weak`], and suffers from the same drawbacks.
    /// In particular, this method will not circumvent the [ABA Problem].
    ///
    /// [ABA Problem]: https://en.wikipedia.org/wiki/ABA_problem
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// let atomic_ptr = AtomicPtr::new(ptr);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_atomic_new", since = "1.24.0")]
    pub const fn new(p: *mut T) -> AtomicPtr<T> {
        AtomicPtr { p: UnsafeCell::new(p) }
    }

    /// Creates a new `AtomicPtr` from a pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{self, AtomicPtr};
    ///
    /// // Get a pointer to an allocated value
    /// let ptr: *mut *mut u8 = Box::into_raw(Box::new(std::ptr::null_mut()));
    ///
    /// assert!(ptr.cast::<AtomicPtr<u8>>().is_aligned());
    ///
    /// {
    ///     // Create an atomic view of the allocated value
    ///     let atomic = unsafe { AtomicPtr::from_ptr(ptr) };
    ///
    ///     // Use `atomic` for atomic operations, possibly share it with other threads
    ///     atomic.store(std::ptr::NonNull::dangling().as_ptr(), atomic::Ordering::Relaxed);
    /// }
    ///
    /// // It's ok to non-atomically access the value behind `ptr`,
    /// // since the reference to the atomic ended its lifetime in the block above
    /// assert!(!unsafe { *ptr }.is_null());
    ///
    /// // Deallocate the value
    /// unsafe { drop(Box::from_raw(ptr)) }
    /// ```
    ///
    /// # Safety
    ///
    /// * `ptr` must be aligned to `align_of::<AtomicPtr<T>>()` (note that on some platforms this
    ///   can be bigger than `align_of::<*mut T>()`).
    /// * `ptr` must be [valid] for both reads and writes for the whole lifetime `'a`.
    /// * You must adhere to the [Memory model for atomic accesses]. In particular, it is not
    ///   allowed to mix atomic and non-atomic accesses, or atomic accesses of different sizes,
    ///   without synchronization.
    ///
    /// [valid]: crate::ptr#safety
    /// [Memory model for atomic accesses]: self#memory-model-for-atomic-accesses
    #[stable(feature = "atomic_from_ptr", since = "1.75.0")]
    #[rustc_const_stable(feature = "const_atomic_from_ptr", since = "CURRENT_RUSTC_VERSION")]
    pub const unsafe fn from_ptr<'a>(ptr: *mut *mut T) -> &'a AtomicPtr<T> {
        // SAFETY: guaranteed by the caller
        unsafe { &*ptr.cast() }
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

    /// Gets atomic access to a pointer.
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
    pub fn from_mut(v: &mut *mut T) -> &mut Self {
        let [] = [(); align_of::<AtomicPtr<()>>() - align_of::<*mut ()>()];
        // SAFETY:
        //  - the mutable reference guarantees unique ownership.
        //  - the alignment of `*mut T` and `Self` is the same on all platforms
        //    supported by rust, as verified above.
        unsafe { &mut *(v as *mut *mut T as *mut Self) }
    }

    /// Gets non-atomic access to a `&mut [AtomicPtr]` slice.
    ///
    /// This is safe because the mutable reference guarantees that no other threads are
    /// concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(atomic_from_mut)]
    /// use std::ptr::null_mut;
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let mut some_ptrs = [const { AtomicPtr::new(null_mut::<String>()) }; 10];
    ///
    /// let view: &mut [*mut String] = AtomicPtr::get_mut_slice(&mut some_ptrs);
    /// assert_eq!(view, [null_mut::<String>(); 10]);
    /// view
    ///     .iter_mut()
    ///     .enumerate()
    ///     .for_each(|(i, ptr)| *ptr = Box::into_raw(Box::new(format!("iteration#{i}"))));
    ///
    /// std::thread::scope(|s| {
    ///     for ptr in &some_ptrs {
    ///         s.spawn(move || {
    ///             let ptr = ptr.load(Ordering::Relaxed);
    ///             assert!(!ptr.is_null());
    ///
    ///             let name = unsafe { Box::from_raw(ptr) };
    ///             println!("Hello, {name}!");
    ///         });
    ///     }
    /// });
    /// ```
    #[inline]
    #[unstable(feature = "atomic_from_mut", issue = "76314")]
    pub fn get_mut_slice(this: &mut [Self]) -> &mut [*mut T] {
        // SAFETY: the mutable reference guarantees unique ownership.
        unsafe { &mut *(this as *mut [Self] as *mut [*mut T]) }
    }

    /// Gets atomic access to a slice of pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(atomic_from_mut)]
    /// use std::ptr::null_mut;
    /// use std::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let mut some_ptrs = [null_mut::<String>(); 10];
    /// let a = &*AtomicPtr::from_mut_slice(&mut some_ptrs);
    /// std::thread::scope(|s| {
    ///     for i in 0..a.len() {
    ///         s.spawn(move || {
    ///             let name = Box::new(format!("thread{i}"));
    ///             a[i].store(Box::into_raw(name), Ordering::Relaxed);
    ///         });
    ///     }
    /// });
    /// for p in some_ptrs {
    ///     assert!(!p.is_null());
    ///     let name = unsafe { Box::from_raw(p) };
    ///     println!("Hello, {name}!");
    /// }
    /// ```
    #[inline]
    #[cfg(target_has_atomic_equal_alignment = "ptr")]
    #[unstable(feature = "atomic_from_mut", issue = "76314")]
    pub fn from_mut_slice(v: &mut [*mut T]) -> &mut [Self] {
        // SAFETY:
        //  - the mutable reference guarantees unique ownership.
        //  - the alignment of `*mut T` and `Self` is the same on all platforms
        //    supported by rust, as verified above.
        unsafe { &mut *(v as *mut [*mut T] as *mut [Self]) }
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
    #[rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0")]
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
    /// let some_ptr = AtomicPtr::new(ptr);
    ///
    /// let value = some_ptr.load(Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// let some_ptr = AtomicPtr::new(ptr);
    ///
    /// let other_ptr = &mut 10;
    ///
    /// some_ptr.store(other_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// let some_ptr = AtomicPtr::new(ptr);
    ///
    /// let other_ptr = &mut 10;
    ///
    /// let value = some_ptr.swap(other_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// let some_ptr = AtomicPtr::new(ptr);
    ///
    /// let other_ptr = &mut 10;
    ///
    /// let value = some_ptr.compare_and_swap(ptr, other_ptr, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(
        since = "1.50.0",
        note = "Use `compare_exchange` or `compare_exchange_weak` instead"
    )]
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
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
    /// let some_ptr = AtomicPtr::new(ptr);
    ///
    /// let other_ptr = &mut 10;
    ///
    /// let value = some_ptr.compare_exchange(ptr, other_ptr,
    ///                                       Ordering::SeqCst, Ordering::Relaxed);
    /// ```
    #[inline]
    #[stable(feature = "extended_compare_and_swap", since = "1.10.0")]
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    /// [`Acquire`] or [`Relaxed`].
    ///
    /// **Note:** This method is only available on platforms that support atomic
    /// operations on pointers.
    ///
    /// # Considerations
    ///
    /// This method is not magic;  it is not provided by the hardware.
    /// It is implemented in terms of [`AtomicPtr::compare_exchange_weak`], and suffers from the same drawbacks.
    /// In particular, this method will not circumvent the [ABA Problem].
    ///
    /// [ABA Problem]: https://en.wikipedia.org/wiki/ABA_problem
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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

    /// Offsets the pointer's address by adding `val` (in units of `T`),
    /// returning the previous pointer.
    ///
    /// This is equivalent to using [`wrapping_add`] to atomically perform the
    /// equivalent of `ptr = ptr.wrapping_add(val);`.
    ///
    /// This method operates in units of `T`, which means that it cannot be used
    /// to offset the pointer by an amount which is not a multiple of
    /// `size_of::<T>()`. This can sometimes be inconvenient, as you may want to
    /// work with a deliberately misaligned pointer. In such cases, you may use
    /// the [`fetch_byte_add`](Self::fetch_byte_add) method instead.
    ///
    /// `fetch_ptr_add` takes an [`Ordering`] argument which describes the
    /// memory ordering of this operation. All ordering modes are possible. Note
    /// that using [`Acquire`] makes the store part of this operation
    /// [`Relaxed`], and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// [`wrapping_add`]: pointer::wrapping_add
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let atom = AtomicPtr::<i64>::new(core::ptr::null_mut());
    /// assert_eq!(atom.fetch_ptr_add(1, Ordering::Relaxed).addr(), 0);
    /// // Note: units of `size_of::<i64>()`.
    /// assert_eq!(atom.load(Ordering::Relaxed).addr(), 8);
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_ptr_add(&self, val: usize, order: Ordering) -> *mut T {
        self.fetch_byte_add(val.wrapping_mul(core::mem::size_of::<T>()), order)
    }

    /// Offsets the pointer's address by subtracting `val` (in units of `T`),
    /// returning the previous pointer.
    ///
    /// This is equivalent to using [`wrapping_sub`] to atomically perform the
    /// equivalent of `ptr = ptr.wrapping_sub(val);`.
    ///
    /// This method operates in units of `T`, which means that it cannot be used
    /// to offset the pointer by an amount which is not a multiple of
    /// `size_of::<T>()`. This can sometimes be inconvenient, as you may want to
    /// work with a deliberately misaligned pointer. In such cases, you may use
    /// the [`fetch_byte_sub`](Self::fetch_byte_sub) method instead.
    ///
    /// `fetch_ptr_sub` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible. Note that
    /// using [`Acquire`] makes the store part of this operation [`Relaxed`],
    /// and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// [`wrapping_sub`]: pointer::wrapping_sub
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let array = [1i32, 2i32];
    /// let atom = AtomicPtr::new(array.as_ptr().wrapping_add(1) as *mut _);
    ///
    /// assert!(core::ptr::eq(
    ///     atom.fetch_ptr_sub(1, Ordering::Relaxed),
    ///     &array[1],
    /// ));
    /// assert!(core::ptr::eq(atom.load(Ordering::Relaxed), &array[0]));
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_ptr_sub(&self, val: usize, order: Ordering) -> *mut T {
        self.fetch_byte_sub(val.wrapping_mul(core::mem::size_of::<T>()), order)
    }

    /// Offsets the pointer's address by adding `val` *bytes*, returning the
    /// previous pointer.
    ///
    /// This is equivalent to using [`wrapping_byte_add`] to atomically
    /// perform `ptr = ptr.wrapping_byte_add(val)`.
    ///
    /// `fetch_byte_add` takes an [`Ordering`] argument which describes the
    /// memory ordering of this operation. All ordering modes are possible. Note
    /// that using [`Acquire`] makes the store part of this operation
    /// [`Relaxed`], and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// [`wrapping_byte_add`]: pointer::wrapping_byte_add
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let atom = AtomicPtr::<i64>::new(core::ptr::null_mut());
    /// assert_eq!(atom.fetch_byte_add(1, Ordering::Relaxed).addr(), 0);
    /// // Note: in units of bytes, not `size_of::<i64>()`.
    /// assert_eq!(atom.load(Ordering::Relaxed).addr(), 1);
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_byte_add(&self, val: usize, order: Ordering) -> *mut T {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_add(self.p.get(), core::ptr::without_provenance_mut(val), order).cast() }
    }

    /// Offsets the pointer's address by subtracting `val` *bytes*, returning the
    /// previous pointer.
    ///
    /// This is equivalent to using [`wrapping_byte_sub`] to atomically
    /// perform `ptr = ptr.wrapping_byte_sub(val)`.
    ///
    /// `fetch_byte_sub` takes an [`Ordering`] argument which describes the
    /// memory ordering of this operation. All ordering modes are possible. Note
    /// that using [`Acquire`] makes the store part of this operation
    /// [`Relaxed`], and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// [`wrapping_byte_sub`]: pointer::wrapping_byte_sub
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let atom = AtomicPtr::<i64>::new(core::ptr::without_provenance_mut(1));
    /// assert_eq!(atom.fetch_byte_sub(1, Ordering::Relaxed).addr(), 1);
    /// assert_eq!(atom.load(Ordering::Relaxed).addr(), 0);
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_byte_sub(&self, val: usize, order: Ordering) -> *mut T {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_sub(self.p.get(), core::ptr::without_provenance_mut(val), order).cast() }
    }

    /// Performs a bitwise "or" operation on the address of the current pointer,
    /// and the argument `val`, and stores a pointer with provenance of the
    /// current pointer and the resulting address.
    ///
    /// This is equivalent to using [`map_addr`] to atomically perform
    /// `ptr = ptr.map_addr(|a| a | val)`. This can be used in tagged
    /// pointer schemes to atomically set tag bits.
    ///
    /// **Caveat**: This operation returns the previous value. To compute the
    /// stored value without losing provenance, you may use [`map_addr`]. For
    /// example: `a.fetch_or(val).map_addr(|a| a | val)`.
    ///
    /// `fetch_or` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible. Note that
    /// using [`Acquire`] makes the store part of this operation [`Relaxed`],
    /// and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// This API and its claimed semantics are part of the Strict Provenance
    /// experiment, see the [module documentation for `ptr`][crate::ptr] for
    /// details.
    ///
    /// [`map_addr`]: pointer::map_addr
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let pointer = &mut 3i64 as *mut i64;
    ///
    /// let atom = AtomicPtr::<i64>::new(pointer);
    /// // Tag the bottom bit of the pointer.
    /// assert_eq!(atom.fetch_or(1, Ordering::Relaxed).addr() & 1, 0);
    /// // Extract and untag.
    /// let tagged = atom.load(Ordering::Relaxed);
    /// assert_eq!(tagged.addr() & 1, 1);
    /// assert_eq!(tagged.map_addr(|p| p & !1), pointer);
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_or(&self, val: usize, order: Ordering) -> *mut T {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_or(self.p.get(), core::ptr::without_provenance_mut(val), order).cast() }
    }

    /// Performs a bitwise "and" operation on the address of the current
    /// pointer, and the argument `val`, and stores a pointer with provenance of
    /// the current pointer and the resulting address.
    ///
    /// This is equivalent to using [`map_addr`] to atomically perform
    /// `ptr = ptr.map_addr(|a| a & val)`. This can be used in tagged
    /// pointer schemes to atomically unset tag bits.
    ///
    /// **Caveat**: This operation returns the previous value. To compute the
    /// stored value without losing provenance, you may use [`map_addr`]. For
    /// example: `a.fetch_and(val).map_addr(|a| a & val)`.
    ///
    /// `fetch_and` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible. Note that
    /// using [`Acquire`] makes the store part of this operation [`Relaxed`],
    /// and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// This API and its claimed semantics are part of the Strict Provenance
    /// experiment, see the [module documentation for `ptr`][crate::ptr] for
    /// details.
    ///
    /// [`map_addr`]: pointer::map_addr
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let pointer = &mut 3i64 as *mut i64;
    /// // A tagged pointer
    /// let atom = AtomicPtr::<i64>::new(pointer.map_addr(|a| a | 1));
    /// assert_eq!(atom.fetch_or(1, Ordering::Relaxed).addr() & 1, 1);
    /// // Untag, and extract the previously tagged pointer.
    /// let untagged = atom.fetch_and(!1, Ordering::Relaxed)
    ///     .map_addr(|a| a & !1);
    /// assert_eq!(untagged, pointer);
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_and(&self, val: usize, order: Ordering) -> *mut T {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_and(self.p.get(), core::ptr::without_provenance_mut(val), order).cast() }
    }

    /// Performs a bitwise "xor" operation on the address of the current
    /// pointer, and the argument `val`, and stores a pointer with provenance of
    /// the current pointer and the resulting address.
    ///
    /// This is equivalent to using [`map_addr`] to atomically perform
    /// `ptr = ptr.map_addr(|a| a ^ val)`. This can be used in tagged
    /// pointer schemes to atomically toggle tag bits.
    ///
    /// **Caveat**: This operation returns the previous value. To compute the
    /// stored value without losing provenance, you may use [`map_addr`]. For
    /// example: `a.fetch_xor(val).map_addr(|a| a ^ val)`.
    ///
    /// `fetch_xor` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible. Note that
    /// using [`Acquire`] makes the store part of this operation [`Relaxed`],
    /// and using [`Release`] makes the load part [`Relaxed`].
    ///
    /// **Note**: This method is only available on platforms that support atomic
    /// operations on [`AtomicPtr`].
    ///
    /// This API and its claimed semantics are part of the Strict Provenance
    /// experiment, see the [module documentation for `ptr`][crate::ptr] for
    /// details.
    ///
    /// [`map_addr`]: pointer::map_addr
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(strict_provenance_atomic_ptr)]
    /// use core::sync::atomic::{AtomicPtr, Ordering};
    ///
    /// let pointer = &mut 3i64 as *mut i64;
    /// let atom = AtomicPtr::<i64>::new(pointer);
    ///
    /// // Toggle a tag bit on the pointer.
    /// atom.fetch_xor(1, Ordering::Relaxed);
    /// assert_eq!(atom.load(Ordering::Relaxed).addr() & 1, 1);
    /// ```
    #[inline]
    #[cfg(target_has_atomic = "ptr")]
    #[unstable(feature = "strict_provenance_atomic_ptr", issue = "99108")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub fn fetch_xor(&self, val: usize, order: Ordering) -> *mut T {
        // SAFETY: data races are prevented by atomic intrinsics.
        unsafe { atomic_xor(self.p.get(), core::ptr::without_provenance_mut(val), order).cast() }
    }

    /// Returns a mutable pointer to the underlying pointer.
    ///
    /// Doing non-atomic reads and writes on the resulting pointer can be a data race.
    /// This method is mostly useful for FFI, where the function signature may use
    /// `*mut *mut T` instead of `&AtomicPtr<T>`.
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
    /// use std::sync::atomic::AtomicPtr;
    ///
    /// extern "C" {
    ///     fn my_atomic_op(arg: *mut *mut u32);
    /// }
    ///
    /// let mut value = 17;
    /// let atomic = AtomicPtr::new(&mut value);
    ///
    /// // SAFETY: Safe as long as `my_atomic_op` is atomic.
    /// unsafe {
    ///     my_atomic_op(atomic.as_ptr());
    /// }
    /// ```
    #[inline]
    #[stable(feature = "atomic_as_ptr", since = "1.70.0")]
    #[rustc_const_stable(feature = "atomic_as_ptr", since = "1.70.0")]
    #[rustc_never_returns_null_ptr]
    pub const fn as_ptr(&self) -> *mut *mut T {
        self.p.get()
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
    /// assert_eq!(format!("{atomic_bool:?}"), "true")
    /// ```
    #[inline]
    fn from(b: bool) -> Self {
        Self::new(b)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "atomic_from", since = "1.23.0")]
impl<T> From<*mut T> for AtomicPtr<T> {
    /// Converts a `*mut T` into an `AtomicPtr<T>`.
    #[inline]
    fn from(p: *mut T) -> Self {
        Self::new(p)
    }
}

#[allow(unused_macros)] // This macro ends up being unused on some architectures.
macro_rules! if_8_bit {
    (u8, $( yes = [$($yes:tt)*], )? $( no = [$($no:tt)*], )? ) => { concat!("", $($($yes)*)?) };
    (i8, $( yes = [$($yes:tt)*], )? $( no = [$($no:tt)*], )? ) => { concat!("", $($($yes)*)?) };
    ($_:ident, $( yes = [$($yes:tt)*], )? $( no = [$($no:tt)*], )? ) => { concat!("", $($($no)*)?) };
}

#[cfg(target_has_atomic_load_store)]
macro_rules! atomic_int {
    ($cfg_cas:meta,
     $cfg_align:meta,
     $stable:meta,
     $stable_cxchg:meta,
     $stable_debug:meta,
     $stable_access:meta,
     $stable_from:meta,
     $stable_nand:meta,
     $const_stable_new:meta,
     $const_stable_into_inner:meta,
     $diagnostic_item:meta,
     $s_int_type:literal,
     $extra_feature:expr,
     $min_fn:ident, $max_fn:ident,
     $align:expr,
     $int_type:ident $atomic_type:ident) => {
        /// An integer type which can be safely shared between threads.
        ///
        /// This type has the same
        #[doc = if_8_bit!(
            $int_type,
            yes = ["size, alignment, and bit validity"],
            no = ["size and bit validity"],
        )]
        /// as the underlying integer type, [`
        #[doc = $s_int_type]
        /// `].
        #[doc = if_8_bit! {
            $int_type,
            no = [
                "However, the alignment of this type is always equal to its ",
                "size, even on targets where [`", $s_int_type, "`] has a ",
                "lesser alignment."
            ],
        }]
        ///
        /// For more about the differences between atomic types and
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
        #[$diagnostic_item]
        #[repr(C, align($align))]
        pub struct $atomic_type {
            v: UnsafeCell<$int_type>,
        }

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
                fmt::Debug::fmt(&self.load(Ordering::Relaxed), f)
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
            #[$const_stable_new]
            #[must_use]
            pub const fn new(v: $int_type) -> Self {
                Self {v: UnsafeCell::new(v)}
            }

            /// Creates a new reference to an atomic integer from a pointer.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!($extra_feature, "use std::sync::atomic::{self, ", stringify!($atomic_type), "};")]
            ///
            /// // Get a pointer to an allocated value
            #[doc = concat!("let ptr: *mut ", stringify!($int_type), " = Box::into_raw(Box::new(0));")]
            ///
            #[doc = concat!("assert!(ptr.cast::<", stringify!($atomic_type), ">().is_aligned());")]
            ///
            /// {
            ///     // Create an atomic view of the allocated value
            // SAFETY: this is a doc comment, tidy, it can't hurt you (also guaranteed by the construction of `ptr` and the assert above)
            #[doc = concat!("    let atomic = unsafe {", stringify!($atomic_type), "::from_ptr(ptr) };")]
            ///
            ///     // Use `atomic` for atomic operations, possibly share it with other threads
            ///     atomic.store(1, atomic::Ordering::Relaxed);
            /// }
            ///
            /// // It's ok to non-atomically access the value behind `ptr`,
            /// // since the reference to the atomic ended its lifetime in the block above
            /// assert_eq!(unsafe { *ptr }, 1);
            ///
            /// // Deallocate the value
            /// unsafe { drop(Box::from_raw(ptr)) }
            /// ```
            ///
            /// # Safety
            ///
            /// * `ptr` must be aligned to
            #[doc = concat!("  `align_of::<", stringify!($atomic_type), ">()`")]
            #[doc = if_8_bit!{
                $int_type,
                yes = [
                    "  (note that this is always true, since `align_of::<",
                    stringify!($atomic_type), ">() == 1`)."
                ],
                no = [
                    "  (note that on some platforms this can be bigger than `align_of::<",
                    stringify!($int_type), ">()`)."
                ],
            }]
            /// * `ptr` must be [valid] for both reads and writes for the whole lifetime `'a`.
            /// * You must adhere to the [Memory model for atomic accesses]. In particular, it is not
            ///   allowed to mix atomic and non-atomic accesses, or atomic accesses of different sizes,
            ///   without synchronization.
            ///
            /// [valid]: crate::ptr#safety
            /// [Memory model for atomic accesses]: self#memory-model-for-atomic-accesses
            #[stable(feature = "atomic_from_ptr", since = "1.75.0")]
            #[rustc_const_stable(feature = "const_atomic_from_ptr", since = "CURRENT_RUSTC_VERSION")]
            pub const unsafe fn from_ptr<'a>(ptr: *mut $int_type) -> &'a $atomic_type {
                // SAFETY: guaranteed by the caller
                unsafe { &*ptr.cast() }
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
            #[doc = if_8_bit! {
                $int_type,
                no = [
                    "**Note:** This function is only available on targets where `",
                    stringify!($int_type), "` has an alignment of ", $align, " bytes."
                ],
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
            pub fn from_mut(v: &mut $int_type) -> &mut Self {
                let [] = [(); align_of::<Self>() - align_of::<$int_type>()];
                // SAFETY:
                //  - the mutable reference guarantees unique ownership.
                //  - the alignment of `$int_type` and `Self` is the
                //    same, as promised by $cfg_align and verified above.
                unsafe { &mut *(v as *mut $int_type as *mut Self) }
            }

            #[doc = concat!("Get non-atomic access to a `&mut [", stringify!($atomic_type), "]` slice")]
            ///
            /// This is safe because the mutable reference guarantees that no other threads are
            /// concurrently accessing the atomic data.
            ///
            /// # Examples
            ///
            /// ```
            /// #![feature(atomic_from_mut)]
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            #[doc = concat!("let mut some_ints = [const { ", stringify!($atomic_type), "::new(0) }; 10];")]
            ///
            #[doc = concat!("let view: &mut [", stringify!($int_type), "] = ", stringify!($atomic_type), "::get_mut_slice(&mut some_ints);")]
            /// assert_eq!(view, [0; 10]);
            /// view
            ///     .iter_mut()
            ///     .enumerate()
            ///     .for_each(|(idx, int)| *int = idx as _);
            ///
            /// std::thread::scope(|s| {
            ///     some_ints
            ///         .iter()
            ///         .enumerate()
            ///         .for_each(|(idx, int)| {
            ///             s.spawn(move || assert_eq!(int.load(Ordering::Relaxed), idx as _));
            ///         })
            /// });
            /// ```
            #[inline]
            #[unstable(feature = "atomic_from_mut", issue = "76314")]
            pub fn get_mut_slice(this: &mut [Self]) -> &mut [$int_type] {
                // SAFETY: the mutable reference guarantees unique ownership.
                unsafe { &mut *(this as *mut [Self] as *mut [$int_type]) }
            }

            #[doc = concat!("Get atomic access to a `&mut [", stringify!($int_type), "]` slice.")]
            ///
            /// # Examples
            ///
            /// ```
            /// #![feature(atomic_from_mut)]
            #[doc = concat!($extra_feature, "use std::sync::atomic::{", stringify!($atomic_type), ", Ordering};")]
            ///
            /// let mut some_ints = [0; 10];
            #[doc = concat!("let a = &*", stringify!($atomic_type), "::from_mut_slice(&mut some_ints);")]
            /// std::thread::scope(|s| {
            ///     for i in 0..a.len() {
            ///         s.spawn(move || a[i].store(i as _, Ordering::Relaxed));
            ///     }
            /// });
            /// for (i, n) in some_ints.into_iter().enumerate() {
            ///     assert_eq!(i, n as usize);
            /// }
            /// ```
            #[inline]
            #[$cfg_align]
            #[unstable(feature = "atomic_from_mut", issue = "76314")]
            pub fn from_mut_slice(v: &mut [$int_type]) -> &mut [Self] {
                let [] = [(); align_of::<Self>() - align_of::<$int_type>()];
                // SAFETY:
                //  - the mutable reference guarantees unique ownership.
                //  - the alignment of `$int_type` and `Self` is the
                //    same, as promised by $cfg_align and verified above.
                unsafe { &mut *(v as *mut [$int_type] as *mut [Self]) }
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
            #[$const_stable_into_inner]
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[deprecated(
                since = "1.50.0",
                note = "Use `compare_exchange` or `compare_exchange_weak` instead")
            ]
            #[$cfg_cas]
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            /// [`Relaxed`]. The failure ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            /// [`Relaxed`]. The (failed) load ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`].
            ///
            /// **Note**: This method is only available on platforms that support atomic operations on
            #[doc = concat!("[`", $s_int_type, "`].")]
            ///
            /// # Considerations
            ///
            /// This method is not magic;  it is not provided by the hardware.
            /// It is implemented in terms of
            #[doc = concat!("[`", stringify!($atomic_type), "::compare_exchange_weak`],")]
            /// and suffers from the same drawbacks.
            /// In particular, this method will not circumvent the [ABA Problem].
            ///
            /// [ABA Problem]: https://en.wikipedia.org/wiki/ABA_problem
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            #[doc = concat!("let atomic = ", stringify!($atomic_type), "::new(1);")]
            ///
            /// // SAFETY: Safe as long as `my_atomic_op` is atomic.
            /// unsafe {
            ///     my_atomic_op(atomic.as_ptr());
            /// }
            /// # }
            /// ```
            #[inline]
            #[stable(feature = "atomic_as_ptr", since = "1.70.0")]
            #[rustc_const_stable(feature = "atomic_as_ptr", since = "1.70.0")]
            #[rustc_never_returns_null_ptr]
            pub const fn as_ptr(&self) -> *mut $int_type {
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicI8"),
    "i8",
    "",
    atomic_min, atomic_max,
    1,
    i8 AtomicI8
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicU8"),
    "u8",
    "",
    atomic_umin, atomic_umax,
    1,
    u8 AtomicU8
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicI16"),
    "i16",
    "",
    atomic_min, atomic_max,
    2,
    i16 AtomicI16
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicU16"),
    "u16",
    "",
    atomic_umin, atomic_umax,
    2,
    u16 AtomicU16
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicI32"),
    "i32",
    "",
    atomic_min, atomic_max,
    4,
    i32 AtomicI32
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicU32"),
    "u32",
    "",
    atomic_umin, atomic_umax,
    4,
    u32 AtomicU32
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicI64"),
    "i64",
    "",
    atomic_min, atomic_max,
    8,
    i64 AtomicI64
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
    rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicU64"),
    "u64",
    "",
    atomic_umin, atomic_umax,
    8,
    u64 AtomicU64
}
#[cfg(target_has_atomic_load_store = "128")]
atomic_int! {
    cfg(target_has_atomic = "128"),
    cfg(target_has_atomic_equal_alignment = "128"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    rustc_const_unstable(feature = "integer_atomics", issue = "99069"),
    rustc_const_unstable(feature = "integer_atomics", issue = "99069"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicI128"),
    "i128",
    "#![feature(integer_atomics)]\n\n",
    atomic_min, atomic_max,
    16,
    i128 AtomicI128
}
#[cfg(target_has_atomic_load_store = "128")]
atomic_int! {
    cfg(target_has_atomic = "128"),
    cfg(target_has_atomic_equal_alignment = "128"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    unstable(feature = "integer_atomics", issue = "99069"),
    rustc_const_unstable(feature = "integer_atomics", issue = "99069"),
    rustc_const_unstable(feature = "integer_atomics", issue = "99069"),
    cfg_attr(not(test), rustc_diagnostic_item = "AtomicU128"),
    "u128",
    "#![feature(integer_atomics)]\n\n",
    atomic_umin, atomic_umax,
    16,
    u128 AtomicU128
}

#[cfg(target_has_atomic_load_store = "ptr")]
macro_rules! atomic_int_ptr_sized {
    ( $($target_pointer_width:literal $align:literal)* ) => { $(
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
            rustc_const_stable(feature = "const_ptr_sized_atomics", since = "1.24.0"),
            rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
            cfg_attr(not(test), rustc_diagnostic_item = "AtomicIsize"),
            "isize",
            "",
            atomic_min, atomic_max,
            $align,
            isize AtomicIsize
        }
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
            rustc_const_stable(feature = "const_ptr_sized_atomics", since = "1.24.0"),
            rustc_const_stable(feature = "const_atomic_into_inner", since = "1.79.0"),
            cfg_attr(not(test), rustc_diagnostic_item = "AtomicUsize"),
            "usize",
            "",
            atomic_umin, atomic_umax,
            $align,
            usize AtomicUsize
        }

        /// An [`AtomicIsize`] initialized to `0`.
        #[cfg(target_pointer_width = $target_pointer_width)]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[deprecated(
            since = "1.34.0",
            note = "the `new` function is now preferred",
            suggestion = "AtomicIsize::new(0)",
        )]
        pub const ATOMIC_ISIZE_INIT: AtomicIsize = AtomicIsize::new(0);

        /// An [`AtomicUsize`] initialized to `0`.
        #[cfg(target_pointer_width = $target_pointer_width)]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[deprecated(
            since = "1.34.0",
            note = "the `new` function is now preferred",
            suggestion = "AtomicUsize::new(0)",
        )]
        pub const ATOMIC_USIZE_INIT: AtomicUsize = AtomicUsize::new(0);
    )* };
}

#[cfg(target_has_atomic_load_store = "ptr")]
atomic_int_ptr_sized! {
    "16" 2
    "32" 4
    "64" 8
}

#[inline]
#[cfg(target_has_atomic)]
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
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_store<T: Copy>(dst: *mut T, val: T, order: Ordering) {
    // SAFETY: the caller must uphold the safety contract for `atomic_store`.
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_store_relaxed(dst, val),
            Release => intrinsics::atomic_store_release(dst, val),
            SeqCst => intrinsics::atomic_store_seqcst(dst, val),
            Acquire => panic!("there is no such thing as an acquire store"),
            AcqRel => panic!("there is no such thing as an acquire-release store"),
        }
    }
}

#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_load<T: Copy>(dst: *const T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_load`.
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_load_relaxed(dst),
            Acquire => intrinsics::atomic_load_acquire(dst),
            SeqCst => intrinsics::atomic_load_seqcst(dst),
            Release => panic!("there is no such thing as a release load"),
            AcqRel => panic!("there is no such thing as an acquire-release load"),
        }
    }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_swap<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_swap`.
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
            Acquire => intrinsics::atomic_xchg_acquire(dst, val),
            Release => intrinsics::atomic_xchg_release(dst, val),
            AcqRel => intrinsics::atomic_xchg_acqrel(dst, val),
            SeqCst => intrinsics::atomic_xchg_seqcst(dst, val),
        }
    }
}

/// Returns the previous value (like __sync_fetch_and_add).
#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_add<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_add`.
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
            Acquire => intrinsics::atomic_xadd_acquire(dst, val),
            Release => intrinsics::atomic_xadd_release(dst, val),
            AcqRel => intrinsics::atomic_xadd_acqrel(dst, val),
            SeqCst => intrinsics::atomic_xadd_seqcst(dst, val),
        }
    }
}

/// Returns the previous value (like __sync_fetch_and_sub).
#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_sub<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_sub`.
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
            Acquire => intrinsics::atomic_xsub_acquire(dst, val),
            Release => intrinsics::atomic_xsub_release(dst, val),
            AcqRel => intrinsics::atomic_xsub_acqrel(dst, val),
            SeqCst => intrinsics::atomic_xsub_seqcst(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            (Relaxed, Relaxed) => intrinsics::atomic_cxchg_relaxed_relaxed(dst, old, new),
            (Relaxed, Acquire) => intrinsics::atomic_cxchg_relaxed_acquire(dst, old, new),
            (Relaxed, SeqCst) => intrinsics::atomic_cxchg_relaxed_seqcst(dst, old, new),
            (Acquire, Relaxed) => intrinsics::atomic_cxchg_acquire_relaxed(dst, old, new),
            (Acquire, Acquire) => intrinsics::atomic_cxchg_acquire_acquire(dst, old, new),
            (Acquire, SeqCst) => intrinsics::atomic_cxchg_acquire_seqcst(dst, old, new),
            (Release, Relaxed) => intrinsics::atomic_cxchg_release_relaxed(dst, old, new),
            (Release, Acquire) => intrinsics::atomic_cxchg_release_acquire(dst, old, new),
            (Release, SeqCst) => intrinsics::atomic_cxchg_release_seqcst(dst, old, new),
            (AcqRel, Relaxed) => intrinsics::atomic_cxchg_acqrel_relaxed(dst, old, new),
            (AcqRel, Acquire) => intrinsics::atomic_cxchg_acqrel_acquire(dst, old, new),
            (AcqRel, SeqCst) => intrinsics::atomic_cxchg_acqrel_seqcst(dst, old, new),
            (SeqCst, Relaxed) => intrinsics::atomic_cxchg_seqcst_relaxed(dst, old, new),
            (SeqCst, Acquire) => intrinsics::atomic_cxchg_seqcst_acquire(dst, old, new),
            (SeqCst, SeqCst) => intrinsics::atomic_cxchg_seqcst_seqcst(dst, old, new),
            (_, AcqRel) => panic!("there is no such thing as an acquire-release failure ordering"),
            (_, Release) => panic!("there is no such thing as a release failure ordering"),
        }
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
            (Relaxed, Relaxed) => intrinsics::atomic_cxchgweak_relaxed_relaxed(dst, old, new),
            (Relaxed, Acquire) => intrinsics::atomic_cxchgweak_relaxed_acquire(dst, old, new),
            (Relaxed, SeqCst) => intrinsics::atomic_cxchgweak_relaxed_seqcst(dst, old, new),
            (Acquire, Relaxed) => intrinsics::atomic_cxchgweak_acquire_relaxed(dst, old, new),
            (Acquire, Acquire) => intrinsics::atomic_cxchgweak_acquire_acquire(dst, old, new),
            (Acquire, SeqCst) => intrinsics::atomic_cxchgweak_acquire_seqcst(dst, old, new),
            (Release, Relaxed) => intrinsics::atomic_cxchgweak_release_relaxed(dst, old, new),
            (Release, Acquire) => intrinsics::atomic_cxchgweak_release_acquire(dst, old, new),
            (Release, SeqCst) => intrinsics::atomic_cxchgweak_release_seqcst(dst, old, new),
            (AcqRel, Relaxed) => intrinsics::atomic_cxchgweak_acqrel_relaxed(dst, old, new),
            (AcqRel, Acquire) => intrinsics::atomic_cxchgweak_acqrel_acquire(dst, old, new),
            (AcqRel, SeqCst) => intrinsics::atomic_cxchgweak_acqrel_seqcst(dst, old, new),
            (SeqCst, Relaxed) => intrinsics::atomic_cxchgweak_seqcst_relaxed(dst, old, new),
            (SeqCst, Acquire) => intrinsics::atomic_cxchgweak_seqcst_acquire(dst, old, new),
            (SeqCst, SeqCst) => intrinsics::atomic_cxchgweak_seqcst_seqcst(dst, old, new),
            (_, AcqRel) => panic!("there is no such thing as an acquire-release failure ordering"),
            (_, Release) => panic!("there is no such thing as a release failure ordering"),
        }
    };
    if ok { Ok(val) } else { Err(val) }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_and<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_and`
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_and_relaxed(dst, val),
            Acquire => intrinsics::atomic_and_acquire(dst, val),
            Release => intrinsics::atomic_and_release(dst, val),
            AcqRel => intrinsics::atomic_and_acqrel(dst, val),
            SeqCst => intrinsics::atomic_and_seqcst(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_nand<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_nand`
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
            Acquire => intrinsics::atomic_nand_acquire(dst, val),
            Release => intrinsics::atomic_nand_release(dst, val),
            AcqRel => intrinsics::atomic_nand_acqrel(dst, val),
            SeqCst => intrinsics::atomic_nand_seqcst(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_or<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_or`
    unsafe {
        match order {
            SeqCst => intrinsics::atomic_or_seqcst(dst, val),
            Acquire => intrinsics::atomic_or_acquire(dst, val),
            Release => intrinsics::atomic_or_release(dst, val),
            AcqRel => intrinsics::atomic_or_acqrel(dst, val),
            Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        }
    }
}

#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_xor<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_xor`
    unsafe {
        match order {
            SeqCst => intrinsics::atomic_xor_seqcst(dst, val),
            Acquire => intrinsics::atomic_xor_acquire(dst, val),
            Release => intrinsics::atomic_xor_release(dst, val),
            AcqRel => intrinsics::atomic_xor_acqrel(dst, val),
            Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        }
    }
}

/// returns the max value (signed comparison)
#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_max<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_max`
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_max_relaxed(dst, val),
            Acquire => intrinsics::atomic_max_acquire(dst, val),
            Release => intrinsics::atomic_max_release(dst, val),
            AcqRel => intrinsics::atomic_max_acqrel(dst, val),
            SeqCst => intrinsics::atomic_max_seqcst(dst, val),
        }
    }
}

/// returns the min value (signed comparison)
#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_min<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_min`
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_min_relaxed(dst, val),
            Acquire => intrinsics::atomic_min_acquire(dst, val),
            Release => intrinsics::atomic_min_release(dst, val),
            AcqRel => intrinsics::atomic_min_acqrel(dst, val),
            SeqCst => intrinsics::atomic_min_seqcst(dst, val),
        }
    }
}

/// returns the max value (unsigned comparison)
#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_umax<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_umax`
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_umax_relaxed(dst, val),
            Acquire => intrinsics::atomic_umax_acquire(dst, val),
            Release => intrinsics::atomic_umax_release(dst, val),
            AcqRel => intrinsics::atomic_umax_acqrel(dst, val),
            SeqCst => intrinsics::atomic_umax_seqcst(dst, val),
        }
    }
}

/// returns the min value (unsigned comparison)
#[inline]
#[cfg(target_has_atomic)]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn atomic_umin<T: Copy>(dst: *mut T, val: T, order: Ordering) -> T {
    // SAFETY: the caller must uphold the safety contract for `atomic_umin`
    unsafe {
        match order {
            Relaxed => intrinsics::atomic_umin_relaxed(dst, val),
            Acquire => intrinsics::atomic_umin_acquire(dst, val),
            Release => intrinsics::atomic_umin_release(dst, val),
            AcqRel => intrinsics::atomic_umin_acqrel(dst, val),
            SeqCst => intrinsics::atomic_umin_seqcst(dst, val),
        }
    }
}

/// An atomic fence.
///
/// Fences create synchronization between themselves and atomic operations or fences in other
/// threads. To achieve this, a fence prevents the compiler and CPU from reordering certain types of
/// memory operations around it.
///
/// A fence 'A' which has (at least) [`Release`] ordering semantics, synchronizes
/// with a fence 'B' with (at least) [`Acquire`] semantics, if and only if there
/// exist operations X and Y, both operating on some atomic object 'M' such
/// that A is sequenced before X, Y is sequenced before B and Y observes
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
/// Note that in the example above, it is crucial that the accesses to `x` are atomic. Fences cannot
/// be used to establish synchronization among non-atomic accesses in different threads. However,
/// thanks to the happens-before relationship between A and B, any non-atomic accesses that
/// happen-before A are now also properly synchronized with any non-atomic accesses that
/// happen-after B.
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
#[rustc_diagnostic_item = "fence"]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub fn fence(order: Ordering) {
    // SAFETY: using an atomic fence is safe.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_fence_acquire(),
            Release => intrinsics::atomic_fence_release(),
            AcqRel => intrinsics::atomic_fence_acqrel(),
            SeqCst => intrinsics::atomic_fence_seqcst(),
            Relaxed => panic!("there is no such thing as a relaxed fence"),
        }
    }
}

/// A "compiler-only" atomic fence.
///
/// Like [`fence`], this function establishes synchronization with other atomic operations and
/// fences. However, unlike [`fence`], `compiler_fence` only establishes synchronization with
/// operations *in the same thread*. This may at first sound rather useless, since code within a
/// thread is typically already totally ordered and does not need any further synchronization.
/// However, there are cases where code can run on the same thread without being ordered:
/// - The most common case is that of a *signal handler*: a signal handler runs in the same thread
///   as the code it interrupted, but it is not ordered with respect to that code. `compiler_fence`
///   can be used to establish synchronization between a thread and its signal handler, the same way
///   that `fence` can be used to establish synchronization across threads.
/// - Similar situations can arise in embedded programming with interrupt handlers, or in custom
///   implementations of preemptive green threads. In general, `compiler_fence` can establish
///   synchronization with code that is guaranteed to run on the same hardware CPU.
///
/// See [`fence`] for how a fence can be used to achieve synchronization. Note that just like
/// [`fence`], synchronization still requires atomic operations to be used in both threads -- it is
/// not possible to perform synchronization entirely with fences and non-atomic operations.
///
/// `compiler_fence` does not emit any machine code, but restricts the kinds of memory re-ordering
/// the compiler is allowed to do. `compiler_fence` corresponds to [`atomic_signal_fence`] in C and
/// C++.
///
/// [`atomic_signal_fence`]: https://en.cppreference.com/w/cpp/atomic/atomic_signal_fence
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
#[inline]
#[stable(feature = "compiler_fences", since = "1.21.0")]
#[rustc_diagnostic_item = "compiler_fence"]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub fn compiler_fence(order: Ordering) {
    // SAFETY: using an atomic fence is safe.
    unsafe {
        match order {
            Acquire => intrinsics::atomic_singlethreadfence_acquire(),
            Release => intrinsics::atomic_singlethreadfence_release(),
            AcqRel => intrinsics::atomic_singlethreadfence_acqrel(),
            SeqCst => intrinsics::atomic_singlethreadfence_seqcst(),
            Relaxed => panic!("there is no such thing as a relaxed compiler fence"),
        }
    }
}

#[cfg(target_has_atomic_load_store = "8")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl fmt::Debug for AtomicBool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::Relaxed), f)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "atomic_debug", since = "1.3.0")]
impl<T> fmt::Debug for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::Relaxed), f)
    }
}

#[cfg(target_has_atomic_load_store = "ptr")]
#[stable(feature = "atomic_pointer", since = "1.24.0")]
impl<T> fmt::Pointer for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.load(Ordering::Relaxed), f)
    }
}

/// Signals the processor that it is inside a busy-wait spin-loop ("spin lock").
///
/// This function is deprecated in favor of [`hint::spin_loop`].
///
/// [`hint::spin_loop`]: crate::hint::spin_loop
#[inline]
#[stable(feature = "spin_loop_hint", since = "1.24.0")]
#[deprecated(since = "1.51.0", note = "use hint::spin_loop instead")]
pub fn spin_loop_hint() {
    spin_loop()
}
