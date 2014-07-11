// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Core atomic primitives

use intrinsics;
use std::kinds::marker;
use ty::Unsafe;

/// An atomic boolean type.
pub struct AtomicBool {
    v: Unsafe<uint>,
    nocopy: marker::NoCopy
}

/// A signed atomic integer type, supporting basic atomic arithmetic operations
pub struct AtomicInt {
    v: Unsafe<int>,
    nocopy: marker::NoCopy
}

/// An unsigned atomic integer type, supporting basic atomic arithmetic operations
pub struct AtomicUint {
    v: Unsafe<uint>,
    nocopy: marker::NoCopy
}

/// An unsafe atomic pointer. Only supports basic atomic operations
pub struct AtomicPtr<T> {
    p: Unsafe<uint>,
    nocopy: marker::NoCopy
}

/// Atomic memory orderings
///
/// Memory orderings limit the ways that both the compiler and CPU may reorder
/// instructions around atomic operations. At its most restrictive,
/// "sequentially consistent" atomics allow neither reads nor writes
/// to be moved either before or after the atomic operation; on the other end
/// "relaxed" atomics allow all reorderings.
///
/// Rust's memory orderings are the same as in C++[1].
///
/// 1: http://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync
pub enum Ordering {
    /// No ordering constraints, only atomic operations
    Relaxed,
    /// When coupled with a store, all previous writes become visible
    /// to another thread that performs a load with `Acquire` ordering
    /// on the same value
    Release,
    /// When coupled with a load, all subsequent loads will see data
    /// written before a store with `Release` ordering on the same value
    /// in another thread
    Acquire,
    /// When coupled with a load, uses `Acquire` ordering, and with a store
    /// `Release` ordering
    AcqRel,
    /// Like `AcqRel` with the additional guarantee that all threads see all
    /// sequentially consistent operations in the same order.
    SeqCst
}

/// An `AtomicBool` initialized to `false`
pub static INIT_ATOMIC_BOOL : AtomicBool = AtomicBool { v: Unsafe{value: 0,
                                                                  marker1: marker::InvariantType},
                                                        nocopy: marker::NoCopy };
/// An `AtomicInt` initialized to `0`
pub static INIT_ATOMIC_INT  : AtomicInt  = AtomicInt  { v: Unsafe{value: 0,
                                                                  marker1: marker::InvariantType},
                                                        nocopy: marker::NoCopy };
/// An `AtomicUint` initialized to `0`
pub static INIT_ATOMIC_UINT : AtomicUint = AtomicUint { v: Unsafe{value: 0,
                                                                  marker1: marker::InvariantType},
                                                        nocopy: marker::NoCopy };

// NB: Needs to be -1 (0b11111111...) to make fetch_nand work correctly
static UINT_TRUE: uint = -1;

impl AtomicBool {
    /// Create a new `AtomicBool`
    pub fn new(v: bool) -> AtomicBool {
        let val = if v { UINT_TRUE } else { 0 };
        AtomicBool { v: Unsafe::new(val), nocopy: marker::NoCopy }
    }

    /// Load the value
    #[inline]
    pub fn load(&self, order: Ordering) -> bool {
        unsafe { atomic_load(self.v.get() as *const uint, order) > 0 }
    }

    /// Store the value
    #[inline]
    pub fn store(&self, val: bool, order: Ordering) {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_store(self.v.get(), val, order); }
    }

    /// Store a value, returning the old value
    #[inline]
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_swap(self.v.get(), val, order) > 0 }
    }

    /// If the current value is the same as expected, store a new value
    ///
    /// Compare the current value with `old`; if they are the same then
    /// replace the current value with `new`. Return the previous value.
    /// If the return value is equal to `old` then the value was updated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use std::sync::atomics::{AtomicBool, SeqCst};
    /// use std::task::deschedule;
    ///
    /// fn main() {
    ///     let spinlock = Arc::new(AtomicBool::new(false));
    ///     let spinlock_clone = spinlock.clone();
    ///
    ///     spawn(proc() {
    ///         with_lock(&spinlock, || println!("task 1 in lock"));
    ///     });
    ///
    ///     spawn(proc() {
    ///         with_lock(&spinlock_clone, || println!("task 2 in lock"));
    ///     });
    /// }
    ///
    /// fn with_lock(spinlock: &Arc<AtomicBool>, f: || -> ()) {
    ///     // CAS loop until we are able to replace `false` with `true`
    ///     while spinlock.compare_and_swap(false, true, SeqCst) != false {
    ///         // Since tasks may not be preemptive (if they are green threads)
    ///         // yield to the scheduler to let the other task run. Low level
    ///         // concurrent code needs to take into account Rust's two threading
    ///         // models.
    ///         deschedule();
    ///     }
    ///
    ///     // Now we have the spinlock
    ///     f();
    ///
    ///     // Release the lock
    ///     spinlock.store(false, SeqCst);
    /// }
    /// ```
    #[inline]
    pub fn compare_and_swap(&self, old: bool, new: bool, order: Ordering) -> bool {
        let old = if old { UINT_TRUE } else { 0 };
        let new = if new { UINT_TRUE } else { 0 };

        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) > 0 }
    }

    /// A logical "and" operation
    ///
    /// Performs a logical "and" operation on the current value and the
    /// argument `val`, and sets the new value to the result.
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicBool, SeqCst};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_and(false, SeqCst));
    /// assert_eq!(false, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_and(true, SeqCst));
    /// assert_eq!(true, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_and(false, SeqCst));
    /// assert_eq!(false, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_and(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_and(self.v.get(), val, order) > 0 }
    }

    /// A logical "nand" operation
    ///
    /// Performs a logical "nand" operation on the current value and the
    /// argument `val`, and sets the new value to the result.
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicBool, SeqCst};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_nand(false, SeqCst));
    /// assert_eq!(true, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_nand(true, SeqCst));
    /// assert_eq!(0, foo.load(SeqCst) as int);
    /// assert_eq!(false, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_nand(false, SeqCst));
    /// assert_eq!(true, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_nand(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_nand(self.v.get(), val, order) > 0 }
    }

    /// A logical "or" operation
    ///
    /// Performs a logical "or" operation on the current value and the
    /// argument `val`, and sets the new value to the result.
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicBool, SeqCst};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_or(false, SeqCst));
    /// assert_eq!(true, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_or(true, SeqCst));
    /// assert_eq!(true, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_or(false, SeqCst));
    /// assert_eq!(false, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_or(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_or(self.v.get(), val, order) > 0 }
    }

    /// A logical "xor" operation
    ///
    /// Performs a logical "xor" operation on the current value and the
    /// argument `val`, and sets the new value to the result.
    /// Returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicBool, SeqCst};
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_xor(false, SeqCst));
    /// assert_eq!(true, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(true);
    /// assert_eq!(true, foo.fetch_xor(true, SeqCst));
    /// assert_eq!(false, foo.load(SeqCst));
    ///
    /// let foo = AtomicBool::new(false);
    /// assert_eq!(false, foo.fetch_xor(false, SeqCst));
    /// assert_eq!(false, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_xor(&self, val: bool, order: Ordering) -> bool {
        let val = if val { UINT_TRUE } else { 0 };

        unsafe { atomic_xor(self.v.get(), val, order) > 0 }
    }
}

impl AtomicInt {
    /// Create a new `AtomicInt`
    pub fn new(v: int) -> AtomicInt {
        AtomicInt {v: Unsafe::new(v), nocopy: marker::NoCopy}
    }

    /// Load the value
    #[inline]
    pub fn load(&self, order: Ordering) -> int {
        unsafe { atomic_load(self.v.get() as *const int, order) }
    }

    /// Store the value
    #[inline]
    pub fn store(&self, val: int, order: Ordering) {
        unsafe { atomic_store(self.v.get(), val, order); }
    }

    /// Store a value, returning the old value
    #[inline]
    pub fn swap(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    /// If the current value is the same as expected, store a new value
    ///
    /// Compare the current value with `old`; if they are the same then
    /// replace the current value with `new`. Return the previous value.
    /// If the return value is equal to `old` then the value was updated.
    #[inline]
    pub fn compare_and_swap(&self, old: int, new: int, order: Ordering) -> int {
        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) }
    }

    /// Add to the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicInt, SeqCst};
    ///
    /// let foo = AtomicInt::new(0);
    /// assert_eq!(0, foo.fetch_add(10, SeqCst));
    /// assert_eq!(10, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_add(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_add(self.v.get(), val, order) }
    }

    /// Subtract from the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicInt, SeqCst};
    ///
    /// let foo = AtomicInt::new(0);
    /// assert_eq!(0, foo.fetch_sub(10, SeqCst));
    /// assert_eq!(-10, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_sub(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_sub(self.v.get(), val, order) }
    }

    /// Bitwise and with the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0b101101);
    /// assert_eq!(0b101101, foo.fetch_and(0b110011, SeqCst));
    /// assert_eq!(0b100001, foo.load(SeqCst));
    #[inline]
    pub fn fetch_and(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_and(self.v.get(), val, order) }
    }

    /// Bitwise or with the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0b101101);
    /// assert_eq!(0b101101, foo.fetch_or(0b110011, SeqCst));
    /// assert_eq!(0b111111, foo.load(SeqCst));
    #[inline]
    pub fn fetch_or(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_or(self.v.get(), val, order) }
    }

    /// Bitwise xor with the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0b101101);
    /// assert_eq!(0b101101, foo.fetch_xor(0b110011, SeqCst));
    /// assert_eq!(0b011110, foo.load(SeqCst));
    #[inline]
    pub fn fetch_xor(&self, val: int, order: Ordering) -> int {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}

impl AtomicUint {
    /// Create a new `AtomicUint`
    pub fn new(v: uint) -> AtomicUint {
        AtomicUint { v: Unsafe::new(v), nocopy: marker::NoCopy }
    }

    /// Load the value
    #[inline]
    pub fn load(&self, order: Ordering) -> uint {
        unsafe { atomic_load(self.v.get() as *const uint, order) }
    }

    /// Store the value
    #[inline]
    pub fn store(&self, val: uint, order: Ordering) {
        unsafe { atomic_store(self.v.get(), val, order); }
    }

    /// Store a value, returning the old value
    #[inline]
    pub fn swap(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_swap(self.v.get(), val, order) }
    }

    /// If the current value is the same as expected, store a new value
    ///
    /// Compare the current value with `old`; if they are the same then
    /// replace the current value with `new`. Return the previous value.
    /// If the return value is equal to `old` then the value was updated.
    #[inline]
    pub fn compare_and_swap(&self, old: uint, new: uint, order: Ordering) -> uint {
        unsafe { atomic_compare_and_swap(self.v.get(), old, new, order) }
    }

    /// Add to the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0);
    /// assert_eq!(0, foo.fetch_add(10, SeqCst));
    /// assert_eq!(10, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_add(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_add(self.v.get(), val, order) }
    }

    /// Subtract from the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(10);
    /// assert_eq!(10, foo.fetch_sub(10, SeqCst));
    /// assert_eq!(0, foo.load(SeqCst));
    /// ```
    #[inline]
    pub fn fetch_sub(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_sub(self.v.get(), val, order) }
    }

    /// Bitwise and with the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0b101101);
    /// assert_eq!(0b101101, foo.fetch_and(0b110011, SeqCst));
    /// assert_eq!(0b100001, foo.load(SeqCst));
    #[inline]
    pub fn fetch_and(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_and(self.v.get(), val, order) }
    }

    /// Bitwise or with the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0b101101);
    /// assert_eq!(0b101101, foo.fetch_or(0b110011, SeqCst));
    /// assert_eq!(0b111111, foo.load(SeqCst));
    #[inline]
    pub fn fetch_or(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_or(self.v.get(), val, order) }
    }

    /// Bitwise xor with the current value, returning the previous
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomics::{AtomicUint, SeqCst};
    ///
    /// let foo = AtomicUint::new(0b101101);
    /// assert_eq!(0b101101, foo.fetch_xor(0b110011, SeqCst));
    /// assert_eq!(0b011110, foo.load(SeqCst));
    #[inline]
    pub fn fetch_xor(&self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_xor(self.v.get(), val, order) }
    }
}

impl<T> AtomicPtr<T> {
    /// Create a new `AtomicPtr`
    pub fn new(p: *mut T) -> AtomicPtr<T> {
        AtomicPtr { p: Unsafe::new(p as uint), nocopy: marker::NoCopy }
    }

    /// Load the value
    #[inline]
    pub fn load(&self, order: Ordering) -> *mut T {
        unsafe {
            atomic_load(self.p.get() as *const *mut T, order) as *mut T
        }
    }

    /// Store the value
    #[inline]
    pub fn store(&self, ptr: *mut T, order: Ordering) {
        unsafe { atomic_store(self.p.get(), ptr as uint, order); }
    }

    /// Store a value, returning the old value
    #[inline]
    pub fn swap(&self, ptr: *mut T, order: Ordering) -> *mut T {
        unsafe { atomic_swap(self.p.get(), ptr as uint, order) as *mut T }
    }

    /// If the current value is the same as expected, store a new value
    ///
    /// Compare the current value with `old`; if they are the same then
    /// replace the current value with `new`. Return the previous value.
    /// If the return value is equal to `old` then the value was updated.
    #[inline]
    pub fn compare_and_swap(&self, old: *mut T, new: *mut T, order: Ordering) -> *mut T {
        unsafe {
            atomic_compare_and_swap(self.p.get(), old as uint,
                                    new as uint, order) as *mut T
        }
    }
}

#[inline]
unsafe fn atomic_store<T>(dst: *mut T, val: T, order:Ordering) {
    match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        Relaxed => intrinsics::atomic_store_relaxed(dst, val),
        _       => intrinsics::atomic_store(dst, val)
    }
}

#[inline]
unsafe fn atomic_load<T>(dst: *const T, order:Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_load_acq(dst),
        Relaxed => intrinsics::atomic_load_relaxed(dst),
        _       => intrinsics::atomic_load(dst)
    }
}

#[inline]
unsafe fn atomic_swap<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        AcqRel  => intrinsics::atomic_xchg_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
        _       => intrinsics::atomic_xchg(dst, val)
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
        _       => intrinsics::atomic_xadd(dst, val)
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
        _       => intrinsics::atomic_xsub(dst, val)
    }
}

#[inline]
unsafe fn atomic_compare_and_swap<T>(dst: *mut T, old:T, new:T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_cxchg_acq(dst, old, new),
        Release => intrinsics::atomic_cxchg_rel(dst, old, new),
        AcqRel  => intrinsics::atomic_cxchg_acqrel(dst, old, new),
        Relaxed => intrinsics::atomic_cxchg_relaxed(dst, old, new),
        _       => intrinsics::atomic_cxchg(dst, old, new),
    }
}

#[inline]
unsafe fn atomic_and<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_and_acq(dst, val),
        Release => intrinsics::atomic_and_rel(dst, val),
        AcqRel  => intrinsics::atomic_and_acqrel(dst, val),
        Relaxed => intrinsics::atomic_and_relaxed(dst, val),
        _       => intrinsics::atomic_and(dst, val)
    }
}

#[inline]
unsafe fn atomic_nand<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_nand_acq(dst, val),
        Release => intrinsics::atomic_nand_rel(dst, val),
        AcqRel  => intrinsics::atomic_nand_acqrel(dst, val),
        Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
        _       => intrinsics::atomic_nand(dst, val)
    }
}


#[inline]
unsafe fn atomic_or<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_or_acq(dst, val),
        Release => intrinsics::atomic_or_rel(dst, val),
        AcqRel  => intrinsics::atomic_or_acqrel(dst, val),
        Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        _       => intrinsics::atomic_or(dst, val)
    }
}


#[inline]
unsafe fn atomic_xor<T>(dst: *mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xor_acq(dst, val),
        Release => intrinsics::atomic_xor_rel(dst, val),
        AcqRel  => intrinsics::atomic_xor_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        _       => intrinsics::atomic_xor(dst, val)
    }
}


/// An atomic fence.
///
/// A fence 'A' which has `Release` ordering semantics, synchronizes with a
/// fence 'B' with (at least) `Acquire` semantics, if and only if there exists
/// atomic operations X and Y, both operating on some atomic object 'M' such
/// that A is sequenced before X, Y is synchronized before B and Y observers
/// the change to M. This provides a happens-before dependence between A and B.
///
/// Atomic operations with `Release` or `Acquire` semantics can also synchronize
/// with a fence.
///
/// A fence with has `SeqCst` ordering, in addition to having both `Acquire` and
/// `Release` semantics, participates in the global program order of the other
/// `SeqCst` operations and/or fences.
///
/// Accepts `Acquire`, `Release`, `AcqRel` and `SeqCst` orderings.
///
/// # Failure
///
/// Fails if `order` is `Relaxed`
#[inline]
pub fn fence(order: Ordering) {
    unsafe {
        match order {
            Acquire => intrinsics::atomic_fence_acq(),
            Release => intrinsics::atomic_fence_rel(),
            AcqRel  => intrinsics::atomic_fence_acqrel(),
            SeqCst  => intrinsics::atomic_fence(),
            Relaxed => fail!("there is no such thing as a relaxed fence")
        }
    }
}
