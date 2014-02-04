// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Atomic types
 *
 * Basic atomic types supporting atomic operations. Each method takes an
 * `Ordering` which represents the strength of the memory barrier for that
 * operation. These orderings are the same as C++11 atomic orderings
 * [http://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync]
 *
 * All atomic types are a single word in size.
 */

#[allow(missing_doc)];

use unstable::intrinsics;
use cast;
use std::kinds::marker;
use option::{Option,Some,None};
use ops::Drop;

/**
 * A simple atomic flag, that can be set and cleared. The most basic atomic type.
 */
pub struct AtomicFlag {
    priv v: int,
    priv nopod: marker::NoPod
}

/**
 * An atomic boolean type.
 */
pub struct AtomicBool {
    priv v: uint,
    priv nopod: marker::NoPod
}

/**
 * A signed atomic integer type, supporting basic atomic arithmetic operations
 */
pub struct AtomicInt {
    priv v: int,
    priv nopod: marker::NoPod
}

/**
 * An unsigned atomic integer type, supporting basic atomic arithmetic operations
 */
pub struct AtomicUint {
    priv v: uint,
    priv nopod: marker::NoPod
}

/**
 * An unsigned atomic integer type that is forced to be 64-bits. This does not
 * support all operations.
 */
#[cfg(not(stage0))]
pub struct AtomicU64 {
    priv v: u64,
    priv nopod: marker::NoPod
}

/**
 * An unsafe atomic pointer. Only supports basic atomic operations
 */
#[cfg(not(stage0))]
pub struct AtomicPtr<T> {
    priv p: uint,
    priv nopod: marker::NoPod
}
#[cfg(stage0)]
pub struct AtomicPtr<T> {
    priv p: *mut T,
    priv nopod: marker::NoPod
}

/**
 * An owned atomic pointer. Ensures that only a single reference to the data is held at any time.
 */
#[unsafe_no_drop_flag]
#[cfg(not(stage0))]
pub struct AtomicOption<T> {
    priv p: uint,
}
#[unsafe_no_drop_flag]
#[cfg(stage0)]
pub struct AtomicOption<T> {
    priv p: *mut u8
}

pub enum Ordering {
    Relaxed,
    Release,
    Acquire,
    AcqRel,
    SeqCst
}

pub static INIT_ATOMIC_FLAG : AtomicFlag = AtomicFlag { v: 0, nopod: marker::NoPod };
pub static INIT_ATOMIC_BOOL : AtomicBool = AtomicBool { v: 0, nopod: marker::NoPod };
pub static INIT_ATOMIC_INT  : AtomicInt  = AtomicInt  { v: 0, nopod: marker::NoPod };
pub static INIT_ATOMIC_UINT : AtomicUint = AtomicUint { v: 0, nopod: marker::NoPod };
#[cfg(not(stage0))]
pub static INIT_ATOMIC_U64 : AtomicU64 = AtomicU64 { v: 0, nopod: marker::NoPod };

impl AtomicFlag {

    pub fn new() -> AtomicFlag {
        AtomicFlag { v: 0, nopod: marker::NoPod}
    }

    /**
     * Clears the atomic flag
     */
    #[inline]
    pub fn clear(&mut self, order: Ordering) {
        unsafe {atomic_store(&mut self.v, 0, order)}
    }

    /**
     * Sets the flag if it was previously unset, returns the previous value of the
     * flag.
     */
    #[inline]
    pub fn test_and_set(&mut self, order: Ordering) -> bool {
        unsafe { atomic_compare_and_swap(&mut self.v, 0, 1, order) > 0 }
    }
}

impl AtomicBool {
    pub fn new(v: bool) -> AtomicBool {
        AtomicBool { v: if v { 1 } else { 0 }, nopod: marker::NoPod }
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> bool {
        unsafe { atomic_load(&self.v, order) > 0 }
    }

    #[inline]
    pub fn store(&mut self, val: bool, order: Ordering) {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline]
    pub fn swap(&mut self, val: bool, order: Ordering) -> bool {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_swap(&mut self.v, val, order) > 0 }
    }

    #[inline]
    pub fn compare_and_swap(&mut self, old: bool, new: bool, order: Ordering) -> bool {
        let old = if old { 1 } else { 0 };
        let new = if new { 1 } else { 0 };

        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) > 0 }
    }

    /// Returns the old value
    #[inline]
    pub fn fetch_and(&mut self, val: bool, order: Ordering) -> bool {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_and(&mut self.v, val, order) > 0 }
    }

    /// Returns the old value
    #[inline]
    pub fn fetch_nand(&mut self, val: bool, order: Ordering) -> bool {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_nand(&mut self.v, val, order) > 0 }
    }

    /// Returns the old value
    #[inline]
    pub fn fetch_or(&mut self, val: bool, order: Ordering) -> bool {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_or(&mut self.v, val, order) > 0 }
    }

    /// Returns the old value
    #[inline]
    pub fn fetch_xor(&mut self, val: bool, order: Ordering) -> bool {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_xor(&mut self.v, val, order) > 0 }
    }
}

impl AtomicInt {
    pub fn new(v: int) -> AtomicInt {
        AtomicInt { v:v, nopod: marker::NoPod}
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> int {
        unsafe { atomic_load(&self.v, order) }
    }

    #[inline]
    pub fn store(&mut self, val: int, order: Ordering) {
        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline]
    pub fn swap(&mut self, val: int, order: Ordering) -> int {
        unsafe { atomic_swap(&mut self.v, val, order) }
    }

    #[inline]
    pub fn compare_and_swap(&mut self, old: int, new: int, order: Ordering) -> int {
        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) }
    }

    /// Returns the old value (like __sync_fetch_and_add).
    #[inline]
    pub fn fetch_add(&mut self, val: int, order: Ordering) -> int {
        unsafe { atomic_add(&mut self.v, val, order) }
    }

    /// Returns the old value (like __sync_fetch_and_sub).
    #[inline]
    pub fn fetch_sub(&mut self, val: int, order: Ordering) -> int {
        unsafe { atomic_sub(&mut self.v, val, order) }
    }
}

#[cfg(not(stage0))]
impl AtomicU64 {
    pub fn new(v: u64) -> AtomicU64 {
        AtomicU64 { v:v, nopod: marker::NoPod }
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> u64 {
        unsafe { atomic_load(&self.v, order) }
    }

    #[inline]
    pub fn store(&mut self, val: u64, order: Ordering) {
        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline]
    pub fn swap(&mut self, val: u64, order: Ordering) -> u64 {
        unsafe { atomic_swap(&mut self.v, val, order) }
    }

    #[inline]
    pub fn compare_and_swap(&mut self, old: u64, new: u64, order: Ordering) -> u64 {
        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) }
    }

    #[inline]
    pub fn fetch_add(&mut self, val: u64, order: Ordering) -> u64 {
        unsafe { atomic_add(&mut self.v, val, order) }
    }

    #[inline]
    pub fn fetch_sub(&mut self, val: u64, order: Ordering) -> u64 {
        unsafe { atomic_sub(&mut self.v, val, order) }
    }
}

impl AtomicUint {
    pub fn new(v: uint) -> AtomicUint {
        AtomicUint { v:v, nopod: marker::NoPod }
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> uint {
        unsafe { atomic_load(&self.v, order) }
    }

    #[inline]
    pub fn store(&mut self, val: uint, order: Ordering) {
        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline]
    pub fn swap(&mut self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_swap(&mut self.v, val, order) }
    }

    #[inline]
    pub fn compare_and_swap(&mut self, old: uint, new: uint, order: Ordering) -> uint {
        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) }
    }

    /// Returns the old value (like __sync_fetch_and_add).
    #[inline]
    pub fn fetch_add(&mut self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_add(&mut self.v, val, order) }
    }

    /// Returns the old value (like __sync_fetch_and_sub)..
    #[inline]
    pub fn fetch_sub(&mut self, val: uint, order: Ordering) -> uint {
        unsafe { atomic_sub(&mut self.v, val, order) }
    }
}

impl<T> AtomicPtr<T> {
    #[cfg(stage0)]
    pub fn new(p: *mut T) -> AtomicPtr<T> {
        AtomicPtr { p: p, nopod: marker::NoPod }
    }
    #[cfg(not(stage0))]
    pub fn new(p: *mut T) -> AtomicPtr<T> {
        AtomicPtr { p: p as uint, nopod: marker::NoPod }
    }

    #[inline]
    #[cfg(not(stage0))]
    pub fn load(&self, order: Ordering) -> *mut T {
        unsafe {
            atomic_load(&self.p, order) as *mut T
        }
    }

    #[inline]
    #[cfg(not(stage0))]
    pub fn store(&mut self, ptr: *mut T, order: Ordering) {
        unsafe { atomic_store(&mut self.p, ptr as uint, order); }
    }

    #[inline]
    #[cfg(not(stage0))]
    pub fn swap(&mut self, ptr: *mut T, order: Ordering) -> *mut T {
        unsafe { atomic_swap(&mut self.p, ptr as uint, order) as *mut T }
    }

    #[inline]
    #[cfg(not(stage0))]
    pub fn compare_and_swap(&mut self, old: *mut T, new: *mut T, order: Ordering) -> *mut T {
        unsafe {
            atomic_compare_and_swap(&mut self.p, old as uint,
                                    new as uint, order) as *mut T
        }
    }

    #[inline]
    #[cfg(stage0)]
    pub fn load(&self, order: Ordering) -> *mut T {
        unsafe { atomic_load(&self.p, order) }
    }

    #[inline]
    #[cfg(stage0)]
    pub fn store(&mut self, ptr: *mut T, order: Ordering) {
        unsafe { atomic_store(&mut self.p, ptr, order); }
    }

    #[inline]
    #[cfg(stage0)]
    pub fn swap(&mut self, ptr: *mut T, order: Ordering) -> *mut T {
        unsafe { atomic_swap(&mut self.p, ptr, order) }
    }

    #[inline]
    #[cfg(stage0)]
    pub fn compare_and_swap(&mut self, old: *mut T, new: *mut T, order: Ordering) -> *mut T {
        unsafe { atomic_compare_and_swap(&mut self.p, old, new, order) }
    }
}

impl<T> AtomicOption<T> {
    pub fn new(p: ~T) -> AtomicOption<T> {
        unsafe { AtomicOption { p: cast::transmute(p) } }
    }

    #[cfg(stage0)]
    pub fn empty() -> AtomicOption<T> { AtomicOption { p: 0 as *mut u8 } }
    #[cfg(not(stage0))]
    pub fn empty() -> AtomicOption<T> { AtomicOption { p: 0 } }

    #[inline]
    pub fn swap(&mut self, val: ~T, order: Ordering) -> Option<~T> {
        unsafe {
            let val = cast::transmute(val);

            let p = atomic_swap(&mut self.p, val, order);
            if p as uint == 0 {
                None
            } else {
                Some(cast::transmute(p))
            }
        }
    }

    #[inline]
    pub fn take(&mut self, order: Ordering) -> Option<~T> {
        unsafe { self.swap(cast::transmute(0), order) }
    }

    /// A compare-and-swap. Succeeds if the option is 'None' and returns 'None'
    /// if so. If the option was already 'Some', returns 'Some' of the rejected
    /// value.
    #[inline]
    pub fn fill(&mut self, val: ~T, order: Ordering) -> Option<~T> {
        unsafe {
            let val = cast::transmute(val);
            let expected = cast::transmute(0);
            let oldval = atomic_compare_and_swap(&mut self.p, expected, val, order);
            if oldval == expected {
                None
            } else {
                Some(cast::transmute(val))
            }
        }
    }

    /// Be careful: The caller must have some external method of ensuring the
    /// result does not get invalidated by another task after this returns.
    #[inline]
    pub fn is_empty(&mut self, order: Ordering) -> bool {
        unsafe { atomic_load(&self.p, order) as uint == 0 }
    }
}

#[unsafe_destructor]
impl<T> Drop for AtomicOption<T> {
    fn drop(&mut self) {
        let _ = self.take(SeqCst);
    }
}

#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_store<T>(dst: &mut T, val: T, order:Ordering) {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        Relaxed => intrinsics::atomic_store_relaxed(dst, val),
        _       => intrinsics::atomic_store(dst, val)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_store<T>(dst: &mut T, val: T, order:Ordering) {
    match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        Relaxed => intrinsics::atomic_store_relaxed(dst, val),
        _       => intrinsics::atomic_store(dst, val)
    }
}

#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_load<T>(dst: &T, order:Ordering) -> T {
    let dst = cast::transmute(dst);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_load_acq(dst),
        Relaxed => intrinsics::atomic_load_relaxed(dst),
        _       => intrinsics::atomic_load(dst)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_load<T>(dst: &T, order:Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_load_acq(dst),
        Relaxed => intrinsics::atomic_load_relaxed(dst),
        _       => intrinsics::atomic_load(dst)
    }
}

#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_swap<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        AcqRel  => intrinsics::atomic_xchg_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
        _       => intrinsics::atomic_xchg(dst, val)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_swap<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        AcqRel  => intrinsics::atomic_xchg_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xchg_relaxed(dst, val),
        _       => intrinsics::atomic_xchg(dst, val)
    }
}

/// Returns the old value (like __sync_fetch_and_add).
#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_add<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_xadd_acq(dst, val),
        Release => intrinsics::atomic_xadd_rel(dst, val),
        AcqRel  => intrinsics::atomic_xadd_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
        _       => intrinsics::atomic_xadd(dst, val)
    })
}
/// Returns the old value (like __sync_fetch_and_add).
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_add<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xadd_acq(dst, val),
        Release => intrinsics::atomic_xadd_rel(dst, val),
        AcqRel  => intrinsics::atomic_xadd_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xadd_relaxed(dst, val),
        _       => intrinsics::atomic_xadd(dst, val)
    }
}

/// Returns the old value (like __sync_fetch_and_sub).
#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_sub<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_xsub_acq(dst, val),
        Release => intrinsics::atomic_xsub_rel(dst, val),
        AcqRel  => intrinsics::atomic_xsub_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
        _       => intrinsics::atomic_xsub(dst, val)
    })
}
/// Returns the old value (like __sync_fetch_and_sub).
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_sub<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xsub_acq(dst, val),
        Release => intrinsics::atomic_xsub_rel(dst, val),
        AcqRel  => intrinsics::atomic_xsub_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xsub_relaxed(dst, val),
        _       => intrinsics::atomic_xsub(dst, val)
    }
}

#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_compare_and_swap<T>(dst:&mut T, old:T, new:T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let new = cast::transmute(new);
    let old = cast::transmute(old);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_cxchg_acq(dst, old, new),
        Release => intrinsics::atomic_cxchg_rel(dst, old, new),
        AcqRel  => intrinsics::atomic_cxchg_acqrel(dst, old, new),
        Relaxed => intrinsics::atomic_cxchg_relaxed(dst, old, new),
        _       => intrinsics::atomic_cxchg(dst, old, new),
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_compare_and_swap<T>(dst:&mut T, old:T, new:T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_cxchg_acq(dst, old, new),
        Release => intrinsics::atomic_cxchg_rel(dst, old, new),
        AcqRel  => intrinsics::atomic_cxchg_acqrel(dst, old, new),
        Relaxed => intrinsics::atomic_cxchg_relaxed(dst, old, new),
        _       => intrinsics::atomic_cxchg(dst, old, new),
    }
}

#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_and<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_and_acq(dst, val),
        Release => intrinsics::atomic_and_rel(dst, val),
        AcqRel  => intrinsics::atomic_and_acqrel(dst, val),
        Relaxed => intrinsics::atomic_and_relaxed(dst, val),
        _       => intrinsics::atomic_and(dst, val)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_and<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_and_acq(dst, val),
        Release => intrinsics::atomic_and_rel(dst, val),
        AcqRel  => intrinsics::atomic_and_acqrel(dst, val),
        Relaxed => intrinsics::atomic_and_relaxed(dst, val),
        _       => intrinsics::atomic_and(dst, val)
    }
}

#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_nand<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_nand_acq(dst, val),
        Release => intrinsics::atomic_nand_rel(dst, val),
        AcqRel  => intrinsics::atomic_nand_acqrel(dst, val),
        Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
        _       => intrinsics::atomic_nand(dst, val)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_nand<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_nand_acq(dst, val),
        Release => intrinsics::atomic_nand_rel(dst, val),
        AcqRel  => intrinsics::atomic_nand_acqrel(dst, val),
        Relaxed => intrinsics::atomic_nand_relaxed(dst, val),
        _       => intrinsics::atomic_nand(dst, val)
    }
}


#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_or<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_or_acq(dst, val),
        Release => intrinsics::atomic_or_rel(dst, val),
        AcqRel  => intrinsics::atomic_or_acqrel(dst, val),
        Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        _       => intrinsics::atomic_or(dst, val)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_or<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_or_acq(dst, val),
        Release => intrinsics::atomic_or_rel(dst, val),
        AcqRel  => intrinsics::atomic_or_acqrel(dst, val),
        Relaxed => intrinsics::atomic_or_relaxed(dst, val),
        _       => intrinsics::atomic_or(dst, val)
    }
}


#[cfg(stage0)]
#[inline]
pub unsafe fn atomic_xor<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);
    cast::transmute(match order {
        Acquire => intrinsics::atomic_xor_acq(dst, val),
        Release => intrinsics::atomic_xor_rel(dst, val),
        AcqRel  => intrinsics::atomic_xor_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        _       => intrinsics::atomic_xor(dst, val)
    })
}
#[cfg(not(stage0))]
#[inline]
pub unsafe fn atomic_xor<T>(dst: &mut T, val: T, order: Ordering) -> T {
    match order {
        Acquire => intrinsics::atomic_xor_acq(dst, val),
        Release => intrinsics::atomic_xor_rel(dst, val),
        AcqRel  => intrinsics::atomic_xor_acqrel(dst, val),
        Relaxed => intrinsics::atomic_xor_relaxed(dst, val),
        _       => intrinsics::atomic_xor(dst, val)
    }
}


/**
 * An atomic fence.
 *
 * A fence 'A' which has `Release` ordering semantics, synchronizes with a
 * fence 'B' with (at least) `Acquire` semantics, if and only if there exists
 * atomic operations X and Y, both operating on some atomic object 'M' such
 * that A is sequenced before X, Y is synchronized before B and Y observers
 * the change to M. This provides a happens-before dependence between A and B.
 *
 * Atomic operations with `Release` or `Acquire` semantics can also synchronize
 * with a fence.
 *
 * A fence with has `SeqCst` ordering, in addition to having both `Acquire` and
 * `Release` semantics, participates in the global program order of the other
 * `SeqCst` operations and/or fences.
 *
 * Accepts `Acquire`, `Release`, `AcqRel` and `SeqCst` orderings.
 */
#[inline]
pub fn fence(order: Ordering) {
    unsafe {
        match order {
            Acquire => intrinsics::atomic_fence_acq(),
            Release => intrinsics::atomic_fence_rel(),
            AcqRel  => intrinsics::atomic_fence_rel(),
            _       => intrinsics::atomic_fence(),
        }
    }
}

#[cfg(test)]
mod test {
    use option::*;
    use super::*;

    #[test]
    fn flag() {
        let mut flg = AtomicFlag::new();
        assert!(!flg.test_and_set(SeqCst));
        assert!(flg.test_and_set(SeqCst));

        flg.clear(SeqCst);
        assert!(!flg.test_and_set(SeqCst));
    }

    #[test]
    fn option_empty() {
        let mut option: AtomicOption<()> = AtomicOption::empty();
        assert!(option.is_empty(SeqCst));
    }

    #[test]
    fn option_swap() {
        let mut p = AtomicOption::new(~1);
        let a = ~2;

        let b = p.swap(a, SeqCst);

        assert_eq!(b, Some(~1));
        assert_eq!(p.take(SeqCst), Some(~2));
    }

    #[test]
    fn option_take() {
        let mut p = AtomicOption::new(~1);

        assert_eq!(p.take(SeqCst), Some(~1));
        assert_eq!(p.take(SeqCst), None);

        let p2 = ~2;
        p.swap(p2, SeqCst);

        assert_eq!(p.take(SeqCst), Some(~2));
    }

    #[test]
    fn option_fill() {
        let mut p = AtomicOption::new(~1);
        assert!(p.fill(~2, SeqCst).is_some()); // should fail; shouldn't leak!
        assert_eq!(p.take(SeqCst), Some(~1));

        assert!(p.fill(~2, SeqCst).is_none()); // shouldn't fail
        assert_eq!(p.take(SeqCst), Some(~2));
    }

    #[test]
    fn bool_and() {
        let mut a = AtomicBool::new(true);
        assert_eq!(a.fetch_and(false, SeqCst),true);
        assert_eq!(a.load(SeqCst),false);
    }

    static mut S_FLAG : AtomicFlag = INIT_ATOMIC_FLAG;
    static mut S_BOOL : AtomicBool = INIT_ATOMIC_BOOL;
    static mut S_INT  : AtomicInt  = INIT_ATOMIC_INT;
    static mut S_UINT : AtomicUint = INIT_ATOMIC_UINT;

    #[test]
    fn static_init() {
        unsafe {
            assert!(!S_FLAG.test_and_set(SeqCst));
            assert!(!S_BOOL.load(SeqCst));
            assert!(S_INT.load(SeqCst) == 0);
            assert!(S_UINT.load(SeqCst) == 0);
        }
    }

    #[test]
    #[cfg(not(stage0))]
    fn different_sizes() {
        unsafe {
            let mut slot = 0u16;
            assert_eq!(super::atomic_swap(&mut slot, 1, SeqCst), 0);

            let mut slot = 0u8;
            assert_eq!(super::atomic_compare_and_swap(&mut slot, 1, 2, SeqCst), 0);

            let mut slot = 0u32;
            assert_eq!(super::atomic_load(&mut slot, SeqCst), 0);

            let mut slot = 0u64;
            super::atomic_store(&mut slot, 2, SeqCst);
        }
    }
}
