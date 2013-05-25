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
 */

use unstable::intrinsics;
use cast;
use option::{Option,Some,None};

pub struct AtomicFlag {
    priv v:int
}

pub struct AtomicBool {
    priv v:uint
}

pub struct AtomicInt {
    priv v:int
}

pub struct AtomicUint {
    priv v:uint
}

pub struct AtomicPtr<T> {
    priv p:~T
}

pub enum Ordering {
    Release,
    Acquire,
    SeqCst
}


impl AtomicFlag {

    fn new() -> AtomicFlag {
        AtomicFlag { v: 0 }
    }

    /**
     * Clears the atomic flag
     */
    #[inline(always)]
    fn clear(&mut self, order:Ordering) {
        unsafe {atomic_store(&mut self.v, 0, order)}
    }

    #[inline(always)]
    /**
     * Sets the flag if it was previously unset, returns the previous value of the
     * flag.
     */
    fn test_and_set(&mut self, order:Ordering) -> bool {
        unsafe {atomic_compare_and_swap(&mut self.v, 0, 1, order) > 0}
    }
}

impl AtomicBool {
    fn new(v:bool) -> AtomicBool {
        AtomicBool { v: if v { 1 } else { 0 } }
    }

    #[inline(always)]
    fn load(&self, order:Ordering) -> bool {
        unsafe { atomic_load(&self.v, order) > 0 }
    }

    #[inline(always)]
    fn store(&mut self, val:bool, order:Ordering) {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline(always)]
    fn swap(&mut self, val:bool, order:Ordering) -> bool {
        let val = if val { 1 } else { 0 };

        unsafe { atomic_swap(&mut self.v, val, order) > 0}
    }

    #[inline(always)]
    fn compare_and_swap(&mut self, old: bool, new: bool, order:Ordering) -> bool {
        let old = if old { 1 } else { 0 };
        let new = if new { 1 } else { 0 };

        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) > 0 }
    }
}

impl AtomicInt {
    fn new(v:int) -> AtomicInt {
        AtomicInt { v:v }
    }

    #[inline(always)]
    fn load(&self, order:Ordering) -> int {
        unsafe { atomic_load(&self.v, order) }
    }

    #[inline(always)]
    fn store(&mut self, val:int, order:Ordering) {
        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline(always)]
    fn swap(&mut self, val:int, order:Ordering) -> int {
        unsafe { atomic_swap(&mut self.v, val, order) }
    }

    #[inline(always)]
    fn compare_and_swap(&mut self, old: int, new: int, order:Ordering) -> int {
        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) }
    }

    #[inline(always)]
    fn fetch_add(&mut self, val:int, order:Ordering) -> int {
        unsafe { atomic_add(&mut self.v, val, order) }
    }

    #[inline(always)]
    fn fetch_sub(&mut self, val:int, order:Ordering) -> int {
        unsafe { atomic_sub(&mut self.v, val, order) }
    }
}

impl AtomicUint {
    fn new(v:uint) -> AtomicUint {
        AtomicUint { v:v }
    }

    #[inline(always)]
    fn load(&self, order:Ordering) -> uint {
        unsafe { atomic_load(&self.v, order) }
    }

    #[inline(always)]
    fn store(&mut self, val:uint, order:Ordering) {
        unsafe { atomic_store(&mut self.v, val, order); }
    }

    #[inline(always)]
    fn swap(&mut self, val:uint, order:Ordering) -> uint {
        unsafe { atomic_swap(&mut self.v, val, order) }
    }

    #[inline(always)]
    fn compare_and_swap(&mut self, old: uint, new: uint, order:Ordering) -> uint {
        unsafe { atomic_compare_and_swap(&mut self.v, old, new, order) }
    }

    #[inline(always)]
    fn fetch_add(&mut self, val:uint, order:Ordering) -> uint {
        unsafe { atomic_add(&mut self.v, val, order) }
    }

    #[inline(always)]
    fn fetch_sub(&mut self, val:uint, order:Ordering) -> uint {
        unsafe { atomic_sub(&mut self.v, val, order) }
    }
}

impl<T> AtomicPtr<T> {
    fn new(p:~T) -> AtomicPtr<T> {
        AtomicPtr { p:p }
    }

    /**
     * Atomically swaps the stored pointer with the one given.
     *
     * Returns None if the pointer stored has been taken
     */
    #[inline(always)]
    fn swap(&mut self, ptr:~T, order:Ordering) -> Option<~T> {
        unsafe {
            let p = atomic_swap(&mut self.p, ptr, order);
            let pv : &uint = cast::transmute(&p);

            if *pv == 0 {
                None
            } else {
                Some(p)
            }
        }
    }

    /**
     * Atomically takes the stored pointer out.
     *
     * Returns None if it was already taken.
     */
    #[inline(always)]
    fn take(&mut self, order:Ordering) -> Option<~T> {
        unsafe { self.swap(cast::transmute(0), order) }
    }

    /**
     * Atomically stores the given pointer, this will overwrite
     * and previous value stored.
     */
    #[inline(always)]
    fn give(&mut self, ptr:~T, order:Ordering) {
        let _ = self.swap(ptr, order);
    }

    /**
     * Checks to see if the stored pointer has been taken.
     */
    fn taken(&self, order:Ordering) -> bool {
        unsafe {
            let p : ~T = atomic_load(&self.p, order);

            let pv : &uint = cast::transmute(&p);

            cast::forget(p);
            *pv == 0
        }
    }
}

#[inline(always)]
pub unsafe fn atomic_store<T>(dst: &mut T, val: T, order:Ordering) {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);

    match order {
        Release => intrinsics::atomic_store_rel(dst, val),
        _       => intrinsics::atomic_store(dst, val)
    }
}

#[inline(always)]
pub unsafe fn atomic_load<T>(dst: &T, order:Ordering) -> T {
    let dst = cast::transmute(dst);

    cast::transmute(match order {
        Acquire => intrinsics::atomic_load_acq(dst),
        _       => intrinsics::atomic_load(dst)
    })
}

#[inline(always)]
pub unsafe fn atomic_swap<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);

    cast::transmute(match order {
        Acquire => intrinsics::atomic_xchg_acq(dst, val),
        Release => intrinsics::atomic_xchg_rel(dst, val),
        _       => intrinsics::atomic_xchg(dst, val)
    })
}

#[inline(always)]
pub unsafe fn atomic_add<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);

    cast::transmute(match order {
        Acquire => intrinsics::atomic_xadd_acq(dst, val),
        Release => intrinsics::atomic_xadd_rel(dst, val),
        _       => intrinsics::atomic_xadd(dst, val)
    })
}

#[inline(always)]
pub unsafe fn atomic_sub<T>(dst: &mut T, val: T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let val = cast::transmute(val);

    cast::transmute(match order {
        Acquire => intrinsics::atomic_xsub_acq(dst, val),
        Release => intrinsics::atomic_xsub_rel(dst, val),
        _       => intrinsics::atomic_xsub(dst, val)
    })
}

#[inline(always)]
pub unsafe fn atomic_compare_and_swap<T>(dst:&mut T, old:T, new:T, order: Ordering) -> T {
    let dst = cast::transmute(dst);
    let old = cast::transmute(old);
    let new = cast::transmute(new);

    cast::transmute(match order {
        Acquire => intrinsics::atomic_cxchg_acq(dst, old, new),
        Release => intrinsics::atomic_cxchg_rel(dst, old, new),
        _       => intrinsics::atomic_cxchg(dst, old, new),
    })
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
    fn pointer_swap() {
        let mut p = AtomicPtr::new(~1);
        let a = ~2;

        let b = p.swap(a, SeqCst);

        assert_eq!(b, Some(~1));
        assert_eq!(p.take(SeqCst), Some(~2));
    }

    #[test]
    fn pointer_take() {
        let mut p = AtomicPtr::new(~1);

        assert_eq!(p.take(SeqCst), Some(~1));
        assert_eq!(p.take(SeqCst), None);
        assert!(p.taken(SeqCst));

        let p2 = ~2;
        p.give(p2, SeqCst);

        assert_eq!(p.take(SeqCst), Some(~2));
    }

}
