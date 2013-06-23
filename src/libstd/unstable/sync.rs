// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast;
use libc;
use option::*;
use task;
use task::atomically;
use unstable::finally::Finally;
use unstable::intrinsics;
use ops::Drop;
use clone::Clone;
use kinds::Owned;

/// An atomically reference counted pointer.
///
/// Enforces no shared-memory safety.
pub struct UnsafeAtomicRcBox<T> {
    data: *mut libc::c_void,
}

struct AtomicRcBoxData<T> {
    count: int,
    data: Option<T>,
}

impl<T: Owned> UnsafeAtomicRcBox<T> {
    pub fn new(data: T) -> UnsafeAtomicRcBox<T> {
        unsafe {
            let data = ~AtomicRcBoxData { count: 1, data: Some(data) };
            let ptr = cast::transmute(data);
            return UnsafeAtomicRcBox { data: ptr };
        }
    }

    #[inline]
    pub unsafe fn get(&self) -> *mut T
    {
        let mut data: ~AtomicRcBoxData<T> = cast::transmute(self.data);
        assert!(data.count > 0);
        let r: *mut T = data.data.get_mut_ref();
        cast::forget(data);
        return r;
    }

    #[inline]
    pub unsafe fn get_immut(&self) -> *T
    {
        let mut data: ~AtomicRcBoxData<T> = cast::transmute(self.data);
        assert!(data.count > 0);
        let r: *T = cast::transmute_immut(data.data.get_mut_ref());
        cast::forget(data);
        return r;
    }
}

impl<T: Owned> Clone for UnsafeAtomicRcBox<T> {
    fn clone(&self) -> UnsafeAtomicRcBox<T> {
        unsafe {
            let mut data: ~AtomicRcBoxData<T> = cast::transmute(self.data);
            let new_count = intrinsics::atomic_xadd(&mut data.count, 1) + 1;
            assert!(new_count >= 2);
            cast::forget(data);
            return UnsafeAtomicRcBox { data: self.data };
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for UnsafeAtomicRcBox<T>{
    fn finalize(&self) {
        unsafe {
            do task::unkillable {
                let mut data: ~AtomicRcBoxData<T> = cast::transmute(self.data);
                let new_count = intrinsics::atomic_xsub(&mut data.count, 1) - 1;
                assert!(new_count >= 0);
                if new_count == 0 {
                    // drop glue takes over.
                } else {
                    cast::forget(data);
                }
            }
        }
    }
}


/****************************************************************************/

#[allow(non_camel_case_types)] // runtime type
pub type rust_little_lock = *libc::c_void;

struct LittleLock {
    l: rust_little_lock,
}

impl Drop for LittleLock {
    fn finalize(&self) {
        unsafe {
            rust_destroy_little_lock(self.l);
        }
    }
}

fn LittleLock() -> LittleLock {
    unsafe {
        LittleLock {
            l: rust_create_little_lock()
        }
    }
}

impl LittleLock {
    #[inline]
    pub unsafe fn lock<T>(&self, f: &fn() -> T) -> T {
        do atomically {
            rust_lock_little_lock(self.l);
            do (|| {
                f()
            }).finally {
                rust_unlock_little_lock(self.l);
            }
        }
    }
}

struct ExData<T> {
    lock: LittleLock,
    failed: bool,
    data: T,
}

/**
 * An arc over mutable data that is protected by a lock. For library use only.
 */
pub struct Exclusive<T> {
    x: UnsafeAtomicRcBox<ExData<T>>
}

pub fn exclusive<T:Owned>(user_data: T) -> Exclusive<T> {
    let data = ExData {
        lock: LittleLock(),
        failed: false,
        data: user_data
    };
    Exclusive {
        x: UnsafeAtomicRcBox::new(data)
    }
}

impl<T:Owned> Clone for Exclusive<T> {
    // Duplicate an exclusive ARC, as std::arc::clone.
    fn clone(&self) -> Exclusive<T> {
        Exclusive { x: self.x.clone() }
    }
}

impl<T:Owned> Exclusive<T> {
    // Exactly like std::arc::mutex_arc,access(), but with the little_lock
    // instead of a proper mutex. Same reason for being unsafe.
    //
    // Currently, scheduling operations (i.e., yielding, receiving on a pipe,
    // accessing the provided condition variable) are prohibited while inside
    // the exclusive. Supporting that is a work in progress.
    #[inline]
    pub unsafe fn with<U>(&self, f: &fn(x: &mut T) -> U) -> U {
        let rec = self.x.get();
        do (*rec).lock.lock {
            if (*rec).failed {
                fail!("Poisoned exclusive - another task failed inside!");
            }
            (*rec).failed = true;
            let result = f(&mut (*rec).data);
            (*rec).failed = false;
            result
        }
    }

    #[inline]
    pub unsafe fn with_imm<U>(&self, f: &fn(x: &T) -> U) -> U {
        do self.with |x| {
            f(cast::transmute_immut(x))
        }
    }
}

fn compare_and_swap(address: &mut int, oldval: int, newval: int) -> bool {
    unsafe {
        let old = intrinsics::atomic_cxchg(address, oldval, newval);
        old == oldval
    }
}

extern {
    fn rust_create_little_lock() -> rust_little_lock;
    fn rust_destroy_little_lock(lock: rust_little_lock);
    fn rust_lock_little_lock(lock: rust_little_lock);
    fn rust_unlock_little_lock(lock: rust_little_lock);
}

#[cfg(test)]
mod tests {
    use comm;
    use super::exclusive;
    use task;
    use uint;

    #[test]
    fn exclusive_arc() {
        unsafe {
            let mut futures = ~[];

            let num_tasks = 10;
            let count = 10;

            let total = exclusive(~0);

            for uint::range(0, num_tasks) |_i| {
                let total = total.clone();
                let (port, chan) = comm::stream();
                futures.push(port);

                do task::spawn || {
                    for uint::range(0, count) |_i| {
                        do total.with |count| {
                            **count += 1;
                        }
                    }
                    chan.send(());
                }
            };

            for futures.iter().advance |f| { f.recv() }

            do total.with |total| {
                assert!(**total == num_tasks * count)
            };
        }
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn exclusive_poison() {
        unsafe {
            // Tests that if one task fails inside of an exclusive, subsequent
            // accesses will also fail.
            let x = exclusive(1);
            let x2 = x.clone();
            do task::try || {
                do x2.with |one| {
                    assert_eq!(*one, 2);
                }
            };
            do x.with |one| {
                assert_eq!(*one, 1);
            }
        }
    }
}
