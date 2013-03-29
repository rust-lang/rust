// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];

use cast;
use libc;
use comm::{GenericChan, GenericPort};
use prelude::*;
use task;
use task::atomically;

#[path = "unstable/at_exit.rs"]
pub mod at_exit;
#[path = "unstable/global.rs"]
pub mod global;
#[path = "unstable/finally.rs"]
pub mod finally;
#[path = "unstable/weak_task.rs"]
pub mod weak_task;
#[path = "unstable/exchange_alloc.rs"]
pub mod exchange_alloc;
#[path = "unstable/intrinsics.rs"]
pub mod intrinsics;
#[path = "unstable/extfmt.rs"]
pub mod extfmt;
#[path = "unstable/lang.rs"]
#[cfg(notest)]
pub mod lang;

mod rustrt {
    use unstable::{raw_thread, rust_little_lock};

    pub extern {
        pub unsafe fn rust_create_little_lock() -> rust_little_lock;
        pub unsafe fn rust_destroy_little_lock(lock: rust_little_lock);
        pub unsafe fn rust_lock_little_lock(lock: rust_little_lock);
        pub unsafe fn rust_unlock_little_lock(lock: rust_little_lock);

        pub unsafe fn rust_raw_thread_start(f: &(&fn())) -> *raw_thread;
        pub unsafe fn rust_raw_thread_join_delete(thread: *raw_thread);
    }
}

#[allow(non_camel_case_types)] // runtime type
pub type raw_thread = libc::c_void;

/**

Start a new thread outside of the current runtime context and wait
for it to terminate.

The executing thread has no access to a task pointer and will be using
a normal large stack.
*/
pub fn run_in_bare_thread(f: ~fn()) {
    let (port, chan) = comm::stream();
    // FIXME #4525: Unfortunate that this creates an extra scheduler but it's
    // necessary since rust_raw_thread_join_delete is blocking
    do task::spawn_sched(task::SingleThreaded) {
        unsafe {
            let closure: &fn() = || {
                f()
            };
            let thread = rustrt::rust_raw_thread_start(&closure);
            rustrt::rust_raw_thread_join_delete(thread);
            chan.send(());
        }
    }
    port.recv();
}

#[test]
fn test_run_in_bare_thread() {
    let i = 100;
    do run_in_bare_thread {
        assert!(i == 100);
    }
}

#[test]
fn test_run_in_bare_thread_exchange() {
    // Does the exchange heap work without the runtime?
    let i = ~100;
    do run_in_bare_thread {
        assert!(i == ~100);
    }
}

fn compare_and_swap(address: &mut int, oldval: int, newval: int) -> bool {
    unsafe {
        let old = intrinsics::atomic_cxchg(address, oldval, newval);
        old == oldval
    }
}

/****************************************************************************
 * Shared state & exclusive ARC
 ****************************************************************************/

struct ArcData<T> {
    mut count:     libc::intptr_t,
    // FIXME(#3224) should be able to make this non-option to save memory
    mut data:      Option<T>,
}

struct ArcDestruct<T> {
    mut data: *libc::c_void,
}

#[unsafe_destructor]
impl<T> Drop for ArcDestruct<T>{
    fn finalize(&self) {
        unsafe {
            do task::unkillable {
                let data: ~ArcData<T> = cast::reinterpret_cast(&self.data);
                let new_count =
                    intrinsics::atomic_xsub(&mut data.count, 1) - 1;
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

fn ArcDestruct<T>(data: *libc::c_void) -> ArcDestruct<T> {
    ArcDestruct {
        data: data
    }
}

/**
 * COMPLETELY UNSAFE. Used as a primitive for the safe versions in std::arc.
 *
 * Data races between tasks can result in crashes and, with sufficient
 * cleverness, arbitrary type coercion.
 */
pub type SharedMutableState<T> = ArcDestruct<T>;

pub unsafe fn shared_mutable_state<T:Owned>(data: T) ->
        SharedMutableState<T> {
    let data = ~ArcData { count: 1, data: Some(data) };
    unsafe {
        let ptr = cast::transmute(data);
        ArcDestruct(ptr)
    }
}

#[inline(always)]
pub unsafe fn get_shared_mutable_state<T:Owned>(
    rc: *SharedMutableState<T>) -> *mut T
{
    unsafe {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&(*rc).data);
        assert!(ptr.count > 0);
        let r = cast::transmute(ptr.data.get_ref());
        cast::forget(ptr);
        return r;
    }
}
#[inline(always)]
pub unsafe fn get_shared_immutable_state<'a,T:Owned>(
        rc: &'a SharedMutableState<T>) -> &'a T {
    unsafe {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&(*rc).data);
        assert!(ptr.count > 0);
        // Cast us back into the correct region
        let r = cast::transmute_region(ptr.data.get_ref());
        cast::forget(ptr);
        return r;
    }
}

pub unsafe fn clone_shared_mutable_state<T:Owned>(rc: &SharedMutableState<T>)
        -> SharedMutableState<T> {
    unsafe {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&(*rc).data);
        let new_count = intrinsics::atomic_xadd(&mut ptr.count, 1) + 1;
        assert!(new_count >= 2);
        cast::forget(ptr);
    }
    ArcDestruct((*rc).data)
}

impl<T:Owned> Clone for SharedMutableState<T> {
    fn clone(&self) -> SharedMutableState<T> {
        unsafe {
            clone_shared_mutable_state(self)
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
            rustrt::rust_destroy_little_lock(self.l);
        }
    }
}

fn LittleLock() -> LittleLock {
    unsafe {
        LittleLock {
            l: rustrt::rust_create_little_lock()
        }
    }
}

pub impl LittleLock {
    #[inline(always)]
    unsafe fn lock<T>(&self, f: &fn() -> T) -> T {
        struct Unlock {
            l: rust_little_lock,
            drop {
                unsafe {
                    rustrt::rust_unlock_little_lock(self.l);
                }
            }
        }

        fn Unlock(l: rust_little_lock) -> Unlock {
            Unlock {
                l: l
            }
        }

        do atomically {
            rustrt::rust_lock_little_lock(self.l);
            let _r = Unlock(self.l);
            f()
        }
    }
}

struct ExData<T> { lock: LittleLock, mut failed: bool, mut data: T, }
/**
 * An arc over mutable data that is protected by a lock. For library use only.
 */
pub struct Exclusive<T> { x: SharedMutableState<ExData<T>> }

pub fn exclusive<T:Owned>(user_data: T) -> Exclusive<T> {
    let data = ExData {
        lock: LittleLock(), mut failed: false, mut data: user_data
    };
    Exclusive { x: unsafe { shared_mutable_state(data) } }
}

impl<T:Owned> Clone for Exclusive<T> {
    // Duplicate an exclusive ARC, as std::arc::clone.
    fn clone(&self) -> Exclusive<T> {
        Exclusive { x: unsafe { clone_shared_mutable_state(&self.x) } }
    }
}

pub impl<T:Owned> Exclusive<T> {
    // Exactly like std::arc::mutex_arc,access(), but with the little_lock
    // instead of a proper mutex. Same reason for being unsafe.
    //
    // Currently, scheduling operations (i.e., yielding, receiving on a pipe,
    // accessing the provided condition variable) are prohibited while inside
    // the exclusive. Supporting that is a work in progress.
    #[inline(always)]
    unsafe fn with<U>(&self, f: &fn(x: &mut T) -> U) -> U {
        unsafe {
            let rec = get_shared_mutable_state(&self.x);
            do (*rec).lock.lock {
                if (*rec).failed {
                    fail!(
                        ~"Poisoned exclusive - another task failed inside!");
                }
                (*rec).failed = true;
                let result = f(&mut (*rec).data);
                (*rec).failed = false;
                result
            }
        }
    }

    #[inline(always)]
    unsafe fn with_imm<U>(&self, f: &fn(x: &T) -> U) -> U {
        do self.with |x| {
            f(cast::transmute_immut(x))
        }
    }
}

#[cfg(test)]
pub mod tests {
    use comm;
    use super::exclusive;
    use task;
    use uint;

    #[test]
    pub fn exclusive_arc() {
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

        for futures.each |f| { f.recv() }

        do total.with |total| {
            assert!(**total == num_tasks * count)
        };
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    pub fn exclusive_poison() {
        // Tests that if one task fails inside of an exclusive, subsequent
        // accesses will also fail.
        let x = exclusive(1);
        let x2 = x.clone();
        do task::try || {
            do x2.with |one| {
                assert!(*one == 2);
            }
        };
        do x.with |one| {
            assert!(*one == 1);
        }
    }
}
