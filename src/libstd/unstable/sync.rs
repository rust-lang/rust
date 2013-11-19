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
use cell::Cell;
use comm;
use ptr;
use option::{Option,Some,None};
use task;
use unstable::atomics::{AtomicOption,AtomicUint,Acquire,Release,Relaxed,SeqCst};
use unstable::finally::Finally;
use unstable::mutex::Mutex;
use ops::Drop;
use clone::Clone;
use kinds::Send;
use vec;

/// An atomically reference counted pointer.
///
/// Enforces no shared-memory safety.
//#[unsafe_no_drop_flag] FIXME: #9758
pub struct UnsafeArc<T> {
    data: *mut ArcData<T>,
}

pub enum UnsafeArcUnwrap<T> {
    UnsafeArcSelf(UnsafeArc<T>),
    UnsafeArcT(T)
}

impl<T> UnsafeArcUnwrap<T> {
    fn expect_t(self, msg: &'static str) -> T {
        match self {
            UnsafeArcSelf(_) => fail!(msg),
            UnsafeArcT(t) => t
        }
    }

    fn is_self(&self) -> bool {
        match *self {
            UnsafeArcSelf(_) => true,
            UnsafeArcT(_) => false
        }
    }
}

struct ArcData<T> {
    count: AtomicUint,
    // An unwrapper uses this protocol to communicate with the "other" task that
    // drops the last refcount on an arc. Unfortunately this can't be a proper
    // pipe protocol because the unwrapper has to access both stages at once.
    // FIXME(#7544): Maybe use AtomicPtr instead (to avoid xchg in take() later)?
    unwrapper: AtomicOption<(comm::ChanOne<()>, comm::PortOne<bool>)>,
    // FIXME(#3224) should be able to make this non-option to save memory
    data: Option<T>,
}

unsafe fn new_inner<T: Send>(data: T, refcount: uint) -> *mut ArcData<T> {
    let data = ~ArcData { count: AtomicUint::new(refcount),
                          unwrapper: AtomicOption::empty(),
                          data: Some(data) };
    cast::transmute(data)
}

impl<T: Send> UnsafeArc<T> {
    pub fn new(data: T) -> UnsafeArc<T> {
        unsafe { UnsafeArc { data: new_inner(data, 1) } }
    }

    /// As new(), but returns an extra pre-cloned handle.
    pub fn new2(data: T) -> (UnsafeArc<T>, UnsafeArc<T>) {
        unsafe {
            let ptr = new_inner(data, 2);
            (UnsafeArc { data: ptr }, UnsafeArc { data: ptr })
        }
    }

    /// As new(), but returns a vector of as many pre-cloned handles as requested.
    pub fn newN(data: T, num_handles: uint) -> ~[UnsafeArc<T>] {
        unsafe {
            if num_handles == 0 {
                ~[] // need to free data here
            } else {
                let ptr = new_inner(data, num_handles);
                vec::from_fn(num_handles, |_| UnsafeArc { data: ptr })
            }
        }
    }

    /// As newN(), but from an already-existing handle. Uses one xadd.
    pub fn cloneN(self, num_handles: uint) -> ~[UnsafeArc<T>] {
        if num_handles == 0 {
            ~[] // The "num_handles - 1" trick (below) fails in the 0 case.
        } else {
            unsafe {
                // Minus one because we are recycling the given handle's refcount.
                let old_count = (*self.data).count.fetch_add(num_handles - 1, Acquire);
                // let old_count = (*self.data).count.fetch_add(num_handles, Acquire);
                assert!(old_count >= 1);
                let ptr = self.data;
                cast::forget(self); // Don't run the destructor on this handle.
                vec::from_fn(num_handles, |_| UnsafeArc { data: ptr })
            }
        }
    }

    #[inline]
    pub fn get(&self) -> *mut T {
        unsafe {
            assert!((*self.data).count.load(Relaxed) > 0);
            let r: *mut T = (*self.data).data.get_mut_ref();
            return r;
        }
    }

    #[inline]
    pub fn get_immut(&self) -> *T {
        unsafe {
            assert!((*self.data).count.load(Relaxed) > 0);
            let r: *T = (*self.data).data.get_ref();
            return r;
        }
    }

    /// Wait until all other handles are dropped, then retrieve the enclosed
    /// data. See extra::arc::Arc for specific semantics documentation.
    /// If called when the task is already unkillable, unwrap will unkillably
    /// block; otherwise, an unwrapping task can be killed by linked failure.
    pub fn unwrap(self) -> T {
        let this = Cell::new(self); // argh
        do task::unkillable {
            unsafe {
                let mut this = this.take();
                // The ~ dtor needs to run if this code succeeds.
                let mut data: ~ArcData<T> = cast::transmute(this.data);
                // Set up the unwrap protocol.
                let (p1,c1) = comm::oneshot(); // ()
                let (p2,c2) = comm::oneshot(); // bool
                // Try to put our server end in the unwrapper slot.
                // This needs no barrier -- it's protected by the release barrier on
                // the xadd, and the acquire+release barrier in the destructor's xadd.
                if data.unwrapper.fill(~(c1,p2), Relaxed).is_none() {
                    // Got in. Tell this handle's destructor not to run (we are now it).
                    this.data = ptr::mut_null();
                    // Drop our own reference.
                    let old_count = data.count.fetch_sub(1, Release);
                    assert!(old_count >= 1);
                    if old_count == 1 {
                        // We were the last owner. Can unwrap immediately.
                        // AtomicOption's destructor will free the server endpoint.
                        // FIXME(#3224): it should be like this
                        // let ~ArcData { data: user_data, _ } = data;
                        // user_data
                        data.data.take_unwrap()
                    } else {
                        // The *next* person who sees the refcount hit 0 will wake us.
                        let p1 = Cell::new(p1); // argh
                        // Unlike the above one, this cell is necessary. It will get
                        // taken either in the do block or in the finally block.
                        let c2_and_data = Cell::new((c2,data));
                        do (|| {
                            do task::rekillable { p1.take().recv(); }
                            // Got here. Back in the 'unkillable' without getting killed.
                            let (c2, data) = c2_and_data.take();
                            c2.send(true);
                            // FIXME(#3224): it should be like this
                            // let ~ArcData { data: user_data, _ } = data;
                            // user_data
                            let mut data = data;
                            data.data.take_unwrap()
                        }).finally {
                            if task::failing() {
                                // Killed during wait. Because this might happen while
                                // someone else still holds a reference, we can't free
                                // the data now; the "other" last refcount will free it.
                                let (c2, data) = c2_and_data.take();
                                c2.send(false);
                                cast::forget(data);
                            } else {
                                assert!(c2_and_data.is_empty());
                            }
                        }
                    }
                } else {
                    // If 'put' returns the server end back to us, we were rejected;
                    // someone else was trying to unwrap. Avoid guaranteed deadlock.
                    cast::forget(data);
                    fail!("Another task is already unwrapping this Arc!");
                }
            }
        }
    }

    /// As unwrap above, but without blocking. Returns 'UnsafeArcSelf(self)' if this is
    /// not the last reference; 'UnsafeArcT(unwrapped_data)' if so.
    pub fn try_unwrap(mut self) -> UnsafeArcUnwrap<T> {
        unsafe {
            // The ~ dtor needs to run if this code succeeds.
            let mut data: ~ArcData<T> = cast::transmute(self.data);
            // This can of course race with anybody else who has a handle, but in
            // such a case, the returned count will always be at least 2. If we
            // see 1, no race was possible. All that matters is 1 or not-1.
            let count = data.count.load(Acquire);
            assert!(count >= 1);
            // The more interesting race is one with an unwrapper. They may have
            // already dropped their count -- but if so, the unwrapper pointer
            // will have been set first, which the barriers ensure we will see.
            // (Note: using is_empty(), not take(), to not free the unwrapper.)
            if count == 1 && data.unwrapper.is_empty(Acquire) {
                // Tell this handle's destructor not to run (we are now it).
                self.data = ptr::mut_null();
                // FIXME(#3224) as above
                UnsafeArcT(data.data.take_unwrap())
            } else {
                cast::forget(data);
                UnsafeArcSelf(self)
            }
        }
    }
}

impl<T: Send> Clone for UnsafeArc<T> {
    fn clone(&self) -> UnsafeArc<T> {
        unsafe {
            // This barrier might be unnecessary, but I'm not sure...
            let old_count = (*self.data).count.fetch_add(1, Acquire);
            assert!(old_count >= 1);
            return UnsafeArc { data: self.data };
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for UnsafeArc<T>{
    fn drop(&mut self) {
        unsafe {
            // Happens when destructing an unwrapper's handle and from `#[unsafe_no_drop_flag]`
            if self.data.is_null() {
                return
            }
            let mut data: ~ArcData<T> = cast::transmute(self.data);
            // Must be acquire+release, not just release, to make sure this
            // doesn't get reordered to after the unwrapper pointer load.
            let old_count = data.count.fetch_sub(1, SeqCst);
            assert!(old_count >= 1);
            if old_count == 1 {
                // Were we really last, or should we hand off to an
                // unwrapper? It's safe to not xchg because the unwrapper
                // will set the unwrap lock *before* dropping his/her
                // reference. In effect, being here means we're the only
                // *awake* task with the data.
                match data.unwrapper.take(Acquire) {
                    Some(~(message,response)) => {
                        let cell = Cell::new((message, response, data));
                        do task::unkillable {
                            let (message, response, data) = cell.take();
                            // Send 'ready' and wait for a response.
                            message.send(());
                            // Unkillable wait. Message guaranteed to come.
                            if response.recv() {
                                // Other task got the data.
                                cast::forget(data);
                            } else {
                                // Other task was killed. drop glue takes over.
                            }
                        }
                    }
                    None => {
                        // drop glue takes over.
                    }
                }
            } else {
                cast::forget(data);
            }
        }
    }
}


/****************************************************************************/

/**
 * Enables a runtime assertion that no operation in the argument closure shall
 * use scheduler operations (deschedule, recv, spawn, etc). This is for use with
 * pthread mutexes, which may block the entire scheduler thread, rather than
 * just one task, and is hence prone to deadlocks if mixed with descheduling.
 *
 * NOTE: THIS DOES NOT PROVIDE LOCKING, or any sort of critical-section
 * synchronization whatsoever. It only makes sense to use for CPU-local issues.
 */
// FIXME(#8140) should not be pub
pub unsafe fn atomically<U>(f: || -> U) -> U {
    use rt::task::{Task, GreenTask, SchedTask};
    use rt::local::Local;

    let task_opt: Option<*mut Task> = Local::try_unsafe_borrow();
    match task_opt {
        Some(t) => {
            match (*t).task_type {
                GreenTask(_) => {
                    do (|| {
                        (*t).death.inhibit_deschedule();
                        f()
                    }).finally {
                        (*t).death.allow_deschedule();
                    }
                }
                SchedTask => f()
            }
        }
        None => f()
    }
}

pub struct LittleLock {
    priv l: Mutex,
}

impl Drop for LittleLock {
    fn drop(&mut self) {
        unsafe {
            self.l.destroy();
        }
    }
}

impl LittleLock {
    pub fn new() -> LittleLock {
        unsafe {
            LittleLock {
                l: Mutex::new()
            }
        }
    }

    pub unsafe fn lock<T>(&self, f: || -> T) -> T {
        let this = cast::transmute_mut(self);
        do atomically {
            this.l.lock();
            do (|| {
                f()
            }).finally {
                this.l.unlock();
            }
        }
    }

    pub unsafe fn try_lock<T>(&self, f: || -> T) -> Option<T> {
        let this = cast::transmute_mut(self);
        do atomically {
            if this.l.trylock() {
                Some(do (|| {
                    f()
                }).finally {
                    this.l.unlock();
                })
            } else {
                None
            }
        }
    }

    pub unsafe fn signal(&self) {
        let this = cast::transmute_mut(self);
        this.l.signal();
    }

    pub unsafe fn lock_and_wait(&self, f: || -> bool) {
        let this = cast::transmute_mut(self);
        do atomically {
            this.l.lock();
            do (|| {
                if f() {
                    this.l.wait();
                }
            }).finally {
                this.l.unlock();
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
 *
 * # Safety note
 *
 * This uses a pthread mutex, not one that's aware of the userspace scheduler.
 * The user of an Exclusive must be careful not to invoke any functions that may
 * reschedule the task while holding the lock, or deadlock may result. If you
 * need to block or deschedule while accessing shared state, use extra::sync::RWArc.
 */
pub struct Exclusive<T> {
    priv x: UnsafeArc<ExData<T>>
}

impl<T:Send> Clone for Exclusive<T> {
    // Duplicate an Exclusive Arc, as std::arc::clone.
    fn clone(&self) -> Exclusive<T> {
        Exclusive { x: self.x.clone() }
    }
}

impl<T:Send> Exclusive<T> {
    pub fn new(user_data: T) -> Exclusive<T> {
        let data = ExData {
            lock: LittleLock::new(),
            failed: false,
            data: user_data
        };
        Exclusive {
            x: UnsafeArc::new(data)
        }
    }

    // Exactly like std::arc::MutexArc,access(), but with the LittleLock
    // instead of a proper mutex. Same reason for being unsafe.
    //
    // Currently, scheduling operations (i.e., descheduling, receiving on a pipe,
    // accessing the provided condition variable) are prohibited while inside
    // the Exclusive. Supporting that is a work in progress.
    #[inline]
    pub unsafe fn with<U>(&self, f: |x: &mut T| -> U) -> U {
        let rec = self.x.get();
        do (*rec).lock.lock {
            if (*rec).failed {
                fail!("Poisoned Exclusive::new - another task failed inside!");
            }
            (*rec).failed = true;
            let result = f(&mut (*rec).data);
            (*rec).failed = false;
            result
        }
    }

    #[inline]
    pub unsafe fn with_imm<U>(&self, f: |x: &T| -> U) -> U {
        do self.with |x| {
            f(cast::transmute_immut(x))
        }
    }

    #[inline]
    pub unsafe fn hold_and_signal(&self, f: |x: &mut T|) {
        let rec = self.x.get();
        do (*rec).lock.lock {
            if (*rec).failed {
                fail!("Poisoned Exclusive::new - another task failed inside!");
            }
            (*rec).failed = true;
            f(&mut (*rec).data);
            (*rec).failed = false;
            (*rec).lock.signal();
        }
    }

    #[inline]
    pub unsafe fn hold_and_wait(&self, f: |x: &T| -> bool) {
        let rec = self.x.get();
        do (*rec).lock.lock_and_wait {
            if (*rec).failed {
                fail!("Poisoned Exclusive::new - another task failed inside!");
            }
            (*rec).failed = true;
            let result = f(&(*rec).data);
            (*rec).failed = false;
            result
        }
    }

    pub fn unwrap(self) -> T {
        let Exclusive { x: x } = self;
        // Someday we might need to unkillably unwrap an Exclusive, but not today.
        let inner = x.unwrap();
        let ExData { data: user_data, _ } = inner; // will destroy the LittleLock
        user_data
    }
}

#[cfg(test)]
mod tests {
    use cell::Cell;
    use comm;
    use option::*;
    use prelude::*;
    use super::{Exclusive, UnsafeArc, atomically};
    use task;
    use util;
    use mem::size_of;

    //#[unsafe_no_drop_flag] FIXME: #9758
    #[ignore]
    #[test]
    fn test_size() {
        assert_eq!(size_of::<UnsafeArc<[int, ..10]>>(), size_of::<*[int, ..10]>());
    }

    #[test]
    fn test_atomically() {
        // NB. The whole runtime will abort on an 'atomic-sleep' violation,
        // so we can't really test for the converse behaviour.
        unsafe { do atomically { } } task::deschedule(); // oughtn't fail
    }

    #[test]
    fn exclusive_new_arc() {
        unsafe {
            let mut futures = ~[];

            let num_tasks = 10;
            let count = 10;

            let total = Exclusive::new(~0);

            for _ in range(0u, num_tasks) {
                let total = total.clone();
                let (port, chan) = comm::stream();
                futures.push(port);

                do task::spawn || {
                    for _ in range(0u, count) {
                        do total.with |count| {
                            **count += 1;
                        }
                    }
                    chan.send(());
                }
            };

            for f in futures.iter() { f.recv() }

            do total.with |total| {
                assert!(**total == num_tasks * count)
            };
        }
    }

    #[test] #[should_fail]
    fn exclusive_new_poison() {
        unsafe {
            // Tests that if one task fails inside of an Exclusive::new, subsequent
            // accesses will also fail.
            let x = Exclusive::new(1);
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

    #[test]
    fn arclike_newN() {
        // Tests that the many-refcounts-at-once constructors don't leak.
        let _ = UnsafeArc::new2(~~"hello");
        let x = UnsafeArc::newN(~~"hello", 0);
        assert_eq!(x.len(), 0)
        let x = UnsafeArc::newN(~~"hello", 1);
        assert_eq!(x.len(), 1)
        let x = UnsafeArc::newN(~~"hello", 10);
        assert_eq!(x.len(), 10)
    }

    #[test]
    fn arclike_cloneN() {
        // Tests that the many-refcounts-at-once special-clone doesn't leak.
        let x = UnsafeArc::new(~~"hello");
        let x = x.cloneN(0);
        assert_eq!(x.len(), 0);
        let x = UnsafeArc::new(~~"hello");
        let x = x.cloneN(1);
        assert_eq!(x.len(), 1);
        let x = UnsafeArc::new(~~"hello");
        let x = x.cloneN(10);
        assert_eq!(x.len(), 10);
    }

    #[test]
    fn arclike_unwrap_basic() {
        let x = UnsafeArc::new(~~"hello");
        assert!(x.unwrap() == ~~"hello");
    }

    #[test]
    fn arclike_try_unwrap() {
        let x = UnsafeArc::new(~~"hello");
        assert!(x.try_unwrap().expect_t("try_unwrap failed") == ~~"hello");
    }

    #[test]
    fn arclike_try_unwrap_fail() {
        let x = UnsafeArc::new(~~"hello");
        let x2 = x.clone();
        let left_x = x.try_unwrap();
        assert!(left_x.is_self());
        util::ignore(left_x);
        assert!(x2.try_unwrap().expect_t("try_unwrap none") == ~~"hello");
    }

    #[test]
    fn arclike_try_unwrap_unwrap_race() {
        // When an unwrap and a try_unwrap race, the unwrapper should always win.
        let x = UnsafeArc::new(~~"hello");
        let x2 = Cell::new(x.clone());
        let (p,c) = comm::stream();
        do task::spawn {
            c.send(());
            assert!(x2.take().unwrap() == ~~"hello");
            c.send(());
        }
        p.recv();
        task::deschedule(); // Try to make the unwrapper get blocked first.
        let left_x = x.try_unwrap();
        assert!(left_x.is_self());
        util::ignore(left_x);
        p.recv();
    }

    #[test]
    fn exclusive_new_unwrap_basic() {
        // Unlike the above, also tests no double-freeing of the LittleLock.
        let x = Exclusive::new(~~"hello");
        assert!(x.unwrap() == ~~"hello");
    }

    #[test]
    fn exclusive_new_unwrap_contended() {
        let x = Exclusive::new(~~"hello");
        let x2 = Cell::new(x.clone());
        do task::spawn {
            let x2 = x2.take();
            unsafe { do x2.with |_hello| { } }
            task::deschedule();
        }
        assert!(x.unwrap() == ~~"hello");

        // Now try the same thing, but with the child task blocking.
        let x = Exclusive::new(~~"hello");
        let x2 = Cell::new(x.clone());
        let mut builder = task::task();
        let res = builder.future_result();
        do builder.spawn {
            let x2 = x2.take();
            assert!(x2.unwrap() == ~~"hello");
        }
        // Have to get rid of our reference before blocking.
        util::ignore(x);
        res.recv();
    }

    #[test] #[should_fail]
    fn exclusive_new_unwrap_conflict() {
        let x = Exclusive::new(~~"hello");
        let x2 = Cell::new(x.clone());
        let mut builder = task::task();
        let res = builder.future_result();
        do builder.spawn {
            let x2 = x2.take();
            assert!(x2.unwrap() == ~~"hello");
        }
        assert!(x.unwrap() == ~~"hello");
        assert!(res.recv().is_ok());
    }

    #[test]
    fn exclusive_new_unwrap_deadlock() {
        // This is not guaranteed to get to the deadlock before being killed,
        // but it will show up sometimes, and if the deadlock were not there,
        // the test would nondeterministically fail.
        let result = do task::try {
            // a task that has two references to the same Exclusive::new will
            // deadlock when it unwraps. nothing to be done about that.
            let x = Exclusive::new(~~"hello");
            let x2 = x.clone();
            do task::spawn {
                do 10.times { task::deschedule(); } // try to let the unwrapper go
                fail!(); // punt it awake from its deadlock
            }
            let _z = x.unwrap();
            unsafe { do x2.with |_hello| { } }
        };
        assert!(result.is_err());
    }
}
