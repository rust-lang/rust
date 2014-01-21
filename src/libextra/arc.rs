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
 * Concurrency-enabled mechanisms for sharing mutable and/or immutable state
 * between tasks.
 *
 * # Example
 *
 * In this example, a large vector of floats is shared between several tasks.
 * With simple pipes, without Arc, a copy would have to be made for each task.
 *
 * ```rust
 * use extra::arc::Arc;
 * use std::{rand, vec};
 *
 * let numbers = vec::from_fn(100, |i| (i as f32) * rand::random());
 * let shared_numbers = Arc::new(numbers);
 *
 *   for _ in range(0, 10) {
 *       let (port, chan) = Chan::new();
 *       chan.send(shared_numbers.clone());
 *
 *       do spawn {
 *           let shared_numbers = port.recv();
 *           let local_numbers = shared_numbers.get();
 *
 *           // Work with the local numbers
 *       }
 *   }
 * ```
 */

#[allow(missing_doc)];


use sync;
use sync::{Mutex, RWLock};

use std::cast;
use std::sync::arc::UnsafeArc;
use std::task;
use std::borrow;
use std::unstable::mutex;

/// As sync::condvar, a mechanism for unlock-and-descheduling and signaling.
pub struct Condvar<'a> {
    priv is_mutex: bool,
    priv failed: &'a bool,
    priv cond: &'a sync::Condvar<'a>
}

impl<'a> Condvar<'a> {
    /// Atomically exit the associated Arc and block until a signal is sent.
    #[inline]
    pub fn wait(&self) { self.wait_on(0) }

    /**
     * Atomically exit the associated Arc and block on a specified condvar
     * until a signal is sent on that same condvar (as sync::cond.wait_on).
     *
     * wait() is equivalent to wait_on(0).
     */
    #[inline]
    pub fn wait_on(&self, condvar_id: uint) {
        assert!(!*self.failed);
        self.cond.wait_on(condvar_id);
        // This is why we need to wrap sync::condvar.
        check_poison(self.is_mutex, *self.failed);
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    #[inline]
    pub fn signal(&self) -> bool { self.signal_on(0) }

    /**
     * Wake up a blocked task on a specified condvar (as
     * sync::cond.signal_on). Returns false if there was no blocked task.
     */
    #[inline]
    pub fn signal_on(&self, condvar_id: uint) -> bool {
        assert!(!*self.failed);
        self.cond.signal_on(condvar_id)
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    #[inline]
    pub fn broadcast(&self) -> uint { self.broadcast_on(0) }

    /**
     * Wake up all blocked tasks on a specified condvar (as
     * sync::cond.broadcast_on). Returns the number of tasks woken.
     */
    #[inline]
    pub fn broadcast_on(&self, condvar_id: uint) -> uint {
        assert!(!*self.failed);
        self.cond.broadcast_on(condvar_id)
    }
}

/****************************************************************************
 * Immutable Arc
 ****************************************************************************/

/// An atomically reference counted wrapper for shared immutable state.
pub struct Arc<T> { priv x: UnsafeArc<T> }


/**
 * Access the underlying data in an atomically reference counted
 * wrapper.
 */
impl<T:Freeze+Send> Arc<T> {
    /// Create an atomically reference counted wrapper.
    #[inline]
    pub fn new(data: T) -> Arc<T> {
        Arc { x: UnsafeArc::new(data) }
    }

    #[inline]
    pub fn get<'a>(&'a self) -> &'a T {
        unsafe { &*self.x.get_immut() }
    }
}

impl<T:Freeze + Send> Clone for Arc<T> {
    /**
    * Duplicate an atomically reference counted wrapper.
    *
    * The resulting two `arc` objects will point to the same underlying data
    * object. However, one of the `arc` objects can be sent to another task,
    * allowing them to share the underlying data.
    */
    #[inline]
    fn clone(&self) -> Arc<T> {
        Arc { x: self.x.clone() }
    }
}

/****************************************************************************
 * Mutex protected Arc
 ****************************************************************************/

#[doc(hidden)]
struct MutexArcInner<T> {
    /// The lock.
    priv lock: mutex::Mutex,
    /// True if this mutex has been poisoned.
    priv poisoned: bool,
    /// The locked data.
    priv data: T,
}

/// Shared mutable data protected by a blocking mutex. This allows you to
/// share data between threads and write to it safely without data races.
///
/// Beware that cycles of `MutexArc` objects will leak. You must break cycles
/// manually.
///
/// If a task fails while the lock is held, the `MutexArc` will become
/// poisoned and any further attempts to lock it will fail.
#[no_freeze]
pub struct MutexArc<T> {
    priv arc: UnsafeArc<MutexArcInner<T>>,
}

/// A guard that represents a locked `MutexArc`.
pub struct MutexArcGuard<'a,T> {
    mutex_arc: &'a MutexArc<T>,
}

impl<T:Send> Clone for MutexArc<T> {
    /// Duplicates a mutex-protected Arc by incrementing the reference count.
    #[inline]
    fn clone(&self) -> MutexArc<T> {
        MutexArc {
            arc: self.arc.clone(),
        }
    }
}

impl<T:Send> MutexArc<T> {
    /// Creates mutex-protected shared data from the supplied value.
    pub fn new(value: T) -> MutexArc<T> {
        unsafe {
            MutexArc {
                arc: UnsafeArc::new(MutexArcInner {
                    lock: mutex::Mutex::new(),
                    poisoned: false,
                    data: value,
                })
            }
        }
    }

    /// Takes the lock and hands out a scoped mutable reference.
    #[inline]
    pub fn lock<'a>(&'a self) -> MutexArcGuard<'a,T> {
        unsafe {
            (*self.arc.get()).lock.lock();
            if (*self.arc.get()).poisoned {
                die_from_poison()
            }
        }
        MutexArcGuard {
            mutex_arc: self,
        }
    }
}

/// Called when the Arc is poisoned. This is out of line to guide
/// inlining.
#[inline(never)]
fn die_from_poison() -> ! {
    fail!("Arc was poisoned; another task failed inside!")
}

impl<'a,T:Send> MutexArcGuard<'a,T> {
    /// Returns a reference to the value inside.
    pub fn get<'b>(&'b self) -> &'b mut T {
        unsafe {
            &mut (*self.mutex_arc.arc.get()).data
        }
    }
}

#[unsafe_destructor]
impl<'a,T:Send> Drop for MutexArcGuard<'a,T> {
    fn drop(&mut self) {
        unsafe {
            if task::failing() {
                (*self.mutex_arc.arc.get()).poisoned = true
            }
            (*self.mutex_arc.arc.get()).lock.unlock()
        }
    }
}


/****************************************************************************
 * R/W lock protected Arc
 ****************************************************************************/

// Common code for {mutex.access,rwlock.write}{,_cond}.
#[inline]
#[doc(hidden)]
fn check_poison(is_mutex: bool, failed: bool) {
    if failed {
        if is_mutex {
            fail!("Poisoned MutexArc - another task failed inside!");
        } else {
            fail!("Poisoned rw_arc - another task failed inside!");
        }
    }
}

struct PoisonOnFail {
    flag: *mut bool,
    failed: bool,
}

impl Drop for PoisonOnFail {
    fn drop(&mut self) {
        unsafe {
            /* assert!(!*self.failed);
               -- might be false in case of cond.wait() */
            if !self.failed && task::failing() {
                *self.flag = true;
            }
        }
    }
}

impl PoisonOnFail {
    fn new<'a>(flag: &'a mut bool) -> PoisonOnFail {
        PoisonOnFail {
            flag: flag,
            failed: task::failing()
        }
    }
}

#[doc(hidden)]
struct RWArcInner<T> { priv lock: RWLock, priv failed: bool, priv data: T }
/**
 * A dual-mode Arc protected by a reader-writer lock. The data can be accessed
 * mutably or immutably, and immutably-accessing tasks may run concurrently.
 *
 * Unlike mutex_arcs, rw_arcs are safe, because they cannot be nested.
 */
#[no_freeze]
pub struct RWArc<T> {
    priv x: UnsafeArc<RWArcInner<T>>,
}

impl<T:Freeze + Send> Clone for RWArc<T> {
    /// Duplicate a rwlock-protected Arc. See arc::clone for more details.
    #[inline]
    fn clone(&self) -> RWArc<T> {
        RWArc { x: self.x.clone() }
    }

}

impl<T:Freeze + Send> RWArc<T> {
    /// Create a reader/writer Arc with the supplied data.
    pub fn new(user_data: T) -> RWArc<T> {
        RWArc::new_with_condvars(user_data, 1)
    }

    /**
     * Create a reader/writer Arc with the supplied data and a specified number
     * of condvars (as sync::RWLock::new_with_condvars).
     */
    pub fn new_with_condvars(user_data: T, num_condvars: uint) -> RWArc<T> {
        let data = RWArcInner {
            lock: RWLock::new_with_condvars(num_condvars),
            failed: false, data: user_data
        };
        RWArc { x: UnsafeArc::new(data), }
    }

    /**
     * Access the underlying data mutably. Locks the rwlock in write mode;
     * other readers and writers will block.
     *
     * # Failure
     *
     * Failing while inside the Arc will unlock the Arc while unwinding, so
     * that other tasks won't block forever. As MutexArc.access, it will also
     * poison the Arc, so subsequent readers and writers will both also fail.
     */
    #[inline]
    pub fn write<U>(&self, blk: |x: &mut T| -> U) -> U {
        unsafe {
            let state = self.x.get();
            (*borrow_rwlock(state)).write(|| {
                check_poison(false, (*state).failed);
                let _z = PoisonOnFail::new(&mut (*state).failed);
                blk(&mut (*state).data)
            })
        }
    }

    /// As write(), but with a condvar, as sync::rwlock.write_cond().
    #[inline]
    pub fn write_cond<U>(&self,
                         blk: |x: &mut T, c: &Condvar| -> U)
                         -> U {
        unsafe {
            let state = self.x.get();
            (*borrow_rwlock(state)).write_cond(|cond| {
                check_poison(false, (*state).failed);
                let _z = PoisonOnFail::new(&mut (*state).failed);
                blk(&mut (*state).data,
                    &Condvar {is_mutex: false,
                              failed: &(*state).failed,
                              cond: cond})
            })
        }
    }

    /**
     * Access the underlying data immutably. May run concurrently with other
     * reading tasks.
     *
     * # Failure
     *
     * Failing will unlock the Arc while unwinding. However, unlike all other
     * access modes, this will not poison the Arc.
     */
    pub fn read<U>(&self, blk: |x: &T| -> U) -> U {
        unsafe {
            let state = self.x.get();
            (*state).lock.read(|| {
                check_poison(false, (*state).failed);
                blk(&(*state).data)
            })
        }
    }

    /**
     * As write(), but with the ability to atomically 'downgrade' the lock.
     * See sync::rwlock.write_downgrade(). The RWWriteMode token must be used
     * to obtain the &mut T, and can be transformed into a RWReadMode token by
     * calling downgrade(), after which a &T can be obtained instead.
     *
     * # Example
     *
     * ```rust
     * use extra::arc::RWArc;
     *
     * let arc = RWArc::new(1);
     * arc.write_downgrade(|mut write_token| {
     *     write_token.write_cond(|state, condvar| {
     *         // ... exclusive access with mutable state ...
     *     });
     *     let read_token = arc.downgrade(write_token);
     *     read_token.read(|state| {
     *         // ... shared access with immutable state ...
     *     });
     * })
     * ```
     */
    pub fn write_downgrade<U>(&self, blk: |v: RWWriteMode<T>| -> U) -> U {
        unsafe {
            let state = self.x.get();
            (*borrow_rwlock(state)).write_downgrade(|write_mode| {
                check_poison(false, (*state).failed);
                blk(RWWriteMode {
                    data: &mut (*state).data,
                    token: write_mode,
                    poison: PoisonOnFail::new(&mut (*state).failed)
                })
            })
        }
    }

    /// To be called inside of the write_downgrade block.
    pub fn downgrade<'a>(&self, token: RWWriteMode<'a, T>)
                         -> RWReadMode<'a, T> {
        unsafe {
            // The rwlock should assert that the token belongs to us for us.
            let state = self.x.get();
            let RWWriteMode {
                data: data,
                token: t,
                poison: _poison
            } = token;
            // Let readers in
            let new_token = (*state).lock.downgrade(t);
            // Whatever region the input reference had, it will be safe to use
            // the same region for the output reference. (The only 'unsafe' part
            // of this cast is removing the mutability.)
            let new_data = data;
            // Downgrade ensured the token belonged to us. Just a sanity check.
            assert!(borrow::ref_eq(&(*state).data, new_data));
            // Produce new token
            RWReadMode {
                data: new_data,
                token: new_token,
            }
        }
    }
}

// Borrowck rightly complains about immutably aliasing the rwlock in order to
// lock it. This wraps the unsafety, with the justification that the 'lock'
// field is never overwritten; only 'failed' and 'data'.
#[doc(hidden)]
fn borrow_rwlock<T:Freeze + Send>(state: *mut RWArcInner<T>) -> *RWLock {
    unsafe { cast::transmute(&(*state).lock) }
}

/// The "write permission" token used for RWArc.write_downgrade().
pub struct RWWriteMode<'a, T> {
    priv data: &'a mut T,
    priv token: sync::RWLockWriteMode<'a>,
    priv poison: PoisonOnFail,
}

/// The "read permission" token used for RWArc.write_downgrade().
pub struct RWReadMode<'a, T> {
    priv data: &'a T,
    priv token: sync::RWLockReadMode<'a>,
}

impl<'a, T:Freeze + Send> RWWriteMode<'a, T> {
    /// Access the pre-downgrade RWArc in write mode.
    pub fn write<U>(&mut self, blk: |x: &mut T| -> U) -> U {
        match *self {
            RWWriteMode {
                data: &ref mut data,
                token: ref token,
                poison: _
            } => {
                token.write(|| blk(data))
            }
        }
    }

    /// Access the pre-downgrade RWArc in write mode with a condvar.
    pub fn write_cond<U>(&mut self,
                         blk: |x: &mut T, c: &Condvar| -> U)
                         -> U {
        match *self {
            RWWriteMode {
                data: &ref mut data,
                token: ref token,
                poison: ref poison
            } => {
                token.write_cond(|cond| {
                    unsafe {
                        let cvar = Condvar {
                            is_mutex: false,
                            failed: &*poison.flag,
                            cond: cond
                        };
                        blk(data, &cvar)
                    }
                })
            }
        }
    }
}

impl<'a, T:Freeze + Send> RWReadMode<'a, T> {
    /// Access the post-downgrade rwlock in read mode.
    pub fn read<U>(&self, blk: |x: &T| -> U) -> U {
        match *self {
            RWReadMode {
                data: data,
                token: ref token
            } => {
                token.read(|| blk(data))
            }
        }
    }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {

    use arc::*;

    use std::task;

    #[test]
    fn manually_share_arc() {
        let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = Arc::new(v);

        let (p, c) = Chan::new();

        do task::spawn {
            let arc_v: Arc<~[int]> = p.recv();

            let v = arc_v.get().clone();
            assert_eq!(v[3], 4);
        };

        c.send(arc_v.clone());

        assert_eq!(arc_v.get()[2], 3);
        assert_eq!(arc_v.get()[4], 5);

        info!("{:?}", arc_v);
    }

    #[test] #[should_fail]
    fn test_mutex_arc_poison() {
        let arc = ~MutexArc::new(1);
        let arc2 = ~arc.clone();
        do task::try || {
            let lock = arc2.lock();
            assert_eq!(*lock.get(), 2);
        };
        let lock = arc.lock();
        assert_eq!(*lock.get(), 1);
    }

    #[test]
    fn test_unsafe_mutex_arc_nested() {
        unsafe {
            // Tests nested mutexes and access
            // to underlaying data.
            let arc = ~MutexArc::new(1);
            let arc2 = ~MutexArc::new((*arc).clone());
            do task::spawn || {
                let lock2 = arc2.lock();
                let lock = arc.lock();
                assert!(*lock.get() == 1)
            };
        }
    }

    #[test] #[should_fail]
    fn test_rw_arc_poison_wr() {
        let arc = RWArc::new(1);
        let arc2 = arc.clone();
        do task::try {
            arc2.write(|one| {
                assert_eq!(*one, 2);
            })
        };
        arc.read(|one| {
            assert_eq!(*one, 1);
        })
    }

    #[test] #[should_fail]
    fn test_rw_arc_poison_ww() {
        let arc = RWArc::new(1);
        let arc2 = arc.clone();
        do task::try {
            arc2.write(|one| {
                assert_eq!(*one, 2);
            })
        };
        arc.write(|one| {
            assert_eq!(*one, 1);
        })
    }
    #[test] #[should_fail]
    fn test_rw_arc_poison_dw() {
        let arc = RWArc::new(1);
        let arc2 = arc.clone();
        do task::try {
            arc2.write_downgrade(|mut write_mode| {
                write_mode.write(|one| {
                    assert_eq!(*one, 2);
                })
            })
        };
        arc.write(|one| {
            assert_eq!(*one, 1);
        })
    }
    #[test]
    fn test_rw_arc_no_poison_rr() {
        let arc = RWArc::new(1);
        let arc2 = arc.clone();
        do task::try {
            arc2.read(|one| {
                assert_eq!(*one, 2);
            })
        };
        arc.read(|one| {
            assert_eq!(*one, 1);
        })
    }
    #[test]
    fn test_rw_arc_no_poison_rw() {
        let arc = RWArc::new(1);
        let arc2 = arc.clone();
        do task::try {
            arc2.read(|one| {
                assert_eq!(*one, 2);
            })
        };
        arc.write(|one| {
            assert_eq!(*one, 1);
        })
    }
    #[test]
    fn test_rw_arc_no_poison_dr() {
        let arc = RWArc::new(1);
        let arc2 = arc.clone();
        do task::try {
            arc2.write_downgrade(|write_mode| {
                let read_mode = arc2.downgrade(write_mode);
                read_mode.read(|one| {
                    assert_eq!(*one, 2);
                })
            })
        };
        arc.write(|one| {
            assert_eq!(*one, 1);
        })
    }
    #[test]
    fn test_rw_arc() {
        let arc = RWArc::new(0);
        let arc2 = arc.clone();
        let (p, c) = Chan::new();

        do task::spawn {
            arc2.write(|num| {
                10.times(|| {
                    let tmp = *num;
                    *num = -1;
                    task::deschedule();
                    *num = tmp + 1;
                });
                c.send(());
            })
        }

        // Readers try to catch the writer in the act
        let mut children = ~[];
        5.times(|| {
            let arc3 = arc.clone();
            let mut builder = task::task();
            children.push(builder.future_result());
            do builder.spawn {
                arc3.read(|num| {
                    assert!(*num >= 0);
                })
            }
        });

        // Wait for children to pass their asserts
        for r in children.mut_iter() {
            r.recv();
        }

        // Wait for writer to finish
        p.recv();
        arc.read(|num| {
            assert_eq!(*num, 10);
        })
    }

    #[test]
    fn test_rw_arc_access_in_unwind() {
        let arc = RWArc::new(1i);
        let arc2 = arc.clone();
        task::try::<()>(proc() {
            struct Unwinder {
                i: RWArc<int>
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    self.i.write(|num| *num += 1);
                }
            }
            let _u = Unwinder { i: arc2 };
            fail!();
        });
        assert_eq!(2, arc.read(|n| *n));
    }

    #[test]
    fn test_rw_downgrade() {
        // (1) A downgrader gets in write mode and does cond.wait.
        // (2) A writer gets in write mode, sets state to 42, and does signal.
        // (3) Downgrader wakes, sets state to 31337.
        // (4) tells writer and all other readers to contend as it downgrades.
        // (5) Writer attempts to set state back to 42, while downgraded task
        //     and all reader tasks assert that it's 31337.
        let arc = RWArc::new(0);

        // Reader tasks
        let mut reader_convos = ~[];
        10.times(|| {
            let ((rp1, rc1), (rp2, rc2)) = (Chan::new(), Chan::new());
            reader_convos.push((rc1, rp2));
            let arcn = arc.clone();
            do task::spawn {
                rp1.recv(); // wait for downgrader to give go-ahead
                arcn.read(|state| {
                    assert_eq!(*state, 31337);
                    rc2.send(());
                })
            }
        });

        // Writer task
        let arc2 = arc.clone();
        let ((wp1, wc1), (wp2, wc2)) = (Chan::new(), Chan::new());
        do task::spawn || {
            wp1.recv();
            arc2.write_cond(|state, cond| {
                assert_eq!(*state, 0);
                *state = 42;
                cond.signal();
            });
            wp1.recv();
            arc2.write(|state| {
                // This shouldn't happen until after the downgrade read
                // section, and all other readers, finish.
                assert_eq!(*state, 31337);
                *state = 42;
            });
            wc2.send(());
        }

        // Downgrader (us)
        arc.write_downgrade(|mut write_mode| {
            write_mode.write_cond(|state, cond| {
                wc1.send(()); // send to another writer who will wake us up
                while *state == 0 {
                    cond.wait();
                }
                assert_eq!(*state, 42);
                *state = 31337;
                // send to other readers
                for &(ref mut rc, _) in reader_convos.mut_iter() {
                    rc.send(())
                }
            });
            let read_mode = arc.downgrade(write_mode);
            read_mode.read(|state| {
                // complete handshake with other readers
                for &(_, ref mut rp) in reader_convos.mut_iter() {
                    rp.recv()
                }
                wc1.send(()); // tell writer to try again
                assert_eq!(*state, 31337);
            });
        });

        wp2.recv(); // complete handshake with writer
    }
    #[cfg(test)]
    fn test_rw_write_cond_downgrade_read_race_helper() {
        // Tests that when a downgrader hands off the "reader cloud" lock
        // because of a contending reader, a writer can't race to get it
        // instead, which would result in readers_and_writers. This tests
        // the sync module rather than this one, but it's here because an
        // rwarc gives us extra shared state to help check for the race.
        // If you want to see this test fail, go to sync.rs and replace the
        // line in RWLock::write_cond() that looks like:
        //     "blk(&Condvar { order: opt_lock, ..*cond })"
        // with just "blk(cond)".
        let x = RWArc::new(true);
        let (wp, wc) = Chan::new();

        // writer task
        let xw = x.clone();
        do task::spawn {
            xw.write_cond(|state, c| {
                wc.send(()); // tell downgrader it's ok to go
                c.wait();
                // The core of the test is here: the condvar reacquire path
                // must involve order_lock, so that it cannot race with a reader
                // trying to receive the "reader cloud lock hand-off".
                *state = false;
            })
        }

        wp.recv(); // wait for writer to get in

        x.write_downgrade(|mut write_mode| {
            write_mode.write_cond(|state, c| {
                assert!(*state);
                // make writer contend in the cond-reacquire path
                c.signal();
            });
            // make a reader task to trigger the "reader cloud lock" handoff
            let xr = x.clone();
            let (rp, rc) = Chan::new();
            do task::spawn {
                rc.send(());
                xr.read(|_state| { })
            }
            rp.recv(); // wait for reader task to exist

            let read_mode = x.downgrade(write_mode);
            read_mode.read(|state| {
                // if writer mistakenly got in, make sure it mutates state
                // before we assert on it
                5.times(|| task::deschedule());
                // make sure writer didn't get in.
                assert!(*state);
            })
        });
    }
    #[test]
    fn test_rw_write_cond_downgrade_read_race() {
        // Ideally the above test case would have deschedule statements in it that
        // helped to expose the race nearly 100% of the time... but adding
        // deschedules in the intuitively-right locations made it even less likely,
        // and I wasn't sure why :( . This is a mediocre "next best" option.
        8.times(|| test_rw_write_cond_downgrade_read_race_helper());
    }
}
