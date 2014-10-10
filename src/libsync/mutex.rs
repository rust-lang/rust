// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A proper mutex implementation regardless of the "flavor of task" which is
//! acquiring the lock.

// # Implementation of Rust mutexes
//
// Most answers to the question of "how do I use a mutex" are "use pthreads",
// but for Rust this isn't quite sufficient. Green threads cannot acquire an OS
// mutex because they can context switch among many OS threads, leading to
// deadlocks with other green threads.
//
// Another problem for green threads grabbing an OS mutex is that POSIX dictates
// that unlocking a mutex on a different thread from where it was locked is
// undefined behavior. Remember that green threads can migrate among OS threads,
// so this would mean that we would have to pin green threads to OS threads,
// which is less than ideal.
//
// ## Using deschedule/reawaken
//
// We already have primitives for descheduling/reawakening tasks, so they're the
// first obvious choice when implementing a mutex. The idea would be to have a
// concurrent queue that everyone is pushed on to, and then the owner of the
// mutex is the one popping from the queue.
//
// Unfortunately, this is not very performant for native tasks. The suspected
// reason for this is that each native thread is suspended on its own condition
// variable, unique from all the other threads. In this situation, the kernel
// has no idea what the scheduling semantics are of the user program, so all of
// the threads are distributed among all cores on the system. This ends up
// having very expensive wakeups of remote cores high up in the profile when
// handing off the mutex among native tasks. On the other hand, when using an OS
// mutex, the kernel knows that all native threads are contended on the same
// mutex, so they're in theory all migrated to a single core (fast context
// switching).
//
// ## Mixing implementations
//
// From that above information, we have two constraints. The first is that
// green threads can't touch os mutexes, and the second is that native tasks
// pretty much *must* touch an os mutex.
//
// As a compromise, the queueing implementation is used for green threads and
// the os mutex is used for native threads (why not have both?). This ends up
// leading to fairly decent performance for both native threads and green
// threads on various workloads (uncontended and contended).
//
// The crux of this implementation is an atomic work which is CAS'd on many
// times in order to manage a few flags about who's blocking where and whether
// it's locked or not.

use core::prelude::*;

use alloc::boxed::Box;
use core::atomic;
use core::mem;
use core::cell::UnsafeCell;
use rustrt::local::Local;
use rustrt::mutex;
use rustrt::task::{BlockedTask, Task};
use rustrt::thread::Thread;

use mpsc_intrusive as q;

pub const LOCKED: uint = 1 << 0;
pub const GREEN_BLOCKED: uint = 1 << 1;
pub const NATIVE_BLOCKED: uint = 1 << 2;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex is an implementation of a lock for all flavors of tasks which may
/// be grabbing. A common problem with green threads is that they cannot grab
/// locks (if they reschedule during the lock a contender could deadlock the
/// system), but this mutex does *not* suffer this problem.
///
/// This mutex will properly block tasks waiting for the lock to become
/// available. The mutex can also be statically initialized or created via a
/// `new` constructor.
///
/// # Example
///
/// ```rust
/// use sync::mutex::Mutex;
///
/// let m = Mutex::new();
/// let guard = m.lock();
/// // do some work
/// drop(guard); // unlock the lock
/// ```
pub struct Mutex {
    // Note that this static mutex is in a *box*, not inlined into the struct
    // itself. This is done for memory safety reasons with the usage of a
    // StaticNativeMutex inside the static mutex above. Once a native mutex has
    // been used once, its address can never change (it can't be moved). This
    // mutex type can be safely moved at any time, so to ensure that the native
    // mutex is used correctly we box the inner lock to give it a constant
    // address.
    lock: Box<StaticMutex>,
}

#[deriving(PartialEq, Show)]
enum Flavor {
    Unlocked,
    TryLockAcquisition,
    GreenAcquisition,
    NativeAcquisition,
}

/// The static mutex type is provided to allow for static allocation of mutexes.
///
/// Note that this is a separate type because using a Mutex correctly means that
/// it needs to have a destructor run. In Rust, statics are not allowed to have
/// destructors. As a result, a `StaticMutex` has one extra method when compared
/// to a `Mutex`, a `destroy` method. This method is unsafe to call, and
/// documentation can be found directly on the method.
///
/// # Example
///
/// ```rust
/// use sync::mutex::{StaticMutex, MUTEX_INIT};
///
/// static mut LOCK: StaticMutex = MUTEX_INIT;
///
/// unsafe {
///     let _g = LOCK.lock();
///     // do some productive work
/// }
/// // lock is unlocked here.
/// ```
pub struct StaticMutex {
    /// Current set of flags on this mutex
    state: atomic::AtomicUint,
    /// an OS mutex used by native threads
    lock: mutex::StaticNativeMutex,

    /// Type of locking operation currently on this mutex
    flavor: UnsafeCell<Flavor>,
    /// uint-cast of the green thread waiting for this mutex
    green_blocker: UnsafeCell<uint>,
    /// uint-cast of the native thread waiting for this mutex
    native_blocker: UnsafeCell<uint>,

    /// A concurrent mpsc queue used by green threads, along with a count used
    /// to figure out when to dequeue and enqueue.
    q: q::Queue<uint>,
    green_cnt: atomic::AtomicUint,
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
#[must_use]
pub struct Guard<'a> {
    lock: &'a StaticMutex,
}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
pub const MUTEX_INIT: StaticMutex = StaticMutex {
    lock: mutex::NATIVE_MUTEX_INIT,
    state: atomic::INIT_ATOMIC_UINT,
    flavor: UnsafeCell { value: Unlocked },
    green_blocker: UnsafeCell { value: 0 },
    native_blocker: UnsafeCell { value: 0 },
    green_cnt: atomic::INIT_ATOMIC_UINT,
    q: q::Queue {
        head: atomic::INIT_ATOMIC_UINT,
        tail: UnsafeCell { value: 0 as *mut q::Node<uint> },
        stub: q::DummyNode {
            next: atomic::INIT_ATOMIC_UINT,
        }
    }
};

impl StaticMutex {
    /// Attempts to grab this lock, see `Mutex::try_lock`
    pub fn try_lock<'a>(&'a self) -> Option<Guard<'a>> {
        // Attempt to steal the mutex from an unlocked state.
        //
        // FIXME: this can mess up the fairness of the mutex, seems bad
        match self.state.compare_and_swap(0, LOCKED, atomic::SeqCst) {
            0 => {
                // After acquiring the mutex, we can safely access the inner
                // fields.
                let prev = unsafe {
                    mem::replace(&mut *self.flavor.get(), TryLockAcquisition)
                };
                assert_eq!(prev, Unlocked);
                Some(Guard::new(self))
            }
            _ => None
        }
    }

    /// Acquires this lock, see `Mutex::lock`
    pub fn lock<'a>(&'a self) -> Guard<'a> {
        // First, attempt to steal the mutex from an unlocked state. The "fast
        // path" needs to have as few atomic instructions as possible, and this
        // one cmpxchg is already pretty expensive.
        //
        // FIXME: this can mess up the fairness of the mutex, seems bad
        match self.try_lock() {
            Some(guard) => return guard,
            None => {}
        }

        // After we've failed the fast path, then we delegate to the different
        // locking protocols for green/native tasks. This will select two tasks
        // to continue further (one native, one green).
        let t: Box<Task> = Local::take();
        let can_block = t.can_block();
        let native_bit;
        if can_block {
            self.native_lock(t);
            native_bit = NATIVE_BLOCKED;
        } else {
            self.green_lock(t);
            native_bit = GREEN_BLOCKED;
        }

        // After we've arbitrated among task types, attempt to re-acquire the
        // lock (avoids a deschedule). This is very important to do in order to
        // allow threads coming out of the native_lock function to try their
        // best to not hit a cvar in deschedule.
        let mut old = match self.state.compare_and_swap(0, LOCKED,
                                                        atomic::SeqCst) {
            0 => {
                let flavor = if can_block {
                    NativeAcquisition
                } else {
                    GreenAcquisition
                };
                // We've acquired the lock, so this unsafe access to flavor is
                // allowed.
                unsafe { *self.flavor.get() = flavor; }
                return Guard::new(self)
            }
            old => old,
        };

        // Alright, everything else failed. We need to deschedule ourselves and
        // flag ourselves as waiting. Note that this case should only happen
        // regularly in native/green contention. Due to try_lock and the header
        // of lock stealing the lock, it's also possible for native/native
        // contention to hit this location, but as less common.
        let t: Box<Task> = Local::take();
        t.deschedule(1, |task| {
            let task = unsafe { task.cast_to_uint() };

            // These accesses are protected by the respective native/green
            // mutexes which were acquired above.
            let prev = if can_block {
                unsafe { mem::replace(&mut *self.native_blocker.get(), task) }
            } else {
                unsafe { mem::replace(&mut *self.green_blocker.get(), task) }
            };
            assert_eq!(prev, 0);

            loop {
                assert_eq!(old & native_bit, 0);
                // If the old state was locked, then we need to flag ourselves
                // as blocking in the state. If the old state was unlocked, then
                // we attempt to acquire the mutex. Everything here is a CAS
                // loop that'll eventually make progress.
                if old & LOCKED != 0 {
                    old = match self.state.compare_and_swap(old,
                                                            old | native_bit,
                                                            atomic::SeqCst) {
                        n if n == old => return Ok(()),
                        n => n
                    };
                } else {
                    assert_eq!(old, 0);
                    old = match self.state.compare_and_swap(old,
                                                            old | LOCKED,
                                                            atomic::SeqCst) {
                        n if n == old => {
                            // After acquiring the lock, we have access to the
                            // flavor field, and we've regained access to our
                            // respective native/green blocker field.
                            let prev = if can_block {
                                unsafe {
                                    *self.native_blocker.get() = 0;
                                    mem::replace(&mut *self.flavor.get(),
                                                 NativeAcquisition)
                                }
                            } else {
                                unsafe {
                                    *self.green_blocker.get() = 0;
                                    mem::replace(&mut *self.flavor.get(),
                                                 GreenAcquisition)
                                }
                            };
                            assert_eq!(prev, Unlocked);
                            return Err(unsafe {
                                BlockedTask::cast_from_uint(task)
                            })
                        }
                        n => n,
                    };
                }
            }
        });

        Guard::new(self)
    }

    // Tasks which can block are super easy. These tasks just call the blocking
    // `lock()` function on an OS mutex
    fn native_lock(&self, t: Box<Task>) {
        Local::put(t);
        unsafe { self.lock.lock_noguard(); }
    }

    fn native_unlock(&self) {
        unsafe { self.lock.unlock_noguard(); }
    }

    fn green_lock(&self, t: Box<Task>) {
        // Green threads flag their presence with an atomic counter, and if they
        // fail to be the first to the mutex, they enqueue themselves on a
        // concurrent internal queue with a stack-allocated node.
        //
        // FIXME: There isn't a cancellation currently of an enqueue, forcing
        //        the unlocker to spin for a bit.
        if self.green_cnt.fetch_add(1, atomic::SeqCst) == 0 {
            Local::put(t);
            return
        }

        let mut node = q::Node::new(0);
        t.deschedule(1, |task| {
            unsafe {
                node.data = task.cast_to_uint();
                self.q.push(&mut node);
            }
            Ok(())
        });
    }

    fn green_unlock(&self) {
        // If we're the only green thread, then no need to check the queue,
        // otherwise the fixme above forces us to spin for a bit.
        if self.green_cnt.fetch_sub(1, atomic::SeqCst) == 1 { return }
        let node;
        loop {
            match unsafe { self.q.pop() } {
                Some(t) => { node = t; break; }
                None => Thread::yield_now(),
            }
        }
        let task = unsafe { BlockedTask::cast_from_uint((*node).data) };
        task.wake().map(|t| t.reawaken());
    }

    fn unlock(&self) {
        // Unlocking this mutex is a little tricky. We favor any task that is
        // manually blocked (not in each of the separate locks) in order to help
        // provide a little fairness (green threads will wake up the pending
        // native thread and native threads will wake up the pending green
        // thread).
        //
        // There's also the question of when we unlock the actual green/native
        // locking halves as well. If we're waking up someone, then we can wait
        // to unlock until we've acquired the task to wake up (we're guaranteed
        // the mutex memory is still valid when there's contenders), but as soon
        // as we don't find any contenders we must unlock the mutex, and *then*
        // flag the mutex as unlocked.
        //
        // This flagging can fail, leading to another round of figuring out if a
        // task needs to be woken, and in this case it's ok that the "mutex
        // halves" are unlocked, we're just mainly dealing with the atomic state
        // of the outer mutex.
        let flavor = unsafe { mem::replace(&mut *self.flavor.get(), Unlocked) };

        let mut state = self.state.load(atomic::SeqCst);
        let mut unlocked = false;
        let task;
        loop {
            assert!(state & LOCKED != 0);
            if state & GREEN_BLOCKED != 0 {
                self.unset(state, GREEN_BLOCKED);
                task = unsafe {
                    *self.flavor.get() = GreenAcquisition;
                    let task = mem::replace(&mut *self.green_blocker.get(), 0);
                    BlockedTask::cast_from_uint(task)
                };
                break;
            } else if state & NATIVE_BLOCKED != 0 {
                self.unset(state, NATIVE_BLOCKED);
                task = unsafe {
                    *self.flavor.get() = NativeAcquisition;
                    let task = mem::replace(&mut *self.native_blocker.get(), 0);
                    BlockedTask::cast_from_uint(task)
                };
                break;
            } else {
                assert_eq!(state, LOCKED);
                if !unlocked {
                    match flavor {
                        GreenAcquisition => { self.green_unlock(); }
                        NativeAcquisition => { self.native_unlock(); }
                        TryLockAcquisition => {}
                        Unlocked => unreachable!(),
                    }
                    unlocked = true;
                }
                match self.state.compare_and_swap(LOCKED, 0, atomic::SeqCst) {
                    LOCKED => return,
                    n => { state = n; }
                }
            }
        }
        if !unlocked {
            match flavor {
                GreenAcquisition => { self.green_unlock(); }
                NativeAcquisition => { self.native_unlock(); }
                TryLockAcquisition => {}
                Unlocked => unreachable!(),
            }
        }

        task.wake().map(|t| t.reawaken());
    }

    /// Loops around a CAS to unset the `bit` in `state`
    fn unset(&self, mut state: uint, bit: uint) {
        loop {
            assert!(state & bit != 0);
            let new = state ^ bit;
            match self.state.compare_and_swap(state, new, atomic::SeqCst) {
                n if n == state => break,
                n => { state = n; }
            }
        }
    }

    /// Deallocates resources associated with this static mutex.
    ///
    /// This method is unsafe because it provides no guarantees that there are
    /// no active users of this mutex, and safety is not guaranteed if there are
    /// active users of this mutex.
    ///
    /// This method is required to ensure that there are no memory leaks on
    /// *all* platforms. It may be the case that some platforms do not leak
    /// memory if this method is not called, but this is not guaranteed to be
    /// true on all platforms.
    pub unsafe fn destroy(&self) {
        self.lock.destroy()
    }
}

impl Mutex {
    /// Creates a new mutex in an unlocked state ready for use.
    pub fn new() -> Mutex {
        Mutex {
            lock: box StaticMutex {
                state: atomic::AtomicUint::new(0),
                flavor: UnsafeCell::new(Unlocked),
                green_blocker: UnsafeCell::new(0),
                native_blocker: UnsafeCell::new(0),
                green_cnt: atomic::AtomicUint::new(0),
                q: q::Queue::new(),
                lock: unsafe { mutex::StaticNativeMutex::new() },
            }
        }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    pub fn try_lock<'a>(&'a self) -> Option<Guard<'a>> {
        self.lock.try_lock()
    }

    /// Acquires a mutex, blocking the current task until it is able to do so.
    ///
    /// This function will block the local task until it is available to acquire
    /// the mutex. Upon returning, the task is the only task with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    pub fn lock<'a>(&'a self) -> Guard<'a> { self.lock.lock() }
}

impl<'a> Guard<'a> {
    fn new<'b>(lock: &'b StaticMutex) -> Guard<'b> {
        if cfg!(debug) {
            // once we've acquired a lock, it's ok to access the flavor
            assert!(unsafe { *lock.flavor.get() != Unlocked });
            assert!(lock.state.load(atomic::SeqCst) & LOCKED != 0);
        }
        Guard { lock: lock }
    }
}

#[unsafe_destructor]
impl<'a> Drop for Guard<'a> {
    #[inline]
    fn drop(&mut self) {
        self.lock.unlock();
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        // This is actually safe b/c we know that there is no further usage of
        // this mutex (it's up to the user to arrange for a mutex to get
        // dropped, that's not our job)
        unsafe { self.lock.destroy() }
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use super::{Mutex, StaticMutex, MUTEX_INIT};
    use native;

    #[test]
    fn smoke() {
        let m = Mutex::new();
        drop(m.lock());
        drop(m.lock());
    }

    #[test]
    fn smoke_static() {
        static mut m: StaticMutex = MUTEX_INIT;
        unsafe {
            drop(m.lock());
            drop(m.lock());
            m.destroy();
        }
    }

    #[test]
    fn lots_and_lots() {
        static mut m: StaticMutex = MUTEX_INIT;
        static mut CNT: uint = 0;
        static M: uint = 1000;
        static N: uint = 3;

        fn inc() {
            for _ in range(0, M) {
                unsafe {
                    let _g = m.lock();
                    CNT += 1;
                }
            }
        }

        let (tx, rx) = channel();
        for _ in range(0, N) {
            let tx2 = tx.clone();
            native::task::spawn(proc() { inc(); tx2.send(()); });
            let tx2 = tx.clone();
            spawn(proc() { inc(); tx2.send(()); });
        }

        drop(tx);
        for _ in range(0, 2 * N) {
            rx.recv();
        }
        assert_eq!(unsafe {CNT}, M * N * 2);
        unsafe {
            m.destroy();
        }
    }

    #[test]
    fn trylock() {
        let m = Mutex::new();
        assert!(m.try_lock().is_some());
    }
}
