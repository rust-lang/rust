// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */


use std::borrow;
use std::unstable::sync::{Exclusive, UnsafeArc};
use std::unstable::atomics;
use std::unstable::finally::Finally;
use std::util;
use std::util::NonCopyable;

/****************************************************************************
 * Internals
 ****************************************************************************/

// Each waiting task receives on one of these.
#[doc(hidden)]
type WaitEnd = Port<()>;
#[doc(hidden)]
type SignalEnd = Chan<()>;
// A doubly-ended queue of waiting tasks.
#[doc(hidden)]
struct WaitQueue { head: Port<SignalEnd>,
                   tail: Chan<SignalEnd> }

impl WaitQueue {
    fn new() -> WaitQueue {
        let (block_head, block_tail) = Chan::new();
        WaitQueue { head: block_head, tail: block_tail }
    }

    // Signals one live task from the queue.
    fn signal(&self) -> bool {
        match self.head.try_recv() {
            Some(ch) => {
                // Send a wakeup signal. If the waiter was killed, its port will
                // have closed. Keep trying until we get a live task.
                if ch.try_send_deferred(()) {
                    true
                } else {
                    self.signal()
                }
            }
            None => false
        }
    }

    fn broadcast(&self) -> uint {
        let mut count = 0;
        loop {
            match self.head.try_recv() {
                None => break,
                Some(ch) => {
                    if ch.try_send_deferred(()) {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn wait_end(&self) -> WaitEnd {
        let (wait_end, signal_end) = Chan::new();
        self.tail.send_deferred(signal_end);
        wait_end
    }
}

// The building-block used to make semaphores, mutexes, and rwlocks.
#[doc(hidden)]
struct SemInner<Q> {
    count: int,
    waiters:   WaitQueue,
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked:   Q
}

#[doc(hidden)]
struct Sem<Q>(Exclusive<SemInner<Q>>);

#[doc(hidden)]
impl<Q:Send> Sem<Q> {
    fn new(count: int, q: Q) -> Sem<Q> {
        Sem(Exclusive::new(SemInner {
            count: count, waiters: WaitQueue::new(), blocked: q }))
    }

    pub fn acquire(&self) {
        unsafe {
            let mut waiter_nobe = None;
            (**self).with(|state| {
                state.count -= 1;
                if state.count < 0 {
                    // Create waiter nobe, enqueue ourself, and tell
                    // outer scope we need to block.
                    waiter_nobe = Some(state.waiters.wait_end());
                }
            });
            // Uncomment if you wish to test for sem races. Not valgrind-friendly.
            /* 1000.times(|| task::deschedule()); */
            // Need to wait outside the exclusive.
            if waiter_nobe.is_some() {
                let _ = waiter_nobe.unwrap().recv();
            }
        }
    }

    pub fn release(&self) {
        unsafe {
            (**self).with(|state| {
                state.count += 1;
                if state.count <= 0 {
                    state.waiters.signal();
                }
            })
        }
    }

    pub fn access<U>(&self, blk: || -> U) -> U {
        (|| {
            self.acquire();
            blk()
        }).finally(|| {
            self.release();
        })
    }
}

#[doc(hidden)]
impl Sem<~[WaitQueue]> {
    fn new_and_signal(count: int, num_condvars: uint)
        -> Sem<~[WaitQueue]> {
        let mut queues = ~[];
        num_condvars.times(|| queues.push(WaitQueue::new()));
        Sem::new(count, queues)
    }
}

// FIXME(#3598): Want to use an Option down below, but we need a custom enum
// that's not polymorphic to get around the fact that lifetimes are invariant
// inside of type parameters.
enum ReacquireOrderLock<'a> {
    Nothing, // c.c
    Just(&'a Semaphore),
}

/// A mechanism for atomic-unlock-and-deschedule blocking and signalling.
pub struct Condvar<'a> {
    // The 'Sem' object associated with this condvar. This is the one that's
    // atomically-unlocked-and-descheduled upon and reacquired during wakeup.
    priv sem: &'a Sem<~[WaitQueue]>,
    // This is (can be) an extra semaphore which is held around the reacquire
    // operation on the first one. This is only used in cvars associated with
    // rwlocks, and is needed to ensure that, when a downgrader is trying to
    // hand off the access lock (which would be the first field, here), a 2nd
    // writer waking up from a cvar wait can't race with a reader to steal it,
    // See the comment in write_cond for more detail.
    priv order: ReacquireOrderLock<'a>,
    // Make sure condvars are non-copyable.
    priv token: util::NonCopyable,
}

impl<'a> Condvar<'a> {
    /**
     * Atomically drop the associated lock, and block until a signal is sent.
     *
     * # Failure
     * A task which is killed (i.e., by linked failure with another task)
     * while waiting on a condition variable will wake up, fail, and unlock
     * the associated lock as it unwinds.
     */
    pub fn wait(&self) { self.wait_on(0) }

    /**
     * As wait(), but can specify which of multiple condition variables to
     * wait on. Only a signal_on() or broadcast_on() with the same condvar_id
     * will wake this thread.
     *
     * The associated lock must have been initialised with an appropriate
     * number of condvars. The condvar_id must be between 0 and num_condvars-1
     * or else this call will fail.
     *
     * wait() is equivalent to wait_on(0).
     */
    pub fn wait_on(&self, condvar_id: uint) {
        let mut WaitEnd = None;
        let mut out_of_bounds = None;
        // Release lock, 'atomically' enqueuing ourselves in so doing.
        unsafe {
            (**self.sem).with(|state| {
                if condvar_id < state.blocked.len() {
                    // Drop the lock.
                    state.count += 1;
                    if state.count <= 0 {
                        state.waiters.signal();
                    }
                    // Create waiter nobe, and enqueue ourself to
                    // be woken up by a signaller.
                    WaitEnd = Some(state.blocked[condvar_id].wait_end());
                } else {
                    out_of_bounds = Some(state.blocked.len());
                }
            })
        }

        // If deschedule checks start getting inserted anywhere, we can be
        // killed before or after enqueueing.
        check_cvar_bounds(out_of_bounds, condvar_id, "cond.wait_on()", || {
            // Unconditionally "block". (Might not actually block if a
            // signaller already sent -- I mean 'unconditionally' in contrast
            // with acquire().)
            (|| {
                let _ = WaitEnd.take_unwrap().recv();
            }).finally(|| {
                // Reacquire the condvar.
                match self.order {
                    Just(lock) => lock.access(|| self.sem.acquire()),
                    Nothing => self.sem.acquire(),
                }
            })
        })
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    pub fn signal(&self) -> bool { self.signal_on(0) }

    /// As signal, but with a specified condvar_id. See wait_on.
    pub fn signal_on(&self, condvar_id: uint) -> bool {
        unsafe {
            let mut out_of_bounds = None;
            let mut result = false;
            (**self.sem).with(|state| {
                if condvar_id < state.blocked.len() {
                    result = state.blocked[condvar_id].signal();
                } else {
                    out_of_bounds = Some(state.blocked.len());
                }
            });
            check_cvar_bounds(out_of_bounds,
                              condvar_id,
                              "cond.signal_on()",
                              || result)
        }
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    pub fn broadcast(&self) -> uint { self.broadcast_on(0) }

    /// As broadcast, but with a specified condvar_id. See wait_on.
    pub fn broadcast_on(&self, condvar_id: uint) -> uint {
        let mut out_of_bounds = None;
        let mut queue = None;
        unsafe {
            (**self.sem).with(|state| {
                if condvar_id < state.blocked.len() {
                    // To avoid :broadcast_heavy, we make a new waitqueue,
                    // swap it out with the old one, and broadcast on the
                    // old one outside of the little-lock.
                    queue = Some(util::replace(&mut state.blocked[condvar_id],
                                               WaitQueue::new()));
                } else {
                    out_of_bounds = Some(state.blocked.len());
                }
            });
            check_cvar_bounds(out_of_bounds,
                              condvar_id,
                              "cond.signal_on()",
                              || {
                queue.take_unwrap().broadcast()
            })
        }
    }
}

// Checks whether a condvar ID was out of bounds, and fails if so, or does
// something else next on success.
#[inline]
#[doc(hidden)]
fn check_cvar_bounds<U>(
                     out_of_bounds: Option<uint>,
                     id: uint,
                     act: &str,
                     blk: || -> U)
                     -> U {
    match out_of_bounds {
        Some(0) =>
            fail!("{} with illegal ID {} - this lock has no condvars!", act, id),
        Some(length) =>
            fail!("{} with illegal ID {} - ID must be less than {}", act, id, length),
        None => blk()
    }
}

#[doc(hidden)]
impl Sem<~[WaitQueue]> {
    // The only other places that condvars get built are rwlock.write_cond()
    // and rwlock_write_mode.
    pub fn access_cond<U>(&self, blk: |c: &Condvar| -> U) -> U {
        self.access(|| {
            blk(&Condvar {
                sem: self,
                order: Nothing,
                token: NonCopyable
            })
        })
    }
}

/****************************************************************************
 * Semaphores
 ****************************************************************************/

/// A counting, blocking, bounded-waiting semaphore.
struct Semaphore { priv sem: Sem<()> }


impl Clone for Semaphore {
    /// Create a new handle to the semaphore.
    fn clone(&self) -> Semaphore {
        Semaphore { sem: Sem((*self.sem).clone()) }
    }
}

impl Semaphore {
    /// Create a new semaphore with the specified count.
    pub fn new(count: int) -> Semaphore {
        Semaphore { sem: Sem::new(count, ()) }
    }

    /**
     * Acquire a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    pub fn acquire(&self) { (&self.sem).acquire() }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist. Won't block the caller.
     */
    pub fn release(&self) { (&self.sem).release() }

    /// Run a function with ownership of one of the semaphore's resources.
    pub fn access<U>(&self, blk: || -> U) -> U { (&self.sem).access(blk) }
}

/****************************************************************************
 * Mutexes
 ****************************************************************************/

/**
 * A blocking, bounded-waiting, mutual exclusion lock with an associated
 * FIFO condition variable.
 *
 * # Failure
 * A task which fails while holding a mutex will unlock the mutex as it
 * unwinds.
 */

pub struct Mutex { priv sem: Sem<~[WaitQueue]> }
impl Clone for Mutex {
    /// Create a new handle to the mutex.
    fn clone(&self) -> Mutex { Mutex { sem: Sem((*self.sem).clone()) } }
}

impl Mutex {
    /// Create a new mutex, with one associated condvar.
    pub fn new() -> Mutex { Mutex::new_with_condvars(1) }

    /**
    * Create a new mutex, with a specified number of associated condvars. This
    * will allow calling wait_on/signal_on/broadcast_on with condvar IDs between
    * 0 and num_condvars-1. (If num_condvars is 0, lock_cond will be allowed but
    * any operations on the condvar will fail.)
    */
    pub fn new_with_condvars(num_condvars: uint) -> Mutex {
        Mutex { sem: Sem::new_and_signal(1, num_condvars) }
    }


    /// Run a function with ownership of the mutex.
    pub fn lock<U>(&self, blk: || -> U) -> U {
        (&self.sem).access(blk)
    }

    /// Run a function with ownership of the mutex and a handle to a condvar.
    pub fn lock_cond<U>(&self, blk: |c: &Condvar| -> U) -> U {
        (&self.sem).access_cond(blk)
    }
}

/****************************************************************************
 * Reader-writer locks
 ****************************************************************************/

// NB: Wikipedia - Readers-writers_problem#The_third_readers-writers_problem

#[doc(hidden)]
struct RWLockInner {
    // You might ask, "Why don't you need to use an atomic for the mode flag?"
    // This flag affects the behaviour of readers (for plain readers, they
    // assert on it; for downgraders, they use it to decide which mode to
    // unlock for). Consider that the flag is only unset when the very last
    // reader exits; therefore, it can never be unset during a reader/reader
    // (or reader/downgrader) race.
    // By the way, if we didn't care about the assert in the read unlock path,
    // we could instead store the mode flag in write_downgrade's stack frame,
    // and have the downgrade tokens store a borrowed pointer to it.
    read_mode:  bool,
    // The only way the count flag is ever accessed is with xadd. Since it is
    // a read-modify-write operation, multiple xadds on different cores will
    // always be consistent with respect to each other, so a monotonic/relaxed
    // consistency ordering suffices (i.e., no extra barriers are needed).
    // FIXME(#6598): The atomics module has no relaxed ordering flag, so I use
    // acquire/release orderings superfluously. Change these someday.
    read_count: atomics::AtomicUint,
}

/**
 * A blocking, no-starvation, reader-writer lock with an associated condvar.
 *
 * # Failure
 * A task which fails while holding an rwlock will unlock the rwlock as it
 * unwinds.
 */
pub struct RWLock {
    priv order_lock:  Semaphore,
    priv access_lock: Sem<~[WaitQueue]>,
    priv state:       UnsafeArc<RWLockInner>,
}

impl RWLock {
    /// Create a new rwlock, with one associated condvar.
    pub fn new() -> RWLock { RWLock::new_with_condvars(1) }

    /**
    * Create a new rwlock, with a specified number of associated condvars.
    * Similar to mutex_with_condvars.
    */
    pub fn new_with_condvars(num_condvars: uint) -> RWLock {
        let state = UnsafeArc::new(RWLockInner {
            read_mode:  false,
            read_count: atomics::AtomicUint::new(0),
        });
        RWLock { order_lock:  Semaphore::new(1),
                access_lock: Sem::new_and_signal(1, num_condvars),
                state:       state, }
    }

    /// Create a new handle to the rwlock.
    pub fn clone(&self) -> RWLock {
        RWLock { order_lock:  (&(self.order_lock)).clone(),
                 access_lock: Sem((*self.access_lock).clone()),
                 state:       self.state.clone() }
    }

    /**
     * Run a function with the rwlock in read mode. Calls to 'read' from other
     * tasks may run concurrently with this one.
     */
    pub fn read<U>(&self, blk: || -> U) -> U {
        unsafe {
            (&self.order_lock).access(|| {
                let state = &mut *self.state.get();
                let old_count = state.read_count.fetch_add(1, atomics::Acquire);
                if old_count == 0 {
                    (&self.access_lock).acquire();
                    state.read_mode = true;
                }
            });
            (|| {
                blk()
            }).finally(|| {
                let state = &mut *self.state.get();
                assert!(state.read_mode);
                let old_count = state.read_count.fetch_sub(1, atomics::Release);
                assert!(old_count > 0);
                if old_count == 1 {
                    state.read_mode = false;
                    // Note: this release used to be outside of a locked access
                    // to exclusive-protected state. If this code is ever
                    // converted back to such (instead of using atomic ops),
                    // this access MUST NOT go inside the exclusive access.
                    (&self.access_lock).release();
                }
            })
        }
    }

    /**
     * Run a function with the rwlock in write mode. No calls to 'read' or
     * 'write' from other tasks will run concurrently with this one.
     */
    pub fn write<U>(&self, blk: || -> U) -> U {
        (&self.order_lock).acquire();
        (&self.access_lock).access(|| {
            (&self.order_lock).release();
            blk()
        })
    }

    /**
     * As write(), but also with a handle to a condvar. Waiting on this
     * condvar will allow readers and writers alike to take the rwlock before
     * the waiting task is signalled. (Note: a writer that waited and then
     * was signalled might reacquire the lock before other waiting writers.)
     */
    pub fn write_cond<U>(&self, blk: |c: &Condvar| -> U) -> U {
        // It's important to thread our order lock into the condvar, so that
        // when a cond.wait() wakes up, it uses it while reacquiring the
        // access lock. If we permitted a waking-up writer to "cut in line",
        // there could arise a subtle race when a downgrader attempts to hand
        // off the reader cloud lock to a waiting reader. This race is tested
        // in arc.rs (test_rw_write_cond_downgrade_read_race) and looks like:
        // T1 (writer)              T2 (downgrader)             T3 (reader)
        // [in cond.wait()]
        //                          [locks for writing]
        //                          [holds access_lock]
        // [is signalled, perhaps by
        //  downgrader or a 4th thread]
        // tries to lock access(!)
        //                                                      lock order_lock
        //                                                      xadd read_count[0->1]
        //                                                      tries to lock access
        //                          [downgrade]
        //                          xadd read_count[1->2]
        //                          unlock access
        // Since T1 contended on the access lock before T3 did, it will steal
        // the lock handoff. Adding order_lock in the condvar reacquire path
        // solves this because T1 will hold order_lock while waiting on access,
        // which will cause T3 to have to wait until T1 finishes its write,
        // which can't happen until T2 finishes the downgrade-read entirely.
        // The astute reader will also note that making waking writers use the
        // order_lock is better for not starving readers.
        (&self.order_lock).acquire();
        (&self.access_lock).access_cond(|cond| {
            (&self.order_lock).release();
            let opt_lock = Just(&self.order_lock);
            blk(&Condvar { sem: cond.sem, order: opt_lock,
                           token: NonCopyable })
        })
    }

    /**
     * As write(), but with the ability to atomically 'downgrade' the lock;
     * i.e., to become a reader without letting other writers get the lock in
     * the meantime (such as unlocking and then re-locking as a reader would
     * do). The block takes a "write mode token" argument, which can be
     * transformed into a "read mode token" by calling downgrade(). Example:
     *
     * # Example
     *
     * ```rust
     * lock.write_downgrade(|mut write_token| {
     *     write_token.write_cond(|condvar| {
     *         ... exclusive access ...
     *     });
     *     let read_token = lock.downgrade(write_token);
     *     read_token.read(|| {
     *         ... shared access ...
     *     })
     * })
     * ```
     */
    pub fn write_downgrade<U>(&self, blk: |v: RWLockWriteMode| -> U) -> U {
        // Implementation slightly different from the slicker 'write's above.
        // The exit path is conditional on whether the caller downgrades.
        (&self.order_lock).acquire();
        (&self.access_lock).acquire();
        (&self.order_lock).release();
        (|| {
            blk(RWLockWriteMode { lock: self, token: NonCopyable })
        }).finally(|| {
            let writer_or_last_reader;
            // Check if we're releasing from read mode or from write mode.
            let state = unsafe { &mut *self.state.get() };
            if state.read_mode {
                // Releasing from read mode.
                let old_count = state.read_count.fetch_sub(1, atomics::Release);
                assert!(old_count > 0);
                // Check if other readers remain.
                if old_count == 1 {
                    // Case 1: Writer downgraded & was the last reader
                    writer_or_last_reader = true;
                    state.read_mode = false;
                } else {
                    // Case 2: Writer downgraded & was not the last reader
                    writer_or_last_reader = false;
                }
            } else {
                // Case 3: Writer did not downgrade
                writer_or_last_reader = true;
            }
            if writer_or_last_reader {
                // Nobody left inside; release the "reader cloud" lock.
                (&self.access_lock).release();
            }
        })
    }

    /// To be called inside of the write_downgrade block.
    pub fn downgrade<'a>(&self, token: RWLockWriteMode<'a>)
                         -> RWLockReadMode<'a> {
        if !borrow::ref_eq(self, token.lock) {
            fail!("Can't downgrade() with a different rwlock's write_mode!");
        }
        unsafe {
            let state = &mut *self.state.get();
            assert!(!state.read_mode);
            state.read_mode = true;
            // If a reader attempts to enter at this point, both the
            // downgrader and reader will set the mode flag. This is fine.
            let old_count = state.read_count.fetch_add(1, atomics::Release);
            // If another reader was already blocking, we need to hand-off
            // the "reader cloud" access lock to them.
            if old_count != 0 {
                // Guaranteed not to let another writer in, because
                // another reader was holding the order_lock. Hence they
                // must be the one to get the access_lock (because all
                // access_locks are acquired with order_lock held). See
                // the comment in write_cond for more justification.
                (&self.access_lock).release();
            }
        }
        RWLockReadMode { lock: token.lock, token: NonCopyable }
    }
}

/// The "write permission" token used for rwlock.write_downgrade().

pub struct RWLockWriteMode<'a> { priv lock: &'a RWLock, priv token: NonCopyable }
/// The "read permission" token used for rwlock.write_downgrade().
pub struct RWLockReadMode<'a> { priv lock: &'a RWLock,
                                   priv token: NonCopyable }

impl<'a> RWLockWriteMode<'a> {
    /// Access the pre-downgrade rwlock in write mode.
    pub fn write<U>(&self, blk: || -> U) -> U { blk() }
    /// Access the pre-downgrade rwlock in write mode with a condvar.
    pub fn write_cond<U>(&self, blk: |c: &Condvar| -> U) -> U {
        // Need to make the condvar use the order lock when reacquiring the
        // access lock. See comment in RWLock::write_cond for why.
        blk(&Condvar { sem:        &self.lock.access_lock,
                       order: Just(&self.lock.order_lock),
                       token: NonCopyable })
    }
}

impl<'a> RWLockReadMode<'a> {
    /// Access the post-downgrade rwlock in read mode.
    pub fn read<U>(&self, blk: || -> U) -> U { blk() }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use sync::*;

    use std::cast;
    use std::result;
    use std::task;

    /************************************************************************
     * Semaphore tests
     ************************************************************************/
    #[test]
    fn test_sem_acquire_release() {
        let s = Semaphore::new(1);
        s.acquire();
        s.release();
        s.acquire();
    }
    #[test]
    fn test_sem_basic() {
        let s = Semaphore::new(1);
        s.access(|| { })
    }
    #[test]
    fn test_sem_as_mutex() {
        let s = Semaphore::new(1);
        let s2 = s.clone();
        do task::spawn {
            s2.access(|| {
                5.times(|| { task::deschedule(); })
            })
        }
        s.access(|| {
            5.times(|| { task::deschedule(); })
        })
    }
    #[test]
    fn test_sem_as_cvar() {
        /* Child waits and parent signals */
        let (p, c) = Chan::new();
        let s = Semaphore::new(0);
        let s2 = s.clone();
        do task::spawn {
            s2.acquire();
            c.send(());
        }
        5.times(|| { task::deschedule(); });
        s.release();
        let _ = p.recv();

        /* Parent waits and child signals */
        let (p, c) = Chan::new();
        let s = Semaphore::new(0);
        let s2 = s.clone();
        do task::spawn {
            5.times(|| { task::deschedule(); });
            s2.release();
            let _ = p.recv();
        }
        s.acquire();
        c.send(());
    }
    #[test]
    fn test_sem_multi_resource() {
        // Parent and child both get in the critical section at the same
        // time, and shake hands.
        let s = Semaphore::new(2);
        let s2 = s.clone();
        let (p1,c1) = Chan::new();
        let (p2,c2) = Chan::new();
        do task::spawn {
            s2.access(|| {
                let _ = p2.recv();
                c1.send(());
            })
        }
        s.access(|| {
            c2.send(());
            let _ = p1.recv();
        })
    }
    #[test]
    fn test_sem_runtime_friendly_blocking() {
        // Force the runtime to schedule two threads on the same sched_loop.
        // When one blocks, it should schedule the other one.
        do task::spawn_sched(task::SingleThreaded) {
            let s = Semaphore::new(1);
            let s2 = s.clone();
            let (p, c) = Chan::new();
            let mut child_data = Some((s2, c));
            s.access(|| {
                let (s2, c) = child_data.take_unwrap();
                do task::spawn {
                    c.send(());
                    s2.access(|| { });
                    c.send(());
                }
                let _ = p.recv(); // wait for child to come alive
                5.times(|| { task::deschedule(); }); // let the child contend
            });
            let _ = p.recv(); // wait for child to be done
        }
    }
    /************************************************************************
     * Mutex tests
     ************************************************************************/
    #[test]
    fn test_mutex_lock() {
        // Unsafely achieve shared state, and do the textbook
        // "load tmp = move ptr; inc tmp; store ptr <- tmp" dance.
        let (p, c) = Chan::new();
        let m = Mutex::new();
        let m2 = m.clone();
        let mut sharedstate = ~0;
        {
            let ptr: *int = &*sharedstate;
            do task::spawn {
                let sharedstate: &mut int =
                    unsafe { cast::transmute(ptr) };
                access_shared(sharedstate, &m2, 10);
                c.send(());

            }
        }
        {
            access_shared(sharedstate, &m, 10);
            let _ = p.recv();

            assert_eq!(*sharedstate, 20);
        }

        fn access_shared(sharedstate: &mut int, m: &Mutex, n: uint) {
            n.times(|| {
                m.lock(|| {
                    let oldval = *sharedstate;
                    task::deschedule();
                    *sharedstate = oldval + 1;
                })
            })
        }
    }
    #[test]
    fn test_mutex_cond_wait() {
        let m = Mutex::new();

        // Child wakes up parent
        m.lock_cond(|cond| {
            let m2 = m.clone();
            do task::spawn {
                m2.lock_cond(|cond| {
                    let woken = cond.signal();
                    assert!(woken);
                })
            }
            cond.wait();
        });
        // Parent wakes up child
        let (port,chan) = Chan::new();
        let m3 = m.clone();
        do task::spawn {
            m3.lock_cond(|cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            })
        }
        let _ = port.recv(); // Wait until child gets in the mutex
        m.lock_cond(|cond| {
            let woken = cond.signal();
            assert!(woken);
        });
        let _ = port.recv(); // Wait until child wakes up
    }
    #[cfg(test)]
    fn test_mutex_cond_broadcast_helper(num_waiters: uint) {
        let m = Mutex::new();
        let mut ports = ~[];

        num_waiters.times(|| {
            let mi = m.clone();
            let (port, chan) = Chan::new();
            ports.push(port);
            do task::spawn {
                mi.lock_cond(|cond| {
                    chan.send(());
                    cond.wait();
                    chan.send(());
                })
            }
        });

        // wait until all children get in the mutex
        for port in ports.mut_iter() { let _ = port.recv(); }
        m.lock_cond(|cond| {
            let num_woken = cond.broadcast();
            assert_eq!(num_woken, num_waiters);
        });
        // wait until all children wake up
        for port in ports.mut_iter() { let _ = port.recv(); }
    }
    #[test]
    fn test_mutex_cond_broadcast() {
        test_mutex_cond_broadcast_helper(12);
    }
    #[test]
    fn test_mutex_cond_broadcast_none() {
        test_mutex_cond_broadcast_helper(0);
    }
    #[test]
    fn test_mutex_cond_no_waiter() {
        let m = Mutex::new();
        let m2 = m.clone();
        do task::try {
            m.lock_cond(|_x| { })
        };
        m2.lock_cond(|cond| {
            assert!(!cond.signal());
        })
    }
    #[test]
    fn test_mutex_killed_simple() {
        // Mutex must get automatically unlocked if failed/killed within.
        let m = Mutex::new();
        let m2 = m.clone();

        let result: result::Result<(), ~Any> = do task::try {
            m2.lock(|| {
                fail!();
            })
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        m.lock(|| { })
    }
    #[ignore(reason = "linked failure")]
    #[test]
    fn test_mutex_killed_cond() {
        // Getting killed during cond wait must not corrupt the mutex while
        // unwinding (e.g. double unlock).
        let m = Mutex::new();
        let m2 = m.clone();

        let result: result::Result<(), ~Any> = do task::try {
            let (p, c) = Chan::new();
            do task::spawn { // linked
                let _ = p.recv(); // wait for sibling to get in the mutex
                task::deschedule();
                fail!();
            }
            m2.lock_cond(|cond| {
                c.send(()); // tell sibling go ahead
                cond.wait(); // block forever
            })
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        m.lock_cond(|cond| {
            let woken = cond.signal();
            assert!(!woken);
        })
    }
    #[ignore(reason = "linked failure")]
    #[test]
    fn test_mutex_killed_broadcast() {
        use std::unstable::finally::Finally;

        let m = Mutex::new();
        let m2 = m.clone();
        let (p, c) = Chan::new();

        let result: result::Result<(), ~Any> = do task::try {
            let mut sibling_convos = ~[];
            2.times(|| {
                let (p, c) = Chan::new();
                sibling_convos.push(p);
                let mi = m2.clone();
                // spawn sibling task
                do task::spawn { // linked
                    mi.lock_cond(|cond| {
                        c.send(()); // tell sibling to go ahead
                        (|| {
                            cond.wait(); // block forever
                        }).finally(|| {
                            error!("task unwinding and sending");
                            c.send(());
                            error!("task unwinding and done sending");
                        })
                    })
                }
            });
            for p in sibling_convos.mut_iter() {
                let _ = p.recv(); // wait for sibling to get in the mutex
            }
            m2.lock(|| { });
            c.send(sibling_convos); // let parent wait on all children
            fail!();
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        let mut r = p.recv();
        for p in r.mut_iter() { p.recv(); } // wait on all its siblings
        m.lock_cond(|cond| {
            let woken = cond.broadcast();
            assert_eq!(woken, 0);
        })
    }
    #[test]
    fn test_mutex_cond_signal_on_0() {
        // Tests that signal_on(0) is equivalent to signal().
        let m = Mutex::new();
        m.lock_cond(|cond| {
            let m2 = m.clone();
            do task::spawn {
                m2.lock_cond(|cond| {
                    cond.signal_on(0);
                })
            }
            cond.wait();
        })
    }
    #[test]
    #[ignore(reason = "linked failure?")]
    fn test_mutex_different_conds() {
        let result = do task::try {
            let m = Mutex::new_with_condvars(2);
            let m2 = m.clone();
            let (p, c) = Chan::new();
            do task::spawn {
                m2.lock_cond(|cond| {
                    c.send(());
                    cond.wait_on(1);
                })
            }
            let _ = p.recv();
            m.lock_cond(|cond| {
                if !cond.signal_on(0) {
                    fail!(); // success; punt sibling awake.
                }
            })
        };
        assert!(result.is_err());
    }
    #[test]
    fn test_mutex_no_condvars() {
        let result = do task::try {
            let m = Mutex::new_with_condvars(0);
            m.lock_cond(|cond| { cond.wait(); })
        };
        assert!(result.is_err());
        let result = do task::try {
            let m = Mutex::new_with_condvars(0);
            m.lock_cond(|cond| { cond.signal(); })
        };
        assert!(result.is_err());
        let result = do task::try {
            let m = Mutex::new_with_condvars(0);
            m.lock_cond(|cond| { cond.broadcast(); })
        };
        assert!(result.is_err());
    }
    /************************************************************************
     * Reader/writer lock tests
     ************************************************************************/
    #[cfg(test)]
    pub enum RWLockMode { Read, Write, Downgrade, DowngradeRead }
    #[cfg(test)]
    fn lock_rwlock_in_mode(x: &RWLock, mode: RWLockMode, blk: ||) {
        match mode {
            Read => x.read(blk),
            Write => x.write(blk),
            Downgrade =>
                x.write_downgrade(|mode| {
                    mode.write(|| { blk() });
                }),
            DowngradeRead =>
                x.write_downgrade(|mode| {
                    let mode = x.downgrade(mode);
                    mode.read(|| { blk() });
                }),
        }
    }
    #[cfg(test)]
    fn test_rwlock_exclusion(x: &RWLock,
                                 mode1: RWLockMode,
                                 mode2: RWLockMode) {
        // Test mutual exclusion between readers and writers. Just like the
        // mutex mutual exclusion test, a ways above.
        let (p, c) = Chan::new();
        let x2 = x.clone();
        let mut sharedstate = ~0;
        {
            let ptr: *int = &*sharedstate;
            do task::spawn {
                let sharedstate: &mut int =
                    unsafe { cast::transmute(ptr) };
                access_shared(sharedstate, &x2, mode1, 10);
                c.send(());
            }
        }
        {
            access_shared(sharedstate, x, mode2, 10);
            let _ = p.recv();

            assert_eq!(*sharedstate, 20);
        }

        fn access_shared(sharedstate: &mut int, x: &RWLock, mode: RWLockMode,
                         n: uint) {
            n.times(|| {
                lock_rwlock_in_mode(x, mode, || {
                    let oldval = *sharedstate;
                    task::deschedule();
                    *sharedstate = oldval + 1;
                })
            })
        }
    }
    #[test]
    fn test_rwlock_readers_wont_modify_the_data() {
        test_rwlock_exclusion(&RWLock::new(), Read, Write);
        test_rwlock_exclusion(&RWLock::new(), Write, Read);
        test_rwlock_exclusion(&RWLock::new(), Read, Downgrade);
        test_rwlock_exclusion(&RWLock::new(), Downgrade, Read);
    }
    #[test]
    fn test_rwlock_writers_and_writers() {
        test_rwlock_exclusion(&RWLock::new(), Write, Write);
        test_rwlock_exclusion(&RWLock::new(), Write, Downgrade);
        test_rwlock_exclusion(&RWLock::new(), Downgrade, Write);
        test_rwlock_exclusion(&RWLock::new(), Downgrade, Downgrade);
    }
    #[cfg(test)]
    fn test_rwlock_handshake(x: &RWLock,
                                 mode1: RWLockMode,
                                 mode2: RWLockMode,
                                 make_mode2_go_first: bool) {
        // Much like sem_multi_resource.
        let x2 = x.clone();
        let (p1, c1) = Chan::new();
        let (p2, c2) = Chan::new();
        do task::spawn {
            if !make_mode2_go_first {
                let _ = p2.recv(); // parent sends to us once it locks, or ...
            }
            lock_rwlock_in_mode(&x2, mode2, || {
                if make_mode2_go_first {
                    c1.send(()); // ... we send to it once we lock
                }
                let _ = p2.recv();
                c1.send(());
            })
        }
        if make_mode2_go_first {
            let _ = p1.recv(); // child sends to us once it locks, or ...
        }
        lock_rwlock_in_mode(x, mode1, || {
            if !make_mode2_go_first {
                c2.send(()); // ... we send to it once we lock
            }
            c2.send(());
            let _ = p1.recv();
        })
    }
    #[test]
    fn test_rwlock_readers_and_readers() {
        test_rwlock_handshake(&RWLock::new(), Read, Read, false);
        // The downgrader needs to get in before the reader gets in, otherwise
        // they cannot end up reading at the same time.
        test_rwlock_handshake(&RWLock::new(), DowngradeRead, Read, false);
        test_rwlock_handshake(&RWLock::new(), Read, DowngradeRead, true);
        // Two downgrade_reads can never both end up reading at the same time.
    }
    #[test]
    fn test_rwlock_downgrade_unlock() {
        // Tests that downgrade can unlock the lock in both modes
        let x = RWLock::new();
        lock_rwlock_in_mode(&x, Downgrade, || { });
        test_rwlock_handshake(&x, Read, Read, false);
        let y = RWLock::new();
        lock_rwlock_in_mode(&y, DowngradeRead, || { });
        test_rwlock_exclusion(&y, Write, Write);
    }
    #[test]
    fn test_rwlock_read_recursive() {
        let x = RWLock::new();
        x.read(|| { x.read(|| { }) })
    }
    #[test]
    fn test_rwlock_cond_wait() {
        // As test_mutex_cond_wait above.
        let x = RWLock::new();

        // Child wakes up parent
        x.write_cond(|cond| {
            let x2 = x.clone();
            do task::spawn {
                x2.write_cond(|cond| {
                    let woken = cond.signal();
                    assert!(woken);
                })
            }
            cond.wait();
        });
        // Parent wakes up child
        let (port, chan) = Chan::new();
        let x3 = x.clone();
        do task::spawn {
            x3.write_cond(|cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            })
        }
        let _ = port.recv(); // Wait until child gets in the rwlock
        x.read(|| { }); // Must be able to get in as a reader in the meantime
        x.write_cond(|cond| { // Or as another writer
            let woken = cond.signal();
            assert!(woken);
        });
        let _ = port.recv(); // Wait until child wakes up
        x.read(|| { }); // Just for good measure
    }
    #[cfg(test)]
    fn test_rwlock_cond_broadcast_helper(num_waiters: uint,
                                             dg1: bool,
                                             dg2: bool) {
        // Much like the mutex broadcast test. Downgrade-enabled.
        fn lock_cond(x: &RWLock, downgrade: bool, blk: |c: &Condvar|) {
            if downgrade {
                x.write_downgrade(|mode| {
                    mode.write_cond(|c| { blk(c) });
                });
            } else {
                x.write_cond(|c| { blk(c) });
            }
        }
        let x = RWLock::new();
        let mut ports = ~[];

        num_waiters.times(|| {
            let xi = x.clone();
            let (port, chan) = Chan::new();
            ports.push(port);
            do task::spawn {
                lock_cond(&xi, dg1, |cond| {
                    chan.send(());
                    cond.wait();
                    chan.send(());
                })
            }
        });

        // wait until all children get in the mutex
        for port in ports.mut_iter() { let _ = port.recv(); }
        lock_cond(&x, dg2, |cond| {
            let num_woken = cond.broadcast();
            assert_eq!(num_woken, num_waiters);
        });
        // wait until all children wake up
        for port in ports.mut_iter() { let _ = port.recv(); }
    }
    #[test]
    fn test_rwlock_cond_broadcast() {
        test_rwlock_cond_broadcast_helper(0, true, true);
        test_rwlock_cond_broadcast_helper(0, true, false);
        test_rwlock_cond_broadcast_helper(0, false, true);
        test_rwlock_cond_broadcast_helper(0, false, false);
        test_rwlock_cond_broadcast_helper(12, true, true);
        test_rwlock_cond_broadcast_helper(12, true, false);
        test_rwlock_cond_broadcast_helper(12, false, true);
        test_rwlock_cond_broadcast_helper(12, false, false);
    }
    #[cfg(test)]
    fn rwlock_kill_helper(mode1: RWLockMode, mode2: RWLockMode) {
        // Mutex must get automatically unlocked if failed/killed within.
        let x = RWLock::new();
        let x2 = x.clone();

        let result: result::Result<(), ~Any> = do task::try || {
            lock_rwlock_in_mode(&x2, mode1, || {
                fail!();
            })
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        lock_rwlock_in_mode(&x, mode2, || { })
    }
    #[test]
    fn test_rwlock_reader_killed_writer() {
        rwlock_kill_helper(Read, Write);
    }
    #[test]
    fn test_rwlock_writer_killed_reader() {
        rwlock_kill_helper(Write, Read);
    }
    #[test]
    fn test_rwlock_reader_killed_reader() {
        rwlock_kill_helper(Read, Read);
    }
    #[test]
    fn test_rwlock_writer_killed_writer() {
        rwlock_kill_helper(Write, Write);
    }
    #[test]
    fn test_rwlock_kill_downgrader() {
        rwlock_kill_helper(Downgrade, Read);
        rwlock_kill_helper(Read, Downgrade);
        rwlock_kill_helper(Downgrade, Write);
        rwlock_kill_helper(Write, Downgrade);
        rwlock_kill_helper(DowngradeRead, Read);
        rwlock_kill_helper(Read, DowngradeRead);
        rwlock_kill_helper(DowngradeRead, Write);
        rwlock_kill_helper(Write, DowngradeRead);
        rwlock_kill_helper(DowngradeRead, Downgrade);
        rwlock_kill_helper(DowngradeRead, Downgrade);
        rwlock_kill_helper(Downgrade, DowngradeRead);
        rwlock_kill_helper(Downgrade, DowngradeRead);
    }
    #[test] #[should_fail]
    fn test_rwlock_downgrade_cant_swap() {
        // Tests that you can't downgrade with a different rwlock's token.
        let x = RWLock::new();
        let y = RWLock::new();
        x.write_downgrade(|xwrite| {
            let mut xopt = Some(xwrite);
            y.write_downgrade(|_ywrite| {
                y.downgrade(xopt.take_unwrap());
                error!("oops, y.downgrade(x) should have failed!");
            })
        })
    }
}
