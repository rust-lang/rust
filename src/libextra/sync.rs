// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */

use core::prelude::*;

use core::comm;
use core::ptr;
use core::task;
use core::unstable::sync::{Exclusive, exclusive};
use core::util;

/****************************************************************************
 * Internals
 ****************************************************************************/

// Each waiting task receives on one of these.
#[doc(hidden)]
type WaitEnd = comm::PortOne<()>;
#[doc(hidden)]
type SignalEnd = comm::ChanOne<()>;
// A doubly-ended queue of waiting tasks.
#[doc(hidden)]
struct Waitqueue { head: comm::Port<SignalEnd>,
                   tail: comm::Chan<SignalEnd> }

fn new_waitqueue() -> Waitqueue {
    let (block_head, block_tail) = comm::stream();
    Waitqueue { head: block_head, tail: block_tail }
}

// Signals one live task from the queue.
#[doc(hidden)]
fn signal_waitqueue(q: &Waitqueue) -> bool {
    // The peek is mandatory to make sure recv doesn't block.
    if q.head.peek() {
        // Pop and send a wakeup signal. If the waiter was killed, its port
        // will have closed. Keep trying until we get a live task.
        if comm::try_send_one(q.head.recv(), ()) {
            true
        } else {
            signal_waitqueue(q)
        }
    } else {
        false
    }
}

#[doc(hidden)]
fn broadcast_waitqueue(q: &Waitqueue) -> uint {
    let mut count = 0;
    while q.head.peek() {
        if comm::try_send_one(q.head.recv(), ()) {
            count += 1;
        }
    }
    count
}

// The building-block used to make semaphores, mutexes, and rwlocks.
#[doc(hidden)]
struct SemInner<Q> {
    count: int,
    waiters:   Waitqueue,
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked:   Q
}

#[doc(hidden)]
struct Sem<Q>(Exclusive<SemInner<Q>>);

#[doc(hidden)]
fn new_sem<Q:Owned>(count: int, q: Q) -> Sem<Q> {
    Sem(exclusive(SemInner {
        count: count, waiters: new_waitqueue(), blocked: q }))
}
#[doc(hidden)]
fn new_sem_and_signal(count: int, num_condvars: uint)
        -> Sem<~[Waitqueue]> {
    let mut queues = ~[];
    for num_condvars.times {
        queues.push(new_waitqueue());
    }
    new_sem(count, queues)
}

#[doc(hidden)]
pub impl<Q:Owned> Sem<Q> {
    fn acquire(&self) {
        unsafe {
            let mut waiter_nobe = None;
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    // Create waiter nobe.
                    let (WaitEnd, SignalEnd) = comm::oneshot();
                    // Tell outer scope we need to block.
                    waiter_nobe = Some(WaitEnd);
                    // Enqueue ourself.
                    state.waiters.tail.send(SignalEnd);
                }
            }
            // Uncomment if you wish to test for sem races. Not valgrind-friendly.
            /* for 1000.times { task::yield(); } */
            // Need to wait outside the exclusive.
            if waiter_nobe.is_some() {
                let _ = comm::recv_one(waiter_nobe.unwrap());
            }
        }
    }
    fn release(&self) {
        unsafe {
            do (**self).with |state| {
                state.count += 1;
                if state.count <= 0 {
                    signal_waitqueue(&state.waiters);
                }
            }
        }
    }
}
// FIXME(#3154) move both copies of this into Sem<Q>, and unify the 2 structs
#[doc(hidden)]
pub impl Sem<()> {
    fn access<U>(&self, blk: &fn() -> U) -> U {
        let mut release = None;
        unsafe {
            do task::unkillable {
                self.acquire();
                release = Some(SemRelease(self));
            }
        }
        blk()
    }
}
#[doc(hidden)]
pub impl Sem<~[Waitqueue]> {
    fn access<U>(&self, blk: &fn() -> U) -> U {
        let mut release = None;
        unsafe {
            do task::unkillable {
                self.acquire();
                release = Some(SemAndSignalRelease(self));
            }
        }
        blk()
    }
}

// FIXME(#3588) should go inside of access()
#[doc(hidden)]
type SemRelease<'self> = SemReleaseGeneric<'self, ()>;
type SemAndSignalRelease<'self> = SemReleaseGeneric<'self, ~[Waitqueue]>;
struct SemReleaseGeneric<'self, Q> { sem: &'self Sem<Q> }

#[unsafe_destructor]
impl<'self, Q:Owned> Drop for SemReleaseGeneric<'self, Q> {
    fn finalize(&self) {
        self.sem.release();
    }
}

fn SemRelease<'r>(sem: &'r Sem<()>) -> SemRelease<'r> {
    SemReleaseGeneric {
        sem: sem
    }
}

fn SemAndSignalRelease<'r>(sem: &'r Sem<~[Waitqueue]>)
                        -> SemAndSignalRelease<'r> {
    SemReleaseGeneric {
        sem: sem
    }
}

/// A mechanism for atomic-unlock-and-deschedule blocking and signalling.
pub struct Condvar<'self> { priv sem: &'self Sem<~[Waitqueue]> }

#[unsafe_destructor]
impl<'self> Drop for Condvar<'self> { fn finalize(&self) {} }

pub impl<'self> Condvar<'self> {
    /**
     * Atomically drop the associated lock, and block until a signal is sent.
     *
     * # Failure
     * A task which is killed (i.e., by linked failure with another task)
     * while waiting on a condition variable will wake up, fail, and unlock
     * the associated lock as it unwinds.
     */
    fn wait(&self) { self.wait_on(0) }

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
    fn wait_on(&self, condvar_id: uint) {
        // Create waiter nobe.
        let (WaitEnd, SignalEnd) = comm::oneshot();
        let mut WaitEnd   = Some(WaitEnd);
        let mut SignalEnd = Some(SignalEnd);
        let mut reacquire = None;
        let mut out_of_bounds = None;
        unsafe {
            do task::unkillable {
                // Release lock, 'atomically' enqueuing ourselves in so doing.
                do (**self.sem).with |state| {
                    if condvar_id < state.blocked.len() {
                        // Drop the lock.
                        state.count += 1;
                        if state.count <= 0 {
                            signal_waitqueue(&state.waiters);
                        }
                        // Enqueue ourself to be woken up by a signaller.
                        let SignalEnd = SignalEnd.swap_unwrap();
                        state.blocked[condvar_id].tail.send(SignalEnd);
                    } else {
                        out_of_bounds = Some(state.blocked.len());
                    }
                }

                // If yield checks start getting inserted anywhere, we can be
                // killed before or after enqueueing. Deciding whether to
                // unkillably reacquire the lock needs to happen atomically
                // wrt enqueuing.
                if out_of_bounds.is_none() {
                    reacquire = Some(SemAndSignalReacquire(self.sem));
                }
            }
        }
        do check_cvar_bounds(out_of_bounds, condvar_id, "cond.wait_on()") {
            // Unconditionally "block". (Might not actually block if a
            // signaller already sent -- I mean 'unconditionally' in contrast
            // with acquire().)
            let _ = comm::recv_one(WaitEnd.swap_unwrap());
        }

        // This is needed for a failing condition variable to reacquire the
        // mutex during unwinding. As long as the wrapper (mutex, etc) is
        // bounded in when it gets released, this shouldn't hang forever.
        struct SemAndSignalReacquire<'self> {
            sem: &'self Sem<~[Waitqueue]>,
        }

        #[unsafe_destructor]
        impl<'self> Drop for SemAndSignalReacquire<'self> {
            fn finalize(&self) {
                unsafe {
                    // Needs to succeed, instead of itself dying.
                    do task::unkillable {
                        self.sem.acquire();
                    }
                }
            }
        }

        fn SemAndSignalReacquire<'r>(sem: &'r Sem<~[Waitqueue]>)
                                  -> SemAndSignalReacquire<'r> {
            SemAndSignalReacquire {
                sem: sem
            }
        }
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    fn signal(&self) -> bool { self.signal_on(0) }

    /// As signal, but with a specified condvar_id. See wait_on.
    fn signal_on(&self, condvar_id: uint) -> bool {
        unsafe {
            let mut out_of_bounds = None;
            let mut result = false;
            do (**self.sem).with |state| {
                if condvar_id < state.blocked.len() {
                    result = signal_waitqueue(&state.blocked[condvar_id]);
                } else {
                    out_of_bounds = Some(state.blocked.len());
                }
            }
            do check_cvar_bounds(out_of_bounds, condvar_id, "cond.signal_on()") {
                result
            }
        }
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    fn broadcast(&self) -> uint { self.broadcast_on(0) }

    /// As broadcast, but with a specified condvar_id. See wait_on.
    fn broadcast_on(&self, condvar_id: uint) -> uint {
        let mut out_of_bounds = None;
        let mut queue = None;
        unsafe {
            do (**self.sem).with |state| {
                if condvar_id < state.blocked.len() {
                    // To avoid :broadcast_heavy, we make a new waitqueue,
                    // swap it out with the old one, and broadcast on the
                    // old one outside of the little-lock.
                    queue = Some(util::replace(&mut state.blocked[condvar_id],
                                               new_waitqueue()));
                } else {
                    out_of_bounds = Some(state.blocked.len());
                }
            }
            do check_cvar_bounds(out_of_bounds, condvar_id, "cond.signal_on()") {
                let queue = queue.swap_unwrap();
                broadcast_waitqueue(&queue)
            }
        }
    }
}

// Checks whether a condvar ID was out of bounds, and fails if so, or does
// something else next on success.
#[inline(always)]
#[doc(hidden)]
fn check_cvar_bounds<U>(out_of_bounds: Option<uint>, id: uint, act: &str,
                        blk: &fn() -> U) -> U {
    match out_of_bounds {
        Some(0) =>
            fail!("%s with illegal ID %u - this lock has no condvars!", act, id),
        Some(length) =>
            fail!("%s with illegal ID %u - ID must be less than %u", act, id, length),
        None => blk()
    }
}

#[doc(hidden)]
pub impl Sem<~[Waitqueue]> {
    // The only other place that condvars get built is rwlock_write_mode.
    fn access_cond<U>(&self, blk: &fn(c: &Condvar) -> U) -> U {
        do self.access { blk(&Condvar { sem: self }) }
    }
}

/****************************************************************************
 * Semaphores
 ****************************************************************************/

/// A counting, blocking, bounded-waiting semaphore.
struct Semaphore { priv sem: Sem<()> }

/// Create a new semaphore with the specified count.
pub fn semaphore(count: int) -> Semaphore {
    Semaphore { sem: new_sem(count, ()) }
}

impl Clone for Semaphore {
    /// Create a new handle to the semaphore.
    fn clone(&self) -> Semaphore {
        Semaphore { sem: Sem((*self.sem).clone()) }
    }
}

pub impl Semaphore {
    /**
     * Acquire a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    fn acquire(&self) { (&self.sem).acquire() }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist. Won't block the caller.
     */
    fn release(&self) { (&self.sem).release() }

    /// Run a function with ownership of one of the semaphore's resources.
    fn access<U>(&self, blk: &fn() -> U) -> U { (&self.sem).access(blk) }
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
pub struct Mutex { priv sem: Sem<~[Waitqueue]> }

/// Create a new mutex, with one associated condvar.
pub fn Mutex() -> Mutex { mutex_with_condvars(1) }
/**
 * Create a new mutex, with a specified number of associated condvars. This
 * will allow calling wait_on/signal_on/broadcast_on with condvar IDs between
 * 0 and num_condvars-1. (If num_condvars is 0, lock_cond will be allowed but
 * any operations on the condvar will fail.)
 */
pub fn mutex_with_condvars(num_condvars: uint) -> Mutex {
    Mutex { sem: new_sem_and_signal(1, num_condvars) }
}

impl Clone for Mutex {
    /// Create a new handle to the mutex.
    fn clone(&self) -> Mutex { Mutex { sem: Sem((*self.sem).clone()) } }
}

pub impl Mutex {
    /// Run a function with ownership of the mutex.
    fn lock<U>(&self, blk: &fn() -> U) -> U { (&self.sem).access(blk) }

    /// Run a function with ownership of the mutex and a handle to a condvar.
    fn lock_cond<U>(&self, blk: &fn(c: &Condvar) -> U) -> U {
        (&self.sem).access_cond(blk)
    }
}

/****************************************************************************
 * Reader-writer locks
 ****************************************************************************/

// NB: Wikipedia - Readers-writers_problem#The_third_readers-writers_problem

#[doc(hidden)]
struct RWlockInner {
    read_mode:  bool,
    read_count: uint
}

/**
 * A blocking, no-starvation, reader-writer lock with an associated condvar.
 *
 * # Failure
 * A task which fails while holding an rwlock will unlock the rwlock as it
 * unwinds.
 */
pub struct RWlock {
    priv order_lock:  Semaphore,
    priv access_lock: Sem<~[Waitqueue]>,
    priv state:       Exclusive<RWlockInner>
}

/// Create a new rwlock, with one associated condvar.
pub fn RWlock() -> RWlock { rwlock_with_condvars(1) }

/**
 * Create a new rwlock, with a specified number of associated condvars.
 * Similar to mutex_with_condvars.
 */
pub fn rwlock_with_condvars(num_condvars: uint) -> RWlock {
    RWlock { order_lock: semaphore(1),
             access_lock: new_sem_and_signal(1, num_condvars),
             state: exclusive(RWlockInner { read_mode:  false,
                                             read_count: 0 }) }
}

pub impl RWlock {
    /// Create a new handle to the rwlock.
    fn clone(&self) -> RWlock {
        RWlock { order_lock:  (&(self.order_lock)).clone(),
                 access_lock: Sem((*self.access_lock).clone()),
                 state:       self.state.clone() }
    }

    /**
     * Run a function with the rwlock in read mode. Calls to 'read' from other
     * tasks may run concurrently with this one.
     */
    fn read<U>(&self, blk: &fn() -> U) -> U {
        let mut release = None;
        unsafe {
            do task::unkillable {
                do (&self.order_lock).access {
                    let mut first_reader = false;
                    do self.state.with |state| {
                        first_reader = (state.read_count == 0);
                        state.read_count += 1;
                    }
                    if first_reader {
                        (&self.access_lock).acquire();
                        do self.state.with |state| {
                            // Must happen *after* getting access_lock. If
                            // this is set while readers are waiting, but
                            // while a writer holds the lock, the writer will
                            // be confused if they downgrade-then-unlock.
                            state.read_mode = true;
                        }
                    }
                }
                release = Some(RWlockReleaseRead(self));
            }
        }
        blk()
    }

    /**
     * Run a function with the rwlock in write mode. No calls to 'read' or
     * 'write' from other tasks will run concurrently with this one.
     */
    fn write<U>(&self, blk: &fn() -> U) -> U {
        unsafe {
            do task::unkillable {
                (&self.order_lock).acquire();
                do (&self.access_lock).access {
                    (&self.order_lock).release();
                    task::rekillable(blk)
                }
            }
        }
    }

    /**
     * As write(), but also with a handle to a condvar. Waiting on this
     * condvar will allow readers and writers alike to take the rwlock before
     * the waiting task is signalled. (Note: a writer that waited and then
     * was signalled might reacquire the lock before other waiting writers.)
     */
    fn write_cond<U>(&self, blk: &fn(c: &Condvar) -> U) -> U {
        // NB: You might think I should thread the order_lock into the cond
        // wait call, so that it gets waited on before access_lock gets
        // reacquired upon being woken up. However, (a) this would be not
        // pleasant to implement (and would mandate a new 'rw_cond' type) and
        // (b) I think violating no-starvation in that case is appropriate.
        unsafe {
            do task::unkillable {
                (&self.order_lock).acquire();
                do (&self.access_lock).access_cond |cond| {
                    (&self.order_lock).release();
                    do task::rekillable { blk(cond) }
                }
            }
        }
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
     * ~~~ {.rust}
     * do lock.write_downgrade |write_mode| {
     *     do (&write_mode).write_cond |condvar| {
     *         ... exclusive access ...
     *     }
     *     let read_mode = lock.downgrade(write_mode);
     *     do (&read_mode).read {
     *         ... shared access ...
     *     }
     * }
     * ~~~
     */
    fn write_downgrade<U>(&self, blk: &fn(v: RWlockWriteMode) -> U) -> U {
        // Implementation slightly different from the slicker 'write's above.
        // The exit path is conditional on whether the caller downgrades.
        let mut _release = None;
        unsafe {
            do task::unkillable {
                (&self.order_lock).acquire();
                (&self.access_lock).acquire();
                (&self.order_lock).release();
            }
            _release = Some(RWlockReleaseDowngrade(self));
        }
        blk(RWlockWriteMode { lock: self })
    }

    /// To be called inside of the write_downgrade block.
    fn downgrade<'a>(&self,
                     token: RWlockWriteMode<'a>)
                  -> RWlockReadMode<'a> {
        if !ptr::ref_eq(self, token.lock) {
            fail!("Can't downgrade() with a different rwlock's write_mode!");
        }
        unsafe {
            do task::unkillable {
                let mut first_reader = false;
                do self.state.with |state| {
                    assert!(!state.read_mode);
                    state.read_mode = true;
                    first_reader = (state.read_count == 0);
                    state.read_count += 1;
                }
                if !first_reader {
                    // Guaranteed not to let another writer in, because
                    // another reader was holding the order_lock. Hence they
                    // must be the one to get the access_lock (because all
                    // access_locks are acquired with order_lock held).
                    (&self.access_lock).release();
                }
            }
        }
        RWlockReadMode { lock: token.lock }
    }
}

// FIXME(#3588) should go inside of read()
#[doc(hidden)]
struct RWlockReleaseRead<'self> {
    lock: &'self RWlock,
}

#[unsafe_destructor]
impl<'self> Drop for RWlockReleaseRead<'self> {
    fn finalize(&self) {
        unsafe {
            do task::unkillable {
                let mut last_reader = false;
                do self.lock.state.with |state| {
                    assert!(state.read_mode);
                    assert!(state.read_count > 0);
                    state.read_count -= 1;
                    if state.read_count == 0 {
                        last_reader = true;
                        state.read_mode = false;
                    }
                }
                if last_reader {
                    (&self.lock.access_lock).release();
                }
            }
        }
    }
}

fn RWlockReleaseRead<'r>(lock: &'r RWlock) -> RWlockReleaseRead<'r> {
    RWlockReleaseRead {
        lock: lock
    }
}

// FIXME(#3588) should go inside of downgrade()
#[doc(hidden)]
#[unsafe_destructor]
struct RWlockReleaseDowngrade<'self> {
    lock: &'self RWlock,
}

#[unsafe_destructor]
impl<'self> Drop for RWlockReleaseDowngrade<'self> {
    fn finalize(&self) {
        unsafe {
            do task::unkillable {
                let mut writer_or_last_reader = false;
                do self.lock.state.with |state| {
                    if state.read_mode {
                        assert!(state.read_count > 0);
                        state.read_count -= 1;
                        if state.read_count == 0 {
                            // Case 1: Writer downgraded & was the last reader
                            writer_or_last_reader = true;
                            state.read_mode = false;
                        } else {
                            // Case 2: Writer downgraded & was not the last
                            // reader
                        }
                    } else {
                        // Case 3: Writer did not downgrade
                        writer_or_last_reader = true;
                    }
                }
                if writer_or_last_reader {
                    (&self.lock.access_lock).release();
                }
            }
        }
    }
}

fn RWlockReleaseDowngrade<'r>(lock: &'r RWlock)
                           -> RWlockReleaseDowngrade<'r> {
    RWlockReleaseDowngrade {
        lock: lock
    }
}

/// The "write permission" token used for rwlock.write_downgrade().
pub struct RWlockWriteMode<'self> { priv lock: &'self RWlock }
#[unsafe_destructor]
impl<'self> Drop for RWlockWriteMode<'self> { fn finalize(&self) {} }

/// The "read permission" token used for rwlock.write_downgrade().
pub struct RWlockReadMode<'self> { priv lock: &'self RWlock }
#[unsafe_destructor]
impl<'self> Drop for RWlockReadMode<'self> { fn finalize(&self) {} }

pub impl<'self> RWlockWriteMode<'self> {
    /// Access the pre-downgrade rwlock in write mode.
    fn write<U>(&self, blk: &fn() -> U) -> U { blk() }
    /// Access the pre-downgrade rwlock in write mode with a condvar.
    fn write_cond<U>(&self, blk: &fn(c: &Condvar) -> U) -> U {
        blk(&Condvar { sem: &self.lock.access_lock })
    }
}

pub impl<'self> RWlockReadMode<'self> {
    /// Access the post-downgrade rwlock in read mode.
    fn read<U>(&self, blk: &fn() -> U) -> U { blk() }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use core::prelude::*;

    use sync::*;

    use core::cast;
    use core::cell::Cell;
    use core::comm;
    use core::ptr;
    use core::result;
    use core::task;
    use core::vec;

    /************************************************************************
     * Semaphore tests
     ************************************************************************/
    #[test]
    fn test_sem_acquire_release() {
        let s = ~semaphore(1);
        s.acquire();
        s.release();
        s.acquire();
    }
    #[test]
    fn test_sem_basic() {
        let s = ~semaphore(1);
        do s.access { }
    }
    #[test]
    fn test_sem_as_mutex() {
        let s = ~semaphore(1);
        let s2 = ~s.clone();
        do task::spawn || {
            do s2.access {
                for 5.times { task::yield(); }
            }
        }
        do s.access {
            for 5.times { task::yield(); }
        }
    }
    #[test]
    fn test_sem_as_cvar() {
        /* Child waits and parent signals */
        let (p,c) = comm::stream();
        let s = ~semaphore(0);
        let s2 = ~s.clone();
        do task::spawn || {
            s2.acquire();
            c.send(());
        }
        for 5.times { task::yield(); }
        s.release();
        let _ = p.recv();

        /* Parent waits and child signals */
        let (p,c) = comm::stream();
        let s = ~semaphore(0);
        let s2 = ~s.clone();
        do task::spawn || {
            for 5.times { task::yield(); }
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
        let s = ~semaphore(2);
        let s2 = ~s.clone();
        let (p1,c1) = comm::stream();
        let (p2,c2) = comm::stream();
        do task::spawn || {
            do s2.access {
                let _ = p2.recv();
                c1.send(());
            }
        }
        do s.access {
            c2.send(());
            let _ = p1.recv();
        }
    }
    #[test]
    fn test_sem_runtime_friendly_blocking() {
        // Force the runtime to schedule two threads on the same sched_loop.
        // When one blocks, it should schedule the other one.
        do task::spawn_sched(task::ManualThreads(1)) {
            let s = ~semaphore(1);
            let s2 = ~s.clone();
            let (p,c) = comm::stream();
            let child_data = Cell((s2, c));
            do s.access {
                let (s2, c) = child_data.take();
                do task::spawn || {
                    c.send(());
                    do s2.access { }
                    c.send(());
                }
                let _ = p.recv(); // wait for child to come alive
                for 5.times { task::yield(); } // let the child contend
            }
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
        let (p,c) = comm::stream();
        let m = ~Mutex();
        let m2 = m.clone();
        let mut sharedstate = ~0;
        {
            let ptr: *int = &*sharedstate;
            do task::spawn || {
                let sharedstate: &mut int =
                    unsafe { cast::transmute(ptr) };
                access_shared(sharedstate, m2, 10);
                c.send(());

            }
        }
        {
            access_shared(sharedstate, m, 10);
            let _ = p.recv();

            assert_eq!(*sharedstate, 20);
        }

        fn access_shared(sharedstate: &mut int, m: &Mutex, n: uint) {
            for n.times {
                do m.lock {
                    let oldval = *sharedstate;
                    task::yield();
                    *sharedstate = oldval + 1;
                }
            }
        }
    }
    #[test]
    fn test_mutex_cond_wait() {
        let m = ~Mutex();

        // Child wakes up parent
        do m.lock_cond |cond| {
            let m2 = ~m.clone();
            do task::spawn || {
                do m2.lock_cond |cond| {
                    let woken = cond.signal();
                    assert!(woken);
                }
            }
            cond.wait();
        }
        // Parent wakes up child
        let (port,chan) = comm::stream();
        let m3 = ~m.clone();
        do task::spawn || {
            do m3.lock_cond |cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            }
        }
        let _ = port.recv(); // Wait until child gets in the mutex
        do m.lock_cond |cond| {
            let woken = cond.signal();
            assert!(woken);
        }
        let _ = port.recv(); // Wait until child wakes up
    }
    #[cfg(test)]
    fn test_mutex_cond_broadcast_helper(num_waiters: uint) {
        let m = ~Mutex();
        let mut ports = ~[];

        for num_waiters.times {
            let mi = ~m.clone();
            let (port, chan) = comm::stream();
            ports.push(port);
            do task::spawn || {
                do mi.lock_cond |cond| {
                    chan.send(());
                    cond.wait();
                    chan.send(());
                }
            }
        }

        // wait until all children get in the mutex
        for ports.each |port| { let _ = port.recv(); }
        do m.lock_cond |cond| {
            let num_woken = cond.broadcast();
            assert_eq!(num_woken, num_waiters);
        }
        // wait until all children wake up
        for ports.each |port| { let _ = port.recv(); }
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
        let m = ~Mutex();
        let m2 = ~m.clone();
        do task::try || {
            do m.lock_cond |_x| { }
        };
        do m2.lock_cond |cond| {
            assert!(!cond.signal());
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_killed_simple() {
        // Mutex must get automatically unlocked if failed/killed within.
        let m = ~Mutex();
        let m2 = ~m.clone();

        let result: result::Result<(),()> = do task::try || {
            do m2.lock {
                fail!();
            }
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        do m.lock { }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_killed_cond() {
        // Getting killed during cond wait must not corrupt the mutex while
        // unwinding (e.g. double unlock).
        let m = ~Mutex();
        let m2 = ~m.clone();

        let result: result::Result<(),()> = do task::try || {
            let (p,c) = comm::stream();
            do task::spawn || { // linked
                let _ = p.recv(); // wait for sibling to get in the mutex
                task::yield();
                fail!();
            }
            do m2.lock_cond |cond| {
                c.send(()); // tell sibling go ahead
                cond.wait(); // block forever
            }
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        do m.lock_cond |cond| {
            let woken = cond.signal();
            assert!(!woken);
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_killed_broadcast() {
        let m = ~Mutex();
        let m2 = ~m.clone();
        let (p,c) = comm::stream();

        let result: result::Result<(),()> = do task::try || {
            let mut sibling_convos = ~[];
            for 2.times {
                let (p,c) = comm::stream();
                let c = Cell(c);
                sibling_convos.push(p);
                let mi = ~m2.clone();
                // spawn sibling task
                do task::spawn { // linked
                    do mi.lock_cond |cond| {
                        let c = c.take();
                        c.send(()); // tell sibling to go ahead
                        let _z = SendOnFailure(c);
                        cond.wait(); // block forever
                    }
                }
            }
            for sibling_convos.each |p| {
                let _ = p.recv(); // wait for sibling to get in the mutex
            }
            do m2.lock { }
            c.send(sibling_convos); // let parent wait on all children
            fail!();
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        for vec::each(p.recv()) |p| { p.recv(); } // wait on all its siblings
        do m.lock_cond |cond| {
            let woken = cond.broadcast();
            assert_eq!(woken, 0);
        }
        struct SendOnFailure {
            c: comm::Chan<()>,
        }

        impl Drop for SendOnFailure {
            fn finalize(&self) {
                self.c.send(());
            }
        }

        fn SendOnFailure(c: comm::Chan<()>) -> SendOnFailure {
            SendOnFailure {
                c: c
            }
        }
    }
    #[test]
    fn test_mutex_cond_signal_on_0() {
        // Tests that signal_on(0) is equivalent to signal().
        let m = ~Mutex();
        do m.lock_cond |cond| {
            let m2 = ~m.clone();
            do task::spawn || {
                do m2.lock_cond |cond| {
                    cond.signal_on(0);
                }
            }
            cond.wait();
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_different_conds() {
        let result = do task::try {
            let m = ~mutex_with_condvars(2);
            let m2 = ~m.clone();
            let (p,c) = comm::stream();
            do task::spawn || {
                do m2.lock_cond |cond| {
                    c.send(());
                    cond.wait_on(1);
                }
            }
            let _ = p.recv();
            do m.lock_cond |cond| {
                if !cond.signal_on(0) {
                    fail!(); // success; punt sibling awake.
                }
            }
        };
        assert!(result.is_err());
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_no_condvars() {
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.wait(); }
        };
        assert!(result.is_err());
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.signal(); }
        };
        assert!(result.is_err());
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.broadcast(); }
        };
        assert!(result.is_err());
    }
    /************************************************************************
     * Reader/writer lock tests
     ************************************************************************/
    #[cfg(test)]
    pub enum RWlockMode { Read, Write, Downgrade, DowngradeRead }
    #[cfg(test)]
    fn lock_rwlock_in_mode(x: &RWlock, mode: RWlockMode, blk: &fn()) {
        match mode {
            Read => x.read(blk),
            Write => x.write(blk),
            Downgrade =>
                do x.write_downgrade |mode| {
                    (&mode).write(blk);
                },
            DowngradeRead =>
                do x.write_downgrade |mode| {
                    let mode = x.downgrade(mode);
                    (&mode).read(blk);
                },
        }
    }
    #[cfg(test)]
    fn test_rwlock_exclusion(x: ~RWlock,
                                 mode1: RWlockMode,
                                 mode2: RWlockMode) {
        // Test mutual exclusion between readers and writers. Just like the
        // mutex mutual exclusion test, a ways above.
        let (p,c) = comm::stream();
        let x2 = (*x).clone();
        let mut sharedstate = ~0;
        {
            let ptr: *int = &*sharedstate;
            do task::spawn || {
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

        fn access_shared(sharedstate: &mut int, x: &RWlock, mode: RWlockMode,
                         n: uint) {
            for n.times {
                do lock_rwlock_in_mode(x, mode) {
                    let oldval = *sharedstate;
                    task::yield();
                    *sharedstate = oldval + 1;
                }
            }
        }
    }
    #[test]
    fn test_rwlock_readers_wont_modify_the_data() {
        test_rwlock_exclusion(~RWlock(), Read, Write);
        test_rwlock_exclusion(~RWlock(), Write, Read);
        test_rwlock_exclusion(~RWlock(), Read, Downgrade);
        test_rwlock_exclusion(~RWlock(), Downgrade, Read);
    }
    #[test]
    fn test_rwlock_writers_and_writers() {
        test_rwlock_exclusion(~RWlock(), Write, Write);
        test_rwlock_exclusion(~RWlock(), Write, Downgrade);
        test_rwlock_exclusion(~RWlock(), Downgrade, Write);
        test_rwlock_exclusion(~RWlock(), Downgrade, Downgrade);
    }
    #[cfg(test)]
    fn test_rwlock_handshake(x: ~RWlock,
                                 mode1: RWlockMode,
                                 mode2: RWlockMode,
                                 make_mode2_go_first: bool) {
        // Much like sem_multi_resource.
        let x2 = (*x).clone();
        let (p1,c1) = comm::stream();
        let (p2,c2) = comm::stream();
        do task::spawn || {
            if !make_mode2_go_first {
                let _ = p2.recv(); // parent sends to us once it locks, or ...
            }
            do lock_rwlock_in_mode(&x2, mode2) {
                if make_mode2_go_first {
                    c1.send(()); // ... we send to it once we lock
                }
                let _ = p2.recv();
                c1.send(());
            }
        }
        if make_mode2_go_first {
            let _ = p1.recv(); // child sends to us once it locks, or ...
        }
        do lock_rwlock_in_mode(x, mode1) {
            if !make_mode2_go_first {
                c2.send(()); // ... we send to it once we lock
            }
            c2.send(());
            let _ = p1.recv();
        }
    }
    #[test]
    fn test_rwlock_readers_and_readers() {
        test_rwlock_handshake(~RWlock(), Read, Read, false);
        // The downgrader needs to get in before the reader gets in, otherwise
        // they cannot end up reading at the same time.
        test_rwlock_handshake(~RWlock(), DowngradeRead, Read, false);
        test_rwlock_handshake(~RWlock(), Read, DowngradeRead, true);
        // Two downgrade_reads can never both end up reading at the same time.
    }
    #[test]
    fn test_rwlock_downgrade_unlock() {
        // Tests that downgrade can unlock the lock in both modes
        let x = ~RWlock();
        do lock_rwlock_in_mode(x, Downgrade) { }
        test_rwlock_handshake(x, Read, Read, false);
        let y = ~RWlock();
        do lock_rwlock_in_mode(y, DowngradeRead) { }
        test_rwlock_exclusion(y, Write, Write);
    }
    #[test]
    fn test_rwlock_read_recursive() {
        let x = ~RWlock();
        do x.read { do x.read { } }
    }
    #[test]
    fn test_rwlock_cond_wait() {
        // As test_mutex_cond_wait above.
        let x = ~RWlock();

        // Child wakes up parent
        do x.write_cond |cond| {
            let x2 = (*x).clone();
            do task::spawn || {
                do x2.write_cond |cond| {
                    let woken = cond.signal();
                    assert!(woken);
                }
            }
            cond.wait();
        }
        // Parent wakes up child
        let (port,chan) = comm::stream();
        let x3 = (*x).clone();
        do task::spawn || {
            do x3.write_cond |cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            }
        }
        let _ = port.recv(); // Wait until child gets in the rwlock
        do x.read { } // Must be able to get in as a reader in the meantime
        do x.write_cond |cond| { // Or as another writer
            let woken = cond.signal();
            assert!(woken);
        }
        let _ = port.recv(); // Wait until child wakes up
        do x.read { } // Just for good measure
    }
    #[cfg(test)]
    fn test_rwlock_cond_broadcast_helper(num_waiters: uint,
                                             dg1: bool,
                                             dg2: bool) {
        // Much like the mutex broadcast test. Downgrade-enabled.
        fn lock_cond(x: &RWlock, downgrade: bool, blk: &fn(c: &Condvar)) {
            if downgrade {
                do x.write_downgrade |mode| {
                    (&mode).write_cond(blk)
                }
            } else {
                x.write_cond(blk)
            }
        }
        let x = ~RWlock();
        let mut ports = ~[];

        for num_waiters.times {
            let xi = (*x).clone();
            let (port, chan) = comm::stream();
            ports.push(port);
            do task::spawn || {
                do lock_cond(&xi, dg1) |cond| {
                    chan.send(());
                    cond.wait();
                    chan.send(());
                }
            }
        }

        // wait until all children get in the mutex
        for ports.each |port| { let _ = port.recv(); }
        do lock_cond(x, dg2) |cond| {
            let num_woken = cond.broadcast();
            assert_eq!(num_woken, num_waiters);
        }
        // wait until all children wake up
        for ports.each |port| { let _ = port.recv(); }
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
    #[cfg(test)] #[ignore(cfg(windows))]
    fn rwlock_kill_helper(mode1: RWlockMode, mode2: RWlockMode) {
        // Mutex must get automatically unlocked if failed/killed within.
        let x = ~RWlock();
        let x2 = (*x).clone();

        let result: result::Result<(),()> = do task::try || {
            do lock_rwlock_in_mode(&x2, mode1) {
                fail!();
            }
        };
        assert!(result.is_err());
        // child task must have finished by the time try returns
        do lock_rwlock_in_mode(x, mode2) { }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_reader_killed_writer() {
        rwlock_kill_helper(Read, Write);
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_writer_killed_reader() {
        rwlock_kill_helper(Write,Read );
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_reader_killed_reader() {
        rwlock_kill_helper(Read, Read );
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_writer_killed_writer() {
        rwlock_kill_helper(Write,Write);
    }
    #[test] #[ignore(cfg(windows))]
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
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rwlock_downgrade_cant_swap() {
        // Tests that you can't downgrade with a different rwlock's token.
        let x = ~RWlock();
        let y = ~RWlock();
        do x.write_downgrade |xwrite| {
            let mut xopt = Some(xwrite);
            do y.write_downgrade |_ywrite| {
                y.downgrade(xopt.swap_unwrap());
                error!("oops, y.downgrade(x) should have failed!");
            }
        }
    }
}
