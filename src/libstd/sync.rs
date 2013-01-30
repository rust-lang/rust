// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */

use core::option;
use core::pipes;
use core::prelude::*;
use core::private::{Exclusive, exclusive};
use core::ptr;
use core::task;
use core::util;
use core::vec;

/****************************************************************************
 * Internals
 ****************************************************************************/

// Each waiting task receives on one of these.
#[doc(hidden)]
type WaitEnd = pipes::PortOne<()>;
#[doc(hidden)]
type SignalEnd = pipes::ChanOne<()>;
// A doubly-ended queue of waiting tasks.
#[doc(hidden)]
struct Waitqueue { head: pipes::Port<SignalEnd>,
                   tail: pipes::Chan<SignalEnd> }

fn new_waitqueue() -> Waitqueue {
    let (block_head, block_tail) = pipes::stream();
    Waitqueue { head: move block_head, tail: move block_tail }
}

// Signals one live task from the queue.
#[doc(hidden)]
fn signal_waitqueue(q: &Waitqueue) -> bool {
    // The peek is mandatory to make sure recv doesn't block.
    if q.head.peek() {
        // Pop and send a wakeup signal. If the waiter was killed, its port
        // will have closed. Keep trying until we get a live task.
        if pipes::try_send_one(q.head.recv(), ()) {
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
        if pipes::try_send_one(q.head.recv(), ()) {
            count += 1;
        }
    }
    count
}

// The building-block used to make semaphores, mutexes, and rwlocks.
#[doc(hidden)]
struct SemInner<Q> {
    mut count: int,
    waiters:   Waitqueue,
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked:   Q
}
#[doc(hidden)]
enum Sem<Q> = Exclusive<SemInner<Q>>;

#[doc(hidden)]
fn new_sem<Q: Owned>(count: int, q: Q) -> Sem<Q> {
    Sem(exclusive(SemInner {
        mut count: count, waiters: new_waitqueue(), blocked: move q }))
}
#[doc(hidden)]
fn new_sem_and_signal(count: int, num_condvars: uint)
        -> Sem<~[Waitqueue]> {
    let mut queues = ~[];
    for num_condvars.times {
        queues.push(new_waitqueue());
    }
    new_sem(count, vec::cast_to_mut(move queues))
}

#[doc(hidden)]
impl<Q: Owned> &Sem<Q> {
    fn acquire() {
        let mut waiter_nobe = None;
        unsafe {
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    // Create waiter nobe.
                    let (WaitEnd, SignalEnd) = pipes::oneshot();
                    // Tell outer scope we need to block.
                    waiter_nobe = Some(move WaitEnd);
                    // Enqueue ourself.
                    state.waiters.tail.send(move SignalEnd);
                }
            }
        }
        // Uncomment if you wish to test for sem races. Not valgrind-friendly.
        /* for 1000.times { task::yield(); } */
        // Need to wait outside the exclusive.
        if waiter_nobe.is_some() {
            let _ = pipes::recv_one(option::unwrap(move waiter_nobe));
        }
    }
    fn release() {
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
impl &Sem<()> {
    fn access<U>(blk: fn() -> U) -> U {
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
impl &Sem<~[Waitqueue]> {
    fn access<U>(blk: fn() -> U) -> U {
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
type SemRelease = SemReleaseGeneric<()>;
type SemAndSignalRelease = SemReleaseGeneric<~[Waitqueue]>;
struct SemReleaseGeneric<Q> { sem: &Sem<Q> }

impl<Q: Owned> SemReleaseGeneric<Q> : Drop {
    fn finalize(&self) {
        self.sem.release();
    }
}

fn SemRelease(sem: &r/Sem<()>) -> SemRelease/&r {
    SemReleaseGeneric {
        sem: sem
    }
}

fn SemAndSignalRelease(sem: &r/Sem<~[Waitqueue]>)
    -> SemAndSignalRelease/&r {
    SemReleaseGeneric {
        sem: sem
    }
}

/// A mechanism for atomic-unlock-and-deschedule blocking and signalling.
pub struct Condvar { priv sem: &Sem<~[Waitqueue]> }

impl Condvar : Drop { fn finalize(&self) {} }

impl &Condvar {
    /**
     * Atomically drop the associated lock, and block until a signal is sent.
     *
     * # Failure
     * A task which is killed (i.e., by linked failure with another task)
     * while waiting on a condition variable will wake up, fail, and unlock
     * the associated lock as it unwinds.
     */
    fn wait() { self.wait_on(0) }
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
    fn wait_on(condvar_id: uint) {
        // Create waiter nobe.
        let (WaitEnd, SignalEnd) = pipes::oneshot();
        let mut WaitEnd   = Some(move WaitEnd);
        let mut SignalEnd = Some(move SignalEnd);
        let mut reacquire = None;
        let mut out_of_bounds = None;
        unsafe {
            do task::unkillable {
                // Release lock, 'atomically' enqueuing ourselves in so doing.
                do (**self.sem).with |state| {
                    if condvar_id < vec::len(state.blocked) {
                        // Drop the lock.
                        state.count += 1;
                        if state.count <= 0 {
                            signal_waitqueue(&state.waiters);
                        }
                        // Enqueue ourself to be woken up by a signaller.
                        let SignalEnd = option::swap_unwrap(&mut SignalEnd);
                        state.blocked[condvar_id].tail.send(move SignalEnd);
                    } else {
                        out_of_bounds = Some(vec::len(state.blocked));
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
            let _ = pipes::recv_one(option::swap_unwrap(&mut WaitEnd));
        }

        // This is needed for a failing condition variable to reacquire the
        // mutex during unwinding. As long as the wrapper (mutex, etc) is
        // bounded in when it gets released, this shouldn't hang forever.
        struct SemAndSignalReacquire {
            sem: &Sem<~[Waitqueue]>,
        }

        impl SemAndSignalReacquire : Drop {
            fn finalize(&self) {
                unsafe {
                    // Needs to succeed, instead of itself dying.
                    do task::unkillable {
                        self.sem.acquire();
                    }
                }
            }
        }

        fn SemAndSignalReacquire(sem: &r/Sem<~[Waitqueue]>)
            -> SemAndSignalReacquire/&r {
            SemAndSignalReacquire {
                sem: sem
            }
        }
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    fn signal() -> bool { self.signal_on(0) }
    /// As signal, but with a specified condvar_id. See wait_on.
    fn signal_on(condvar_id: uint) -> bool {
        let mut out_of_bounds = None;
        let mut result = false;
        unsafe {
            do (**self.sem).with |state| {
                if condvar_id < vec::len(state.blocked) {
                    result = signal_waitqueue(&state.blocked[condvar_id]);
                } else {
                    out_of_bounds = Some(vec::len(state.blocked));
                }
            }
        }
        do check_cvar_bounds(out_of_bounds, condvar_id, "cond.signal_on()") {
            result
        }
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    fn broadcast() -> uint { self.broadcast_on(0) }
    /// As broadcast, but with a specified condvar_id. See wait_on.
    fn broadcast_on(condvar_id: uint) -> uint {
        let mut out_of_bounds = None;
        let mut queue = None;
        unsafe {
            do (**self.sem).with |state| {
                if condvar_id < vec::len(state.blocked) {
                    // To avoid :broadcast_heavy, we make a new waitqueue,
                    // swap it out with the old one, and broadcast on the
                    // old one outside of the little-lock.
                    queue = Some(util::replace(&mut state.blocked[condvar_id],
                                               new_waitqueue()));
                } else {
                    out_of_bounds = Some(vec::len(state.blocked));
                }
            }
        }
        do check_cvar_bounds(out_of_bounds, condvar_id, "cond.signal_on()") {
            let queue = option::swap_unwrap(&mut queue);
            broadcast_waitqueue(&queue)
        }
    }
}

// Checks whether a condvar ID was out of bounds, and fails if so, or does
// something else next on success.
#[inline(always)]
#[doc(hidden)]
fn check_cvar_bounds<U>(out_of_bounds: Option<uint>, id: uint, act: &str,
                        blk: fn() -> U) -> U {
    match out_of_bounds {
        Some(0) =>
            fail fmt!("%s with illegal ID %u - this lock has no condvars!",
                      act, id),
        Some(length) =>
            fail fmt!("%s with illegal ID %u - ID must be less than %u",
                      act, id, length),
        None => blk()
    }
}

#[doc(hidden)]
impl &Sem<~[Waitqueue]> {
    // The only other place that condvars get built is rwlock_write_mode.
    fn access_cond<U>(blk: fn(c: &Condvar) -> U) -> U {
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

impl Semaphore: Clone {
    /// Create a new handle to the semaphore.
    fn clone(&self) -> Semaphore {
        Semaphore { sem: Sem((*self.sem).clone()) }
    }
}

impl &Semaphore {
    /**
     * Acquire a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    fn acquire() { (&self.sem).acquire() }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist. Won't block the caller.
     */
    fn release() { (&self.sem).release() }

    /// Run a function with ownership of one of the semaphore's resources.
    fn access<U>(blk: fn() -> U) -> U { (&self.sem).access(blk) }
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
struct Mutex { priv sem: Sem<~[Waitqueue]> }

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

impl Mutex: Clone {
    /// Create a new handle to the mutex.
    fn clone(&self) -> Mutex { Mutex { sem: Sem((*self.sem).clone()) } }
}

impl &Mutex {
    /// Run a function with ownership of the mutex.
    fn lock<U>(blk: fn() -> U) -> U { (&self.sem).access(blk) }

    /// Run a function with ownership of the mutex and a handle to a condvar.
    fn lock_cond<U>(blk: fn(c: &Condvar) -> U) -> U {
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
struct RWlock {
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

impl &RWlock {
    /// Create a new handle to the rwlock.
    fn clone() -> RWlock {
        RWlock { order_lock:  (&(self.order_lock)).clone(),
                 access_lock: Sem((*self.access_lock).clone()),
                 state:       self.state.clone() }
    }

    /**
     * Run a function with the rwlock in read mode. Calls to 'read' from other
     * tasks may run concurrently with this one.
     */
    fn read<U>(blk: fn() -> U) -> U {
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
    fn write<U>(blk: fn() -> U) -> U {
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
    fn write_cond<U>(blk: fn(c: &Condvar) -> U) -> U {
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
     * ~~~
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
    fn write_downgrade<U>(blk: fn(v: RWlockWriteMode) -> U) -> U {
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
    fn downgrade(token: RWlockWriteMode/&a) -> RWlockReadMode/&a {
        if !ptr::ref_eq(self, token.lock) {
            fail ~"Can't downgrade() with a different rwlock's write_mode!";
        }
        unsafe {
            do task::unkillable {
                let mut first_reader = false;
                do self.state.with |state| {
                    assert !state.read_mode;
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
struct RWlockReleaseRead {
    lock: &RWlock,
}

impl RWlockReleaseRead : Drop {
    fn finalize(&self) {
        unsafe {
            do task::unkillable {
                let mut last_reader = false;
                do self.lock.state.with |state| {
                    assert state.read_mode;
                    assert state.read_count > 0;
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

fn RWlockReleaseRead(lock: &r/RWlock) -> RWlockReleaseRead/&r {
    RWlockReleaseRead {
        lock: lock
    }
}

// FIXME(#3588) should go inside of downgrade()
#[doc(hidden)]
struct RWlockReleaseDowngrade {
    lock: &RWlock,
}

impl RWlockReleaseDowngrade : Drop {
    fn finalize(&self) {
        unsafe {
            do task::unkillable {
                let mut writer_or_last_reader = false;
                do self.lock.state.with |state| {
                    if state.read_mode {
                        assert state.read_count > 0;
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

fn RWlockReleaseDowngrade(lock: &r/RWlock) -> RWlockReleaseDowngrade/&r {
    RWlockReleaseDowngrade {
        lock: lock
    }
}

/// The "write permission" token used for rwlock.write_downgrade().
pub struct RWlockWriteMode { priv lock: &RWlock }
impl RWlockWriteMode : Drop { fn finalize(&self) {} }
/// The "read permission" token used for rwlock.write_downgrade().
pub struct RWlockReadMode  { priv lock: &RWlock }
impl RWlockReadMode : Drop { fn finalize(&self) {} }

impl &RWlockWriteMode {
    /// Access the pre-downgrade rwlock in write mode.
    fn write<U>(blk: fn() -> U) -> U { blk() }
    /// Access the pre-downgrade rwlock in write mode with a condvar.
    fn write_cond<U>(blk: fn(c: &Condvar) -> U) -> U {
        blk(&Condvar { sem: &self.lock.access_lock })
    }
}
impl &RWlockReadMode {
    /// Access the post-downgrade rwlock in read mode.
    fn read<U>(blk: fn() -> U) -> U { blk() }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use core::prelude::*;

    use sync::*;

    use core::cast;
    use core::option;
    use core::pipes;
    use core::ptr;
    use core::result;
    use core::task;
    use core::vec;

    /************************************************************************
     * Semaphore tests
     ************************************************************************/
    #[test]
    pub fn test_sem_acquire_release() {
        let s = ~semaphore(1);
        s.acquire();
        s.release();
        s.acquire();
    }
    #[test]
    pub fn test_sem_basic() {
        let s = ~semaphore(1);
        do s.access { }
    }
    #[test]
    pub fn test_sem_as_mutex() {
        let s = ~semaphore(1);
        let s2 = ~s.clone();
        do task::spawn |move s2| {
            do s2.access {
                for 5.times { task::yield(); }
            }
        }
        do s.access {
            for 5.times { task::yield(); }
        }
    }
    #[test]
    pub fn test_sem_as_cvar() {
        /* Child waits and parent signals */
        let (p,c) = pipes::stream();
        let s = ~semaphore(0);
        let s2 = ~s.clone();
        do task::spawn |move s2, move c| {
            s2.acquire();
            c.send(());
        }
        for 5.times { task::yield(); }
        s.release();
        let _ = p.recv();

        /* Parent waits and child signals */
        let (p,c) = pipes::stream();
        let s = ~semaphore(0);
        let s2 = ~s.clone();
        do task::spawn |move s2, move p| {
            for 5.times { task::yield(); }
            s2.release();
            let _ = p.recv();
        }
        s.acquire();
        c.send(());
    }
    #[test]
    pub fn test_sem_multi_resource() {
        // Parent and child both get in the critical section at the same
        // time, and shake hands.
        let s = ~semaphore(2);
        let s2 = ~s.clone();
        let (p1,c1) = pipes::stream();
        let (p2,c2) = pipes::stream();
        do task::spawn |move s2, move c1, move p2| {
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
    pub fn test_sem_runtime_friendly_blocking() {
        // Force the runtime to schedule two threads on the same sched_loop.
        // When one blocks, it should schedule the other one.
        do task::spawn_sched(task::ManualThreads(1)) {
            let s = ~semaphore(1);
            let s2 = ~s.clone();
            let (p,c) = pipes::stream();
            let child_data = ~mut Some((move s2, move c));
            do s.access {
                let (s2,c) = option::swap_unwrap(child_data);
                do task::spawn |move c, move s2| {
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
    pub fn test_mutex_lock() {
        // Unsafely achieve shared state, and do the textbook
        // "load tmp = move ptr; inc tmp; store ptr <- tmp" dance.
        let (p,c) = pipes::stream();
        let m = ~Mutex();
        let m2 = ~m.clone();
        let mut sharedstate = ~0;
        let ptr = ptr::addr_of(&(*sharedstate));
        do task::spawn |move m2, move c| {
            let sharedstate: &mut int =
                unsafe { cast::reinterpret_cast(&ptr) };
            access_shared(sharedstate, m2, 10);
            c.send(());

        }
        access_shared(sharedstate, m, 10);
        let _ = p.recv();

        assert *sharedstate == 20;

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
    pub fn test_mutex_cond_wait() {
        let m = ~Mutex();

        // Child wakes up parent
        do m.lock_cond |cond| {
            let m2 = ~m.clone();
            do task::spawn |move m2| {
                do m2.lock_cond |cond| {
                    let woken = cond.signal();
                    assert woken;
                }
            }
            cond.wait();
        }
        // Parent wakes up child
        let (port,chan) = pipes::stream();
        let m3 = ~m.clone();
        do task::spawn |move chan, move m3| {
            do m3.lock_cond |cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            }
        }
        let _ = port.recv(); // Wait until child gets in the mutex
        do m.lock_cond |cond| {
            let woken = cond.signal();
            assert woken;
        }
        let _ = port.recv(); // Wait until child wakes up
    }
    #[cfg(test)]
    pub fn test_mutex_cond_broadcast_helper(num_waiters: uint) {
        let m = ~Mutex();
        let mut ports = ~[];

        for num_waiters.times {
            let mi = ~m.clone();
            let (port, chan) = pipes::stream();
            ports.push(move port);
            do task::spawn |move chan, move mi| {
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
            assert num_woken == num_waiters;
        }
        // wait until all children wake up
        for ports.each |port| { let _ = port.recv(); }
    }
    #[test]
    pub fn test_mutex_cond_broadcast() {
        test_mutex_cond_broadcast_helper(12);
    }
    #[test]
    pub fn test_mutex_cond_broadcast_none() {
        test_mutex_cond_broadcast_helper(0);
    }
    #[test]
    pub fn test_mutex_cond_no_waiter() {
        let m = ~Mutex();
        let m2 = ~m.clone();
        do task::try |move m| {
            do m.lock_cond |_x| { }
        };
        do m2.lock_cond |cond| {
            assert !cond.signal();
        }
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_mutex_killed_simple() {
        // Mutex must get automatically unlocked if failed/killed within.
        let m = ~Mutex();
        let m2 = ~m.clone();

        let result: result::Result<(),()> = do task::try |move m2| {
            do m2.lock {
                fail;
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do m.lock { }
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_mutex_killed_cond() {
        // Getting killed during cond wait must not corrupt the mutex while
        // unwinding (e.g. double unlock).
        let m = ~Mutex();
        let m2 = ~m.clone();

        let result: result::Result<(),()> = do task::try |move m2| {
            let (p,c) = pipes::stream();
            do task::spawn |move p| { // linked
                let _ = p.recv(); // wait for sibling to get in the mutex
                task::yield();
                fail;
            }
            do m2.lock_cond |cond| {
                c.send(()); // tell sibling go ahead
                cond.wait(); // block forever
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do m.lock_cond |cond| {
            let woken = cond.signal();
            assert !woken;
        }
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_mutex_killed_broadcast() {
        let m = ~Mutex();
        let m2 = ~m.clone();
        let (p,c) = pipes::stream();

        let result: result::Result<(),()> = do task::try |move c, move m2| {
            let mut sibling_convos = ~[];
            for 2.times {
                let (p,c) = pipes::stream();
                let c = ~mut Some(move c);
                sibling_convos.push(move p);
                let mi = ~m2.clone();
                // spawn sibling task
                do task::spawn |move mi, move c| { // linked
                    do mi.lock_cond |cond| {
                        let c = option::swap_unwrap(c);
                        c.send(()); // tell sibling to go ahead
                        let _z = SendOnFailure(move c);
                        cond.wait(); // block forever
                    }
                }
            }
            for vec::each(sibling_convos) |p| {
                let _ = p.recv(); // wait for sibling to get in the mutex
            }
            do m2.lock { }
            c.send(move sibling_convos); // let parent wait on all children
            fail;
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        for vec::each(p.recv()) |p| { p.recv(); } // wait on all its siblings
        do m.lock_cond |cond| {
            let woken = cond.broadcast();
            assert woken == 0;
        }
        struct SendOnFailure {
            c: pipes::Chan<()>,
        }

        impl SendOnFailure : Drop {
            fn finalize(&self) {
                self.c.send(());
            }
        }

        fn SendOnFailure(c: pipes::Chan<()>) -> SendOnFailure {
            SendOnFailure {
                c: move c
            }
        }
    }
    #[test]
    pub fn test_mutex_cond_signal_on_0() {
        // Tests that signal_on(0) is equivalent to signal().
        let m = ~Mutex();
        do m.lock_cond |cond| {
            let m2 = ~m.clone();
            do task::spawn |move m2| {
                do m2.lock_cond |cond| {
                    cond.signal_on(0);
                }
            }
            cond.wait();
        }
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_mutex_different_conds() {
        let result = do task::try {
            let m = ~mutex_with_condvars(2);
            let m2 = ~m.clone();
            let (p,c) = pipes::stream();
            do task::spawn |move m2, move c| {
                do m2.lock_cond |cond| {
                    c.send(());
                    cond.wait_on(1);
                }
            }
            let _ = p.recv();
            do m.lock_cond |cond| {
                if !cond.signal_on(0) {
                    fail; // success; punt sibling awake.
                }
            }
        };
        assert result.is_err();
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_mutex_no_condvars() {
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.wait(); }
        };
        assert result.is_err();
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.signal(); }
        };
        assert result.is_err();
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.broadcast(); }
        };
        assert result.is_err();
    }
    /************************************************************************
     * Reader/writer lock tests
     ************************************************************************/
    #[cfg(test)]
    pub enum RWlockMode { Read, Write, Downgrade, DowngradeRead }
    #[cfg(test)]
    pub fn lock_rwlock_in_mode(x: &RWlock, mode: RWlockMode, blk: fn()) {
        match mode {
            Read => x.read(blk),
            Write => x.write(blk),
            Downgrade =>
                do x.write_downgrade |mode| {
                    (&mode).write(blk);
                },
            DowngradeRead =>
                do x.write_downgrade |mode| {
                    let mode = x.downgrade(move mode);
                    (&mode).read(blk);
                },
        }
    }
    #[cfg(test)]
    pub fn test_rwlock_exclusion(x: ~RWlock,
                                 mode1: RWlockMode,
                                 mode2: RWlockMode) {
        // Test mutual exclusion between readers and writers. Just like the
        // mutex mutual exclusion test, a ways above.
        let (p,c) = pipes::stream();
        let x2 = ~x.clone();
        let mut sharedstate = ~0;
        let ptr = ptr::addr_of(&(*sharedstate));
        do task::spawn |move c, move x2| {
            let sharedstate: &mut int =
                unsafe { cast::reinterpret_cast(&ptr) };
            access_shared(sharedstate, x2, mode1, 10);
            c.send(());
        }
        access_shared(sharedstate, x, mode2, 10);
        let _ = p.recv();

        assert *sharedstate == 20;

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
    pub fn test_rwlock_readers_wont_modify_the_data() {
        test_rwlock_exclusion(~RWlock(), Read, Write);
        test_rwlock_exclusion(~RWlock(), Write, Read);
        test_rwlock_exclusion(~RWlock(), Read, Downgrade);
        test_rwlock_exclusion(~RWlock(), Downgrade, Read);
    }
    #[test]
    pub fn test_rwlock_writers_and_writers() {
        test_rwlock_exclusion(~RWlock(), Write, Write);
        test_rwlock_exclusion(~RWlock(), Write, Downgrade);
        test_rwlock_exclusion(~RWlock(), Downgrade, Write);
        test_rwlock_exclusion(~RWlock(), Downgrade, Downgrade);
    }
    #[cfg(test)]
    pub fn test_rwlock_handshake(x: ~RWlock,
                                 mode1: RWlockMode,
                                 mode2: RWlockMode,
                                 make_mode2_go_first: bool) {
        // Much like sem_multi_resource.
        let x2 = ~x.clone();
        let (p1,c1) = pipes::stream();
        let (p2,c2) = pipes::stream();
        do task::spawn |move c1, move x2, move p2| {
            if !make_mode2_go_first {
                let _ = p2.recv(); // parent sends to us once it locks, or ...
            }
            do lock_rwlock_in_mode(x2, mode2) {
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
    pub fn test_rwlock_readers_and_readers() {
        test_rwlock_handshake(~RWlock(), Read, Read, false);
        // The downgrader needs to get in before the reader gets in, otherwise
        // they cannot end up reading at the same time.
        test_rwlock_handshake(~RWlock(), DowngradeRead, Read, false);
        test_rwlock_handshake(~RWlock(), Read, DowngradeRead, true);
        // Two downgrade_reads can never both end up reading at the same time.
    }
    #[test]
    pub fn test_rwlock_downgrade_unlock() {
        // Tests that downgrade can unlock the lock in both modes
        let x = ~RWlock();
        do lock_rwlock_in_mode(x, Downgrade) { }
        test_rwlock_handshake(move x, Read, Read, false);
        let y = ~RWlock();
        do lock_rwlock_in_mode(y, DowngradeRead) { }
        test_rwlock_exclusion(move y, Write, Write);
    }
    #[test]
    pub fn test_rwlock_read_recursive() {
        let x = ~RWlock();
        do x.read { do x.read { } }
    }
    #[test]
    pub fn test_rwlock_cond_wait() {
        // As test_mutex_cond_wait above.
        let x = ~RWlock();

        // Child wakes up parent
        do x.write_cond |cond| {
            let x2 = ~x.clone();
            do task::spawn |move x2| {
                do x2.write_cond |cond| {
                    let woken = cond.signal();
                    assert woken;
                }
            }
            cond.wait();
        }
        // Parent wakes up child
        let (port,chan) = pipes::stream();
        let x3 = ~x.clone();
        do task::spawn |move x3, move chan| {
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
            assert woken;
        }
        let _ = port.recv(); // Wait until child wakes up
        do x.read { } // Just for good measure
    }
    #[cfg(test)]
    pub fn test_rwlock_cond_broadcast_helper(num_waiters: uint,
                                             dg1: bool,
                                             dg2: bool) {
        // Much like the mutex broadcast test. Downgrade-enabled.
        fn lock_cond(x: &RWlock, downgrade: bool, blk: fn(c: &Condvar)) {
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
            let xi = ~x.clone();
            let (port, chan) = pipes::stream();
            ports.push(move port);
            do task::spawn |move chan, move xi| {
                do lock_cond(xi, dg1) |cond| {
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
            assert num_woken == num_waiters;
        }
        // wait until all children wake up
        for ports.each |port| { let _ = port.recv(); }
    }
    #[test]
    pub fn test_rwlock_cond_broadcast() {
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
    pub fn rwlock_kill_helper(mode1: RWlockMode, mode2: RWlockMode) {
        // Mutex must get automatically unlocked if failed/killed within.
        let x = ~RWlock();
        let x2 = ~x.clone();

        let result: result::Result<(),()> = do task::try |move x2| {
            do lock_rwlock_in_mode(x2, mode1) {
                fail;
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do lock_rwlock_in_mode(x, mode2) { }
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_rwlock_reader_killed_writer() {
        rwlock_kill_helper(Read, Write);
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_rwlock_writer_killed_reader() {
        rwlock_kill_helper(Write,Read );
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_rwlock_reader_killed_reader() {
        rwlock_kill_helper(Read, Read );
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_rwlock_writer_killed_writer() {
        rwlock_kill_helper(Write,Write);
    }
    #[test] #[ignore(cfg(windows))]
    pub fn test_rwlock_kill_downgrader() {
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
    pub fn test_rwlock_downgrade_cant_swap() {
        // Tests that you can't downgrade with a different rwlock's token.
        let x = ~RWlock();
        let y = ~RWlock();
        do x.write_downgrade |xwrite| {
            let mut xopt = Some(move xwrite);
            do y.write_downgrade |_ywrite| {
                y.downgrade(option::swap_unwrap(&mut xopt));
                error!("oops, y.downgrade(x) should have failed!");
            }
        }
    }
}
