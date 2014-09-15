// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Raw concurrency primitives you know and love.
//!
//! These primitives are not recommended for general use, but are provided for
//! flavorful use-cases. It is recommended to use the types at the top of the
//! `sync` crate which wrap values directly and provide safer abstractions for
//! containing data.

use core::prelude::*;

use core::atomic;
use core::finally::Finally;
use core::kinds::marker;
use core::mem;
use core::cell::UnsafeCell;
use collections::{Vec, MutableSeq};

use mutex;
use comm::{Receiver, Sender, channel};

/****************************************************************************
 * Internals
 ****************************************************************************/

// Each waiting task receives on one of these.
type WaitEnd = Receiver<()>;
type SignalEnd = Sender<()>;
// A doubly-ended queue of waiting tasks.
struct WaitQueue {
    head: Receiver<SignalEnd>,
    tail: Sender<SignalEnd>,
}

impl WaitQueue {
    fn new() -> WaitQueue {
        let (block_tail, block_head) = channel();
        WaitQueue { head: block_head, tail: block_tail }
    }

    // Signals one live task from the queue.
    fn signal(&self) -> bool {
        match self.head.try_recv() {
            Ok(ch) => {
                // Send a wakeup signal. If the waiter was killed, its port will
                // have closed. Keep trying until we get a live task.
                if ch.send_opt(()).is_ok() {
                    true
                } else {
                    self.signal()
                }
            }
            _ => false
        }
    }

    fn broadcast(&self) -> uint {
        let mut count = 0;
        loop {
            match self.head.try_recv() {
                Ok(ch) => {
                    if ch.send_opt(()).is_ok() {
                        count += 1;
                    }
                }
                _ => break
            }
        }
        count
    }

    fn wait_end(&self) -> WaitEnd {
        let (signal_end, wait_end) = channel();
        self.tail.send(signal_end);
        wait_end
    }
}

// The building-block used to make semaphores, mutexes, and rwlocks.
struct Sem<Q> {
    lock: mutex::Mutex,
    // n.b, we need Sem to be `Sync`, but the WaitQueue type is not send/share
    //      (for good reason). We have an internal invariant on this semaphore,
    //      however, that the queue is never accessed outside of a locked
    //      context.
    inner: UnsafeCell<SemInner<Q>>
}

struct SemInner<Q> {
    count: int,
    waiters: WaitQueue,
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked: Q,
}

#[must_use]
struct SemGuard<'a, Q:'a> {
    sem: &'a Sem<Q>,
}

impl<Q: Send> Sem<Q> {
    fn new(count: int, q: Q) -> Sem<Q> {
        assert!(count >= 0,
                "semaphores cannot be initialized with negative values");
        Sem {
            lock: mutex::Mutex::new(),
            inner: UnsafeCell::new(SemInner {
                waiters: WaitQueue::new(),
                count: count,
                blocked: q,
            })
        }
    }

    unsafe fn with(&self, f: |&mut SemInner<Q>|) {
        let _g = self.lock.lock();
        // This &mut is safe because, due to the lock, we are the only one who can touch the data
        f(&mut *self.inner.get())
    }

    pub fn acquire(&self) {
        unsafe {
            let mut waiter_nobe = None;
            self.with(|state| {
                state.count -= 1;
                if state.count < 0 {
                    // Create waiter nobe, enqueue ourself, and tell
                    // outer scope we need to block.
                    waiter_nobe = Some(state.waiters.wait_end());
                }
            });
            // Uncomment if you wish to test for sem races. Not
            // valgrind-friendly.
            /* for _ in range(0u, 1000) { task::deschedule(); } */
            // Need to wait outside the exclusive.
            if waiter_nobe.is_some() {
                let _ = waiter_nobe.unwrap().recv();
            }
        }
    }

    pub fn release(&self) {
        unsafe {
            self.with(|state| {
                state.count += 1;
                if state.count <= 0 {
                    state.waiters.signal();
                }
            })
        }
    }

    pub fn access<'a>(&'a self) -> SemGuard<'a, Q> {
        self.acquire();
        SemGuard { sem: self }
    }
}

#[unsafe_destructor]
impl<'a, Q: Send> Drop for SemGuard<'a, Q> {
    fn drop(&mut self) {
        self.sem.release();
    }
}

impl Sem<Vec<WaitQueue>> {
    fn new_and_signal(count: int, num_condvars: uint) -> Sem<Vec<WaitQueue>> {
        let mut queues = Vec::new();
        for _ in range(0, num_condvars) { queues.push(WaitQueue::new()); }
        Sem::new(count, queues)
    }

    // The only other places that condvars get built are rwlock.write_cond()
    // and rwlock_write_mode.
    pub fn access_cond<'a>(&'a self) -> SemCondGuard<'a> {
        SemCondGuard {
            guard: self.access(),
            cvar: Condvar { sem: self, order: Nothing, nocopy: marker::NoCopy },
        }
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
    sem: &'a Sem<Vec<WaitQueue> >,
    // This is (can be) an extra semaphore which is held around the reacquire
    // operation on the first one. This is only used in cvars associated with
    // rwlocks, and is needed to ensure that, when a downgrader is trying to
    // hand off the access lock (which would be the first field, here), a 2nd
    // writer waking up from a cvar wait can't race with a reader to steal it,
    // See the comment in write_cond for more detail.
    order: ReacquireOrderLock<'a>,
    // Make sure condvars are non-copyable.
    nocopy: marker::NoCopy,
}

impl<'a> Condvar<'a> {
    /// Atomically drop the associated lock, and block until a signal is sent.
    ///
    /// # Failure
    ///
    /// A task which is killed while waiting on a condition variable will wake
    /// up, fail, and unlock the associated lock as it unwinds.
    pub fn wait(&self) { self.wait_on(0) }

    /// As wait(), but can specify which of multiple condition variables to
    /// wait on. Only a signal_on() or broadcast_on() with the same condvar_id
    /// will wake this thread.
    ///
    /// The associated lock must have been initialised with an appropriate
    /// number of condvars. The condvar_id must be between 0 and num_condvars-1
    /// or else this call will fail.
    ///
    /// wait() is equivalent to wait_on(0).
    pub fn wait_on(&self, condvar_id: uint) {
        let mut wait_end = None;
        let mut out_of_bounds = None;
        // Release lock, 'atomically' enqueuing ourselves in so doing.
        unsafe {
            self.sem.with(|state| {
                if condvar_id < state.blocked.len() {
                    // Drop the lock.
                    state.count += 1;
                    if state.count <= 0 {
                        state.waiters.signal();
                    }
                    // Create waiter nobe, and enqueue ourself to
                    // be woken up by a signaller.
                    wait_end = Some(state.blocked[condvar_id].wait_end());
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
                let _ = wait_end.take().unwrap().recv();
            }).finally(|| {
                // Reacquire the condvar.
                match self.order {
                    Just(lock) => {
                        let _g = lock.access();
                        self.sem.acquire();
                    }
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
            self.sem.with(|state| {
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
            self.sem.with(|state| {
                if condvar_id < state.blocked.len() {
                    // To avoid :broadcast_heavy, we make a new waitqueue,
                    // swap it out with the old one, and broadcast on the
                    // old one outside of the little-lock.
                    queue = Some(mem::replace(state.blocked.get_mut(condvar_id),
                                              WaitQueue::new()));
                } else {
                    out_of_bounds = Some(state.blocked.len());
                }
            });
            check_cvar_bounds(out_of_bounds,
                              condvar_id,
                              "cond.signal_on()",
                              || {
                queue.take().unwrap().broadcast()
            })
        }
    }
}

// Checks whether a condvar ID was out of bounds, and fails if so, or does
// something else next on success.
#[inline]
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

#[must_use]
struct SemCondGuard<'a> {
    guard: SemGuard<'a, Vec<WaitQueue>>,
    cvar: Condvar<'a>,
}

/****************************************************************************
 * Semaphores
 ****************************************************************************/

/// A counting, blocking, bounded-waiting semaphore.
pub struct Semaphore {
    sem: Sem<()>,
}

/// An RAII guard used to represent an acquired resource to a semaphore. When
/// dropped, this value will release the resource back to the semaphore.
#[must_use]
pub struct SemaphoreGuard<'a> {
    _guard: SemGuard<'a, ()>,
}

impl Semaphore {
    /// Create a new semaphore with the specified count.
    ///
    /// # Failure
    ///
    /// This function will fail if `count` is negative.
    pub fn new(count: int) -> Semaphore {
        Semaphore { sem: Sem::new(count, ()) }
    }

    /// Acquire a resource represented by the semaphore. Blocks if necessary
    /// until resource(s) become available.
    pub fn acquire(&self) { self.sem.acquire() }

    /// Release a held resource represented by the semaphore. Wakes a blocked
    /// contending task, if any exist. Won't block the caller.
    pub fn release(&self) { self.sem.release() }

    /// Acquire a resource of this semaphore, returning an RAII guard which will
    /// release the resource when dropped.
    pub fn access<'a>(&'a self) -> SemaphoreGuard<'a> {
        SemaphoreGuard { _guard: self.sem.access() }
    }
}

/****************************************************************************
 * Mutexes
 ****************************************************************************/

/// A blocking, bounded-waiting, mutual exclusion lock with an associated
/// FIFO condition variable.
///
/// # Failure
/// A task which fails while holding a mutex will unlock the mutex as it
/// unwinds.
pub struct Mutex {
    sem: Sem<Vec<WaitQueue>>,
}

/// An RAII structure which is used to gain access to a mutex's condition
/// variable. Additionally, when a value of this type is dropped, the
/// corresponding mutex is also unlocked.
#[must_use]
pub struct MutexGuard<'a> {
    _guard: SemGuard<'a, Vec<WaitQueue>>,
    /// Inner condition variable which is connected to the outer mutex, and can
    /// be used for atomic-unlock-and-deschedule.
    pub cond: Condvar<'a>,
}

impl Mutex {
    /// Create a new mutex, with one associated condvar.
    pub fn new() -> Mutex { Mutex::new_with_condvars(1) }

    /// Create a new mutex, with a specified number of associated condvars. This
    /// will allow calling wait_on/signal_on/broadcast_on with condvar IDs
    /// between 0 and num_condvars-1. (If num_condvars is 0, lock_cond will be
    /// allowed but any operations on the condvar will fail.)
    pub fn new_with_condvars(num_condvars: uint) -> Mutex {
        Mutex { sem: Sem::new_and_signal(1, num_condvars) }
    }

    /// Acquires ownership of this mutex, returning an RAII guard which will
    /// unlock the mutex when dropped. The associated condition variable can
    /// also be accessed through the returned guard.
    pub fn lock<'a>(&'a self) -> MutexGuard<'a> {
        let SemCondGuard { guard, cvar } = self.sem.access_cond();
        MutexGuard { _guard: guard, cond: cvar }
    }
}

/****************************************************************************
 * Reader-writer locks
 ****************************************************************************/

// NB: Wikipedia - Readers-writers_problem#The_third_readers-writers_problem

/// A blocking, no-starvation, reader-writer lock with an associated condvar.
///
/// # Failure
///
/// A task which fails while holding an rwlock will unlock the rwlock as it
/// unwinds.
pub struct RWLock {
    order_lock:  Semaphore,
    access_lock: Sem<Vec<WaitQueue>>,

    // The only way the count flag is ever accessed is with xadd. Since it is
    // a read-modify-write operation, multiple xadds on different cores will
    // always be consistent with respect to each other, so a monotonic/relaxed
    // consistency ordering suffices (i.e., no extra barriers are needed).
    //
    // FIXME(#6598): The atomics module has no relaxed ordering flag, so I use
    // acquire/release orderings superfluously. Change these someday.
    read_count: atomic::AtomicUint,
}

/// An RAII helper which is created by acquiring a read lock on an RWLock. When
/// dropped, this will unlock the RWLock.
#[must_use]
pub struct RWLockReadGuard<'a> {
    lock: &'a RWLock,
}

/// An RAII helper which is created by acquiring a write lock on an RWLock. When
/// dropped, this will unlock the RWLock.
///
/// A value of this type can also be consumed to downgrade to a read-only lock.
#[must_use]
pub struct RWLockWriteGuard<'a> {
    lock: &'a RWLock,
    /// Inner condition variable that is connected to the write-mode of the
    /// outer rwlock.
    pub cond: Condvar<'a>,
}

impl RWLock {
    /// Create a new rwlock, with one associated condvar.
    pub fn new() -> RWLock { RWLock::new_with_condvars(1) }

    /// Create a new rwlock, with a specified number of associated condvars.
    /// Similar to mutex_with_condvars.
    pub fn new_with_condvars(num_condvars: uint) -> RWLock {
        RWLock {
            order_lock: Semaphore::new(1),
            access_lock: Sem::new_and_signal(1, num_condvars),
            read_count: atomic::AtomicUint::new(0),
        }
    }

    /// Acquires a read-lock, returning an RAII guard that will unlock the lock
    /// when dropped. Calls to 'read' from other tasks may run concurrently with
    /// this one.
    pub fn read<'a>(&'a self) -> RWLockReadGuard<'a> {
        let _guard = self.order_lock.access();
        let old_count = self.read_count.fetch_add(1, atomic::Acquire);
        if old_count == 0 {
            self.access_lock.acquire();
        }
        RWLockReadGuard { lock: self }
    }

    /// Acquire a write-lock, returning an RAII guard that will unlock the lock
    /// when dropped. No calls to 'read' or 'write' from other tasks will run
    /// concurrently with this one.
    ///
    /// You can also downgrade a write to a read by calling the `downgrade`
    /// method on the returned guard. Additionally, the guard will contain a
    /// `Condvar` attached to this lock.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sync::raw::RWLock;
    ///
    /// let lock = RWLock::new();
    /// let write = lock.write();
    /// // ... exclusive access ...
    /// let read = write.downgrade();
    /// // ... shared access ...
    /// drop(read);
    /// ```
    pub fn write<'a>(&'a self) -> RWLockWriteGuard<'a> {
        let _g = self.order_lock.access();
        self.access_lock.acquire();

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
        RWLockWriteGuard {
            lock: self,
            cond: Condvar {
                sem: &self.access_lock,
                order: Just(&self.order_lock),
                nocopy: marker::NoCopy,
            }
        }
    }
}

impl<'a> RWLockWriteGuard<'a> {
    /// Consumes this write lock and converts it into a read lock.
    pub fn downgrade(self) -> RWLockReadGuard<'a> {
        let lock = self.lock;
        // Don't run the destructor of the write guard, we're in charge of
        // things from now on
        unsafe { mem::forget(self) }

        let old_count = lock.read_count.fetch_add(1, atomic::Release);
        // If another reader was already blocking, we need to hand-off
        // the "reader cloud" access lock to them.
        if old_count != 0 {
            // Guaranteed not to let another writer in, because
            // another reader was holding the order_lock. Hence they
            // must be the one to get the access_lock (because all
            // access_locks are acquired with order_lock held). See
            // the comment in write_cond for more justification.
            lock.access_lock.release();
        }
        RWLockReadGuard { lock: lock }
    }
}

#[unsafe_destructor]
impl<'a> Drop for RWLockWriteGuard<'a> {
    fn drop(&mut self) {
        self.lock.access_lock.release();
    }
}

#[unsafe_destructor]
impl<'a> Drop for RWLockReadGuard<'a> {
    fn drop(&mut self) {
        let old_count = self.lock.read_count.fetch_sub(1, atomic::Release);
        assert!(old_count > 0);
        if old_count == 1 {
            // Note: this release used to be outside of a locked access
            // to exclusive-protected state. If this code is ever
            // converted back to such (instead of using atomic ops),
            // this access MUST NOT go inside the exclusive access.
            self.lock.access_lock.release();
        }
    }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use std::prelude::*;

    use Arc;
    use super::{Semaphore, Mutex, RWLock, Condvar};

    use std::mem;
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
        let _g = s.access();
    }
    #[test]
    #[should_fail]
    fn test_sem_basic2() {
        Semaphore::new(-1);
    }
    #[test]
    fn test_sem_as_mutex() {
        let s = Arc::new(Semaphore::new(1));
        let s2 = s.clone();
        task::spawn(proc() {
            let _g = s2.access();
            for _ in range(0u, 5) { task::deschedule(); }
        });
        let _g = s.access();
        for _ in range(0u, 5) { task::deschedule(); }
    }
    #[test]
    fn test_sem_as_cvar() {
        /* Child waits and parent signals */
        let (tx, rx) = channel();
        let s = Arc::new(Semaphore::new(0));
        let s2 = s.clone();
        task::spawn(proc() {
            s2.acquire();
            tx.send(());
        });
        for _ in range(0u, 5) { task::deschedule(); }
        s.release();
        let _ = rx.recv();

        /* Parent waits and child signals */
        let (tx, rx) = channel();
        let s = Arc::new(Semaphore::new(0));
        let s2 = s.clone();
        task::spawn(proc() {
            for _ in range(0u, 5) { task::deschedule(); }
            s2.release();
            let _ = rx.recv();
        });
        s.acquire();
        tx.send(());
    }
    #[test]
    fn test_sem_multi_resource() {
        // Parent and child both get in the critical section at the same
        // time, and shake hands.
        let s = Arc::new(Semaphore::new(2));
        let s2 = s.clone();
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        task::spawn(proc() {
            let _g = s2.access();
            let _ = rx2.recv();
            tx1.send(());
        });
        let _g = s.access();
        tx2.send(());
        let _ = rx1.recv();
    }
    #[test]
    fn test_sem_runtime_friendly_blocking() {
        // Force the runtime to schedule two threads on the same sched_loop.
        // When one blocks, it should schedule the other one.
        let s = Arc::new(Semaphore::new(1));
        let s2 = s.clone();
        let (tx, rx) = channel();
        {
            let _g = s.access();
            task::spawn(proc() {
                tx.send(());
                drop(s2.access());
                tx.send(());
            });
            rx.recv(); // wait for child to come alive
            for _ in range(0u, 5) { task::deschedule(); } // let the child contend
        }
        rx.recv(); // wait for child to be done
    }
    /************************************************************************
     * Mutex tests
     ************************************************************************/
    #[test]
    fn test_mutex_lock() {
        // Unsafely achieve shared state, and do the textbook
        // "load tmp = move ptr; inc tmp; store ptr <- tmp" dance.
        let (tx, rx) = channel();
        let m = Arc::new(Mutex::new());
        let m2 = m.clone();
        let mut sharedstate = box 0;
        {
            let ptr: *mut int = &mut *sharedstate;
            task::spawn(proc() {
                access_shared(ptr, &m2, 10);
                tx.send(());
            });
        }
        {
            access_shared(&mut *sharedstate, &m, 10);
            let _ = rx.recv();

            assert_eq!(*sharedstate, 20);
        }

        fn access_shared(sharedstate: *mut int, m: &Arc<Mutex>, n: uint) {
            for _ in range(0u, n) {
                let _g = m.lock();
                let oldval = unsafe { *sharedstate };
                task::deschedule();
                unsafe { *sharedstate = oldval + 1; }
            }
        }
    }
    #[test]
    fn test_mutex_cond_wait() {
        let m = Arc::new(Mutex::new());

        // Child wakes up parent
        {
            let lock = m.lock();
            let m2 = m.clone();
            task::spawn(proc() {
                let lock = m2.lock();
                let woken = lock.cond.signal();
                assert!(woken);
            });
            lock.cond.wait();
        }
        // Parent wakes up child
        let (tx, rx) = channel();
        let m3 = m.clone();
        task::spawn(proc() {
            let lock = m3.lock();
            tx.send(());
            lock.cond.wait();
            tx.send(());
        });
        rx.recv(); // Wait until child gets in the mutex
        {
            let lock = m.lock();
            let woken = lock.cond.signal();
            assert!(woken);
        }
        rx.recv(); // Wait until child wakes up
    }

    fn test_mutex_cond_broadcast_helper(num_waiters: uint) {
        let m = Arc::new(Mutex::new());
        let mut rxs = Vec::new();

        for _ in range(0u, num_waiters) {
            let mi = m.clone();
            let (tx, rx) = channel();
            rxs.push(rx);
            task::spawn(proc() {
                let lock = mi.lock();
                tx.send(());
                lock.cond.wait();
                tx.send(());
            });
        }

        // wait until all children get in the mutex
        for rx in rxs.iter_mut() { rx.recv(); }
        {
            let lock = m.lock();
            let num_woken = lock.cond.broadcast();
            assert_eq!(num_woken, num_waiters);
        }
        // wait until all children wake up
        for rx in rxs.iter_mut() { rx.recv(); }
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
        let m = Arc::new(Mutex::new());
        let m2 = m.clone();
        let _ = task::try(proc() {
            drop(m.lock());
        });
        let lock = m2.lock();
        assert!(!lock.cond.signal());
    }
    #[test]
    fn test_mutex_killed_simple() {
        use std::any::Any;

        // Mutex must get automatically unlocked if failed/killed within.
        let m = Arc::new(Mutex::new());
        let m2 = m.clone();

        let result: result::Result<(), Box<Any + Send>> = task::try(proc() {
            let _lock = m2.lock();
            fail!();
        });
        assert!(result.is_err());
        // child task must have finished by the time try returns
        drop(m.lock());
    }
    #[test]
    fn test_mutex_cond_signal_on_0() {
        // Tests that signal_on(0) is equivalent to signal().
        let m = Arc::new(Mutex::new());
        let lock = m.lock();
        let m2 = m.clone();
        task::spawn(proc() {
            let lock = m2.lock();
            lock.cond.signal_on(0);
        });
        lock.cond.wait();
    }
    #[test]
    fn test_mutex_no_condvars() {
        let result = task::try(proc() {
            let m = Mutex::new_with_condvars(0);
            m.lock().cond.wait();
        });
        assert!(result.is_err());
        let result = task::try(proc() {
            let m = Mutex::new_with_condvars(0);
            m.lock().cond.signal();
        });
        assert!(result.is_err());
        let result = task::try(proc() {
            let m = Mutex::new_with_condvars(0);
            m.lock().cond.broadcast();
        });
        assert!(result.is_err());
    }
    /************************************************************************
     * Reader/writer lock tests
     ************************************************************************/
    #[cfg(test)]
    pub enum RWLockMode { Read, Write, Downgrade, DowngradeRead }
    #[cfg(test)]
    fn lock_rwlock_in_mode(x: &Arc<RWLock>, mode: RWLockMode, blk: ||) {
        match mode {
            Read => { let _g = x.read(); blk() }
            Write => { let _g = x.write(); blk() }
            Downgrade => { let _g = x.write(); blk() }
            DowngradeRead => { let _g = x.write().downgrade(); blk() }
        }
    }
    #[cfg(test)]
    fn test_rwlock_exclusion(x: Arc<RWLock>,
                             mode1: RWLockMode,
                             mode2: RWLockMode) {
        // Test mutual exclusion between readers and writers. Just like the
        // mutex mutual exclusion test, a ways above.
        let (tx, rx) = channel();
        let x2 = x.clone();
        let mut sharedstate = box 0;
        {
            let ptr: *const int = &*sharedstate;
            task::spawn(proc() {
                let sharedstate: &mut int =
                    unsafe { mem::transmute(ptr) };
                access_shared(sharedstate, &x2, mode1, 10);
                tx.send(());
            });
        }
        {
            access_shared(&mut *sharedstate, &x, mode2, 10);
            let _ = rx.recv();

            assert_eq!(*sharedstate, 20);
        }

        fn access_shared(sharedstate: &mut int, x: &Arc<RWLock>,
                         mode: RWLockMode, n: uint) {
            for _ in range(0u, n) {
                lock_rwlock_in_mode(x, mode, || {
                    let oldval = *sharedstate;
                    task::deschedule();
                    *sharedstate = oldval + 1;
                })
            }
        }
    }
    #[test]
    fn test_rwlock_readers_wont_modify_the_data() {
        test_rwlock_exclusion(Arc::new(RWLock::new()), Read, Write);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Write, Read);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Read, Downgrade);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Downgrade, Read);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Write, DowngradeRead);
        test_rwlock_exclusion(Arc::new(RWLock::new()), DowngradeRead, Write);
    }
    #[test]
    fn test_rwlock_writers_and_writers() {
        test_rwlock_exclusion(Arc::new(RWLock::new()), Write, Write);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Write, Downgrade);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Downgrade, Write);
        test_rwlock_exclusion(Arc::new(RWLock::new()), Downgrade, Downgrade);
    }
    #[cfg(test)]
    fn test_rwlock_handshake(x: Arc<RWLock>,
                             mode1: RWLockMode,
                             mode2: RWLockMode,
                             make_mode2_go_first: bool) {
        // Much like sem_multi_resource.
        let x2 = x.clone();
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        task::spawn(proc() {
            if !make_mode2_go_first {
                rx2.recv(); // parent sends to us once it locks, or ...
            }
            lock_rwlock_in_mode(&x2, mode2, || {
                if make_mode2_go_first {
                    tx1.send(()); // ... we send to it once we lock
                }
                rx2.recv();
                tx1.send(());
            })
        });
        if make_mode2_go_first {
            rx1.recv(); // child sends to us once it locks, or ...
        }
        lock_rwlock_in_mode(&x, mode1, || {
            if !make_mode2_go_first {
                tx2.send(()); // ... we send to it once we lock
            }
            tx2.send(());
            rx1.recv();
        })
    }
    #[test]
    fn test_rwlock_readers_and_readers() {
        test_rwlock_handshake(Arc::new(RWLock::new()), Read, Read, false);
        // The downgrader needs to get in before the reader gets in, otherwise
        // they cannot end up reading at the same time.
        test_rwlock_handshake(Arc::new(RWLock::new()), DowngradeRead, Read, false);
        test_rwlock_handshake(Arc::new(RWLock::new()), Read, DowngradeRead, true);
        // Two downgrade_reads can never both end up reading at the same time.
    }
    #[test]
    fn test_rwlock_downgrade_unlock() {
        // Tests that downgrade can unlock the lock in both modes
        let x = Arc::new(RWLock::new());
        lock_rwlock_in_mode(&x, Downgrade, || { });
        test_rwlock_handshake(x, Read, Read, false);
        let y = Arc::new(RWLock::new());
        lock_rwlock_in_mode(&y, DowngradeRead, || { });
        test_rwlock_exclusion(y, Write, Write);
    }
    #[test]
    fn test_rwlock_read_recursive() {
        let x = RWLock::new();
        let _g1 = x.read();
        let _g2 = x.read();
    }
    #[test]
    fn test_rwlock_cond_wait() {
        // As test_mutex_cond_wait above.
        let x = Arc::new(RWLock::new());

        // Child wakes up parent
        {
            let lock = x.write();
            let x2 = x.clone();
            task::spawn(proc() {
                let lock = x2.write();
                assert!(lock.cond.signal());
            });
            lock.cond.wait();
        }
        // Parent wakes up child
        let (tx, rx) = channel();
        let x3 = x.clone();
        task::spawn(proc() {
            let lock = x3.write();
            tx.send(());
            lock.cond.wait();
            tx.send(());
        });
        rx.recv(); // Wait until child gets in the rwlock
        drop(x.read()); // Must be able to get in as a reader
        {
            let x = x.write();
            assert!(x.cond.signal());
        }
        rx.recv(); // Wait until child wakes up
        drop(x.read()); // Just for good measure
    }
    #[cfg(test)]
    fn test_rwlock_cond_broadcast_helper(num_waiters: uint) {
        // Much like the mutex broadcast test. Downgrade-enabled.
        fn lock_cond(x: &Arc<RWLock>, blk: |c: &Condvar|) {
            let lock = x.write();
            blk(&lock.cond);
        }

        let x = Arc::new(RWLock::new());
        let mut rxs = Vec::new();

        for _ in range(0u, num_waiters) {
            let xi = x.clone();
            let (tx, rx) = channel();
            rxs.push(rx);
            task::spawn(proc() {
                lock_cond(&xi, |cond| {
                    tx.send(());
                    cond.wait();
                    tx.send(());
                })
            });
        }

        // wait until all children get in the mutex
        for rx in rxs.iter_mut() { let _ = rx.recv(); }
        lock_cond(&x, |cond| {
            let num_woken = cond.broadcast();
            assert_eq!(num_woken, num_waiters);
        });
        // wait until all children wake up
        for rx in rxs.iter_mut() { let _ = rx.recv(); }
    }
    #[test]
    fn test_rwlock_cond_broadcast() {
        test_rwlock_cond_broadcast_helper(0);
        test_rwlock_cond_broadcast_helper(12);
    }
    #[cfg(test)]
    fn rwlock_kill_helper(mode1: RWLockMode, mode2: RWLockMode) {
        use std::any::Any;

        // Mutex must get automatically unlocked if failed/killed within.
        let x = Arc::new(RWLock::new());
        let x2 = x.clone();

        let result: result::Result<(), Box<Any + Send>> = task::try(proc() {
            lock_rwlock_in_mode(&x2, mode1, || {
                fail!();
            })
        });
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
}
