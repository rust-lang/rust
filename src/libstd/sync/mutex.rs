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

// # The implementation of Rust's mutexes
//
// As hinted in the doc-comment above, the fundamental problem of implementing a
// mutex for rust is that you can't "just use pthreads". Green tasks are not
// allowed to block on a pthread mutex, because this can very easily lead to
// deadlock. Otherwise, there are other properties that we would want out of an
// "official mutex":
//
// * Any flavor of task can acquire the mutex, green or native
// * Any mixing of flavors of tasks can acquire the mutex. It should be possible
//   for green and native threads to contend over acquiring the mutex
// * This mutex should be "just as fast" as pthreads
// * Mutexes should be statically initializeable
// * Mutexes should really not need to have destructors (see static
//   initialization)
//
// Some properties which have been deemed not critical
//
// * Enforcing bounded waiting among all tasks acquiring the mutex. Mixing
//   green/native tasks is predicted to be a fairly rare case.
//
// ## Mutexes, take 1
//
// Within these constraints, the primitives we have available to us for blocking
// a task are the `deschedule` and `reawaken` methods on the `rt::Runtime`
// trait. These are the obvious choices to use first because they're "what we
// havel already" and should certainly be efficient.
//
// The sketch behind this mutex would be to use an intrusive (to avoid
// allocations) MPSC queue (the consumer is the lock holder) with some
// sprinkling of atomics to wake threads up. Each `BlockedTask` would be stored
// in the nodes of the queue.
//
// This implementation is all fine and dandy for green threads (user space
// context switching is fast), but when implemented, it was found that this
// implementation was about 50x slower than pthreads for native threads.
//
// Upon profiling, nearly all time was spent in cvar signal/wait (that's how
// native threads implement deschedule/reawaken). The problem was never tracked
// down with 100% certainty, but it was able discovered that this huge slowdown
// was only on a multicore system, not a single core system. With this knowledge
// in hand, plus some idea of how pthread mutexes are implemented, it was
// deduced that the kernel essentially knows what's going on when everyone's
// contended on the same mutex (as in the pthreads case). The kernel can
// cleverly schedule threads to *not* wake up on remote cores because all the
// work needs to happen on the same core (that's the whole point of a mutex).
// The deschedule/reawaken methods put threads to sleep on localized cvars, so
// the kernel had no idea that all our threads were contending *on the same
// mutex*.
//
// With this information in mind, it was concluded that it's impossible to
// create a pthreads-competitive mutex with the deschedule/reawaken primitives.
// We simply have no way of instructing the kernel that all native threads are
// contended on one object and should therefore *not* be spread out on many
// cores.
//
// ## Mutexes, take 2
//
// Back do the drawing board, the key idea was to actually have this mutex be a
// wrapper around a pthreads mutex. This would clearly solve the native threads
// problem (we'd be "just as fast" as pthreads), but the green problem comes
// back into play (you can't just grab the lock).
//
// The solution found (and the current implementation) ended up having a hybrid
// solution of queues/mutexes. The key idea is that green threads only ever
// *trylock* and use an internal queue to keep track of who's waiting, and
// native threads will simply just call *lock*.
//
// With this scheme, we get all the benefits of both worlds:
//
// * Any flavor of task (even mixed) can grab a mutex, pthreads arbitrates among
//   all native and the first green tasks, and then green tasks use atomics to
//   arbitrate among themselves.
// * We're just as fast as pthreads (within a small percentage of course)
// * Native mutexes are statically initializeable, and some clever usage of
//   atomics can make the green halves of the mutex also statically
//   initializeable.
// * No destructors are necessary (there is no memory allocation). The caveat
//   here is that windows doesn't have statically initialized mutexes, but it is
//   predicted that statically initialized mutexes won't be *too* common. Plus,
//   the "free" happens at program end when cleaning up doesn't matter *that*
//   much.
//
// As you'll find out in the implementation, this approach cannot be fair to
// native and green threads. In order to soundly drain the internal queue of
// green threads, they *must* be favored over native threads. It was an explicit
// non-goal of these mutexes to be completely fair to everyone, so this has been
// deemed acceptable.
//
// ## Mutexes, take 3
//
// The idea in take 2 of having the Mutex reduce to a native mutex for native
// tasks was awesome.
//
// However, the implementation in take 2 used thread_yield()
// to cover races due to the lack of a necessary additional lock.
//
// So, the implementation was rewritten to add an internal lock, queue_lock, which
// allows a more robust implementation.
//
// In this version, native threads will queue up if any thread is queued, making
// the implementation fair even when green and native threads are mixed
//
// This is the high-level implementation of the mutexes, but the nitty gritty
// details can be found in the code below.

use ops::Drop;
use option::{Option, Some, None};
use result::{Err, Ok};
use rt::local::Local;
use rt::task::{BlockedTask, Task};
use sync::atomics;
use unstable::mutex;

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
/// use std::sync::Mutex;
///
/// let mut m = Mutex::new();
/// let guard = m.lock();
/// // do some work
/// drop(guard); // unlock the lock
///
/// {
///     let _g = m.lock();
///     // do some work in a scope
/// }
///
/// // now the mutex is unlocked
/// ```
pub struct Mutex {
    priv lock: StaticMutex,
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
/// use std::sync::{StaticMutex, MUTEX_INIT};
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
    /// The OS mutex (pthreads/windows equivalent) that we're wrapping.
    priv lock: mutex::Mutex,
    priv owner: atomics::AtomicUint,
    priv queue_nonempty: atomics::AtomicBool,

    priv queue_lock: mutex::Mutex,
    /// Internal queue that all green threads will be blocked on.
    priv queue: *mut Node,
    priv queue_tail: *mut *mut Node,
}

struct Node {
    task: Option<BlockedTask>,
    next: *mut Node,
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
pub struct Guard<'a> {
    priv lock: &'a mut StaticMutex,
}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
pub static MUTEX_INIT: StaticMutex = StaticMutex {
    lock: mutex::MUTEX_INIT,
    owner: atomics::INIT_ATOMIC_UINT,
    queue_nonempty: atomics::INIT_ATOMIC_BOOL,

    queue_lock: mutex::MUTEX_INIT,
    queue: 0 as *mut Node,
    queue_tail: 0 as *mut *mut Node,
};

// set this to false to remove green task support and speed up mutexes
// XXX: this should probably be a static mut in a central location
static support_nonblocking_tasks: bool = true;

impl StaticMutex {
    /// Attempts to grab this lock, see `Mutex::try_lock`
    pub fn try_lock<'a>(&'a mut self) -> Option<Guard<'a>> {
        let queue_nonempty = self.queue_nonempty.load(atomics::Acquire);
        if !queue_nonempty && unsafe { self.lock.trylock() } {
            Some(Guard{ lock: self })
        } else {
            None
        }
        // XXX: should we check if we are the owner and fail! here?
    }

    /// Acquires this lock, see `Mutex::lock`
    pub fn lock<'a>(&'a mut self) -> Guard<'a> {
        let t: *mut Task = unsafe {Local::unsafe_borrow()};
        let ourselves: uint = t as uint;

        let (queue_nonempty, can_block) = if support_nonblocking_tasks {
            (self.queue_nonempty.load(atomics::Acquire), unsafe {(*t).can_block()})
        } else {
            (false, true)
        };

        // if any task is queued, we must always queue too, even if native, to preserve fairness
        if !queue_nonempty && can_block {
            if self.owner.load(atomics::Relaxed) == ourselves {
                if !unsafe { self.lock.trylock() } {
                    // ABA-safe because the trylock is an acquire barrier
                    if self.owner.load(atomics::Relaxed) == ourselves {
                        fail!("attempted to lock mutex already owned by the current task");
                    } else {
                        unsafe { self.lock.lock(); }
                    }
                }
            } else {
                unsafe { self.lock.lock(); }
            }
        } else if !queue_nonempty && unsafe { self.lock.trylock() } {
            // nice, we are already done
        } else {
            // we need to take queue_lock and redo all checks there
            unsafe {self.queue_lock.lock();}

            let queue_nonempty = self.queue_nonempty.load(atomics::Relaxed);
            if !queue_nonempty {
                self.queue_nonempty.store(true, atomics::Relaxed);

                // we must ensure that either the unlocker sees the queue_nonempty store
                // or that we succeed in the trylock
                // it seems SeqCst is required for that
                atomics::fence(atomics::SeqCst);
            }

            // None => we have the lock, continue
            // Some(Ok(()) => queue up and go to sleep
            // Some(Err(task)) => queue up and steal task
            let mut decision = if unsafe {self.lock.trylock()} {
                if !queue_nonempty {
                    // we unexpectedly got the lock with no one queued up
                    // this is executed rarely, since we already tried this at the beginning
                    self.queue_nonempty.store(false, atomics::Relaxed);
                    None
                } else {
                    // the trylock succeeded and the queue is non-empty
                    // we need to queue and wake up the next task ourselves

                    let node = self.queue;
                    self.queue = unsafe {(*node).next};
                    if self.queue == 0 as *mut Node {
                        self.queue_tail = 0 as *mut *mut Node;
                    }
                    let stolen_task = unsafe {(*node).task.take().unwrap().wake().unwrap()};

                    // we can only steal tasks if both are green
                    // XXX: deschedule is implemented horribly and should handle this stuff itself
                    Some(if !can_block && !stolen_task.can_block() {
                        Err(BlockedTask::block(stolen_task))
                    } else {
                        stolen_task.reawaken();
                        Ok(())
                    })
                }
            } else {
                // the trylock failed, so something must have the lock but not queue_lock
                //     this only happens when somebody is holding the Mutex
                //     hence, he is going to run unlock()
                //     and since queue_nonempty is true, he is going to wake up a task
                //     thus, we can happily go to sleep after the recursion check

                // ABA-safe because the trylock attempt is an acquire barrier
                if self.owner.load(atomics::Relaxed) == ourselves {
                    unsafe {self.queue_lock.unlock();}
                    fail!("attempted to lock mutex already owned by the current task");
                }

                Some(Ok(()))
            };

            if decision.is_none() {
                unsafe {self.queue_lock.unlock();}
            } else {
                let mut our_node = Node {task: None, next: 0 as *mut Node};
                if self.queue_tail == 0 as *mut *mut Node {
                    self.queue = &mut our_node as *mut Node;
                } else {
                    unsafe {*self.queue_tail = &mut our_node as *mut Node}
                }
                self.queue_tail = &mut our_node.next as *mut *mut Node;

                let t: ~Task = Local::take();
                t.deschedule(1, |task| {
                    our_node.task = Some(task);

                    unsafe {self.queue_lock.unlock();}

                    decision.take().unwrap()
                });
            }
        }
        self.owner.store(ourselves, atomics::Relaxed);
        Guard { lock: self }
    }

    fn unlock(&mut self) {
        self.owner.store(0, atomics::Relaxed);

        if support_nonblocking_tasks {
            unsafe {self.lock.unlock();}

            atomics::fence(atomics::SeqCst);

            if self.queue_nonempty.load(atomics::Relaxed) {
                unsafe {self.queue_lock.lock();}
                let node = self.queue;

                let task = if node != 0 as *mut Node && unsafe {self.lock.trylock()} {
                    let next = unsafe {(*node).next};
                    if next == 0 as *mut Node {
                        self.queue_tail = 0 as *mut *mut Node;
                        self.queue_nonempty.store(false, atomics::Relaxed);
                    }
                    self.queue = next;
                    unsafe {(*node).task.take()}
                } else {
                    None
                };
                unsafe {self.queue_lock.unlock();}
                match task {
                    Some(task) => {task.wake().map(|t| t.reawaken());},
                    None => {}
                }
            }
        } else {
            unsafe {self.lock.unlock();}
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
    pub unsafe fn destroy(&mut self) {
        self.lock.destroy()
    }
}

impl Mutex {
    /// Creates a new mutex in an unlocked state ready for use.
    pub fn new() -> Mutex {
        Mutex {
            lock: StaticMutex {
                lock: unsafe { mutex::Mutex::new() },
                owner: atomics::AtomicUint::new(0),
                queue_nonempty: atomics::AtomicBool::new(false),
                queue_lock: unsafe { mutex::Mutex::new() },
                queue: 0 as *mut Node,
                queue_tail: 0 as *mut *mut Node
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
    pub fn try_lock<'a>(&'a mut self) -> Option<Guard<'a>> {
        self.lock.try_lock()
    }

    /// Acquires a mutex, blocking the current task until it is able to do so.
    ///
    /// This function will block the local task until it is availble to acquire
    /// the mutex. Upon returning, the task is the only task with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    pub fn lock<'a>(&'a mut self) -> Guard<'a> { self.lock.lock() }
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
    use prelude::*;
    use super::{Mutex, StaticMutex, MUTEX_INIT};
    use native;

    #[test]
    fn smoke() {
        let mut m = Mutex::new();
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
        static M: uint = 10000;
        static N: uint = 3;

        fn inc() {
            for _ in range(0, M) {
                unsafe {
                    let _g = m.lock();
                    CNT += 1;
                }
            }
        }

        let (p, c) = SharedChan::new();
        for _ in range(0, N) {
            let c2 = c.clone();
            do native::task::spawn { inc(); c2.send(()); }
            let c2 = c.clone();
            do spawn { inc(); c2.send(()); }
        }

        drop(c);
        for _ in range(0, 2 * N) {
            p.recv();
        }
        assert_eq!(unsafe {CNT}, M * N * 2);
        unsafe {
            m.destroy();
        }
    }

    #[test]
    fn trylock() {
        let mut m = Mutex::new();
        assert!(m.try_lock().is_some());
    }

    #[test] #[should_fail]
    fn double_lock() {
        static mut m: StaticMutex = MUTEX_INIT;
        unsafe {
            let _g = m.lock();
            m.lock();
        }
    }
}
