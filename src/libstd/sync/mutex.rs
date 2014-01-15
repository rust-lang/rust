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
// ## Mutexes, take 3
//
// Take 3 uses a more sophisticated atomic state, allowing it to not use yield loops:
// we use an atomic integer containing a (queue_size, lockers) tuple, where queue_size
// is the size of the queue of queued up tasks, and lockers is the number of tasks who
// have or are about to take the OS mutex using a blocking lock call.
//
// It is now as fair as the OS mutex allows, even when mixing green and native tasks,
// since native threads will queue like green tasks, if any green task is queued.
//
// This is the high-level implementation of the mutexes, but the nitty gritty
// details can be found in the code below.

use ops::Drop;
use q = sync::mpsc_intrusive;
use option::{Option, Some, None};
use result::{Ok, Err};
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
    /// Internal mutex state
    priv state: MutexState,
    /// Internal queue that all green threads will be blocked on.
    priv q: q::Queue<uint>,
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
pub struct Guard<'a> {
    priv lock: &'a StaticMutex,
}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
pub static MUTEX_INIT: StaticMutex = StaticMutex {
    lock: mutex::MUTEX_INIT,
    state: INIT_MUTEX_STATE,
    q: q::Queue {
        producer: atomics::INIT_ATOMIC_UINT,
        consumer: 0 as *mut q::Node<uint>,
    }
};

/// this is logically an atomic tuple of (lockers, queue_size)
/// lockers is the number of tasks about to call lock() or holding the mutex
/// queue_size is the number of queued up tasks
struct MutexState {
    priv state: atomics::AtomicUint // XXX: this needs to become AtomicU64
}

static INIT_MUTEX_STATE: MutexState = MutexState {state: atomics::INIT_ATOMIC_UINT};

static LOCKERS_SHIFT: uint = 0;
// XXX: this limits 32-bit tasks to 2^16; we need to use 64-bit atomics on 32-bit too to fix this
#[cfg(target_word_size = "32")] static QUEUE_SIZE_SHIFT: uint = 16;
#[cfg(target_word_size = "64")] static QUEUE_SIZE_SHIFT: uint = 32;

static LOCKERS_MASK: uint = (1 << QUEUE_SIZE_SHIFT) - (1 << LOCKERS_SHIFT);
static QUEUE_SIZE_MASK: uint = -(1 << QUEUE_SIZE_SHIFT);

impl MutexState {
    pub fn new() -> MutexState {
        MutexState {state: atomics::AtomicUint::new(0)}
    }

    // if queue_size == 0 {++lockers; true} else {false}
    pub fn should_lock(&self) -> bool {
        // optimistically speculate we have no contention
        let mut a = self.state.compare_and_swap(0, (1 << LOCKERS_SHIFT), atomics::SeqCst);
        if a == 0 {return true;}

        loop {
            let (b, r) = if (a & QUEUE_SIZE_MASK) != 0 {
                return false;
            } else {
                (a + (1 << LOCKERS_SHIFT), true)
            };
            let v = self.state.compare_and_swap(a, b, atomics::SeqCst);
            if a == v {return r;}
            a = v;
        }
    }

    // ++queue_size; if(lockers == 0) {++lockers; true} else {false}
    pub fn queue_and_should_lock(&self) -> bool {
        // optimistically speculate we have only green tasks and nothing MUST_QUEUE
        let mut a = self.state.compare_and_swap((1 << LOCKERS_SHIFT),
            (1 << LOCKERS_SHIFT) + (1 << QUEUE_SIZE_SHIFT), atomics::SeqCst);
        if a == (1 << LOCKERS_SHIFT) {return false;}

        loop {
            let (b, r) = if (a & LOCKERS_MASK) == 0 {
                (a + (1 << LOCKERS_SHIFT) + (1 << QUEUE_SIZE_SHIFT), true)
            } else {
                (a + (1 << QUEUE_SIZE_SHIFT), false)
            };
            let v = self.state.compare_and_swap(a, b, atomics::SeqCst);
            if a == v {return r;}
            a = v;
        }
    }

    // --queue_size;
    pub fn dequeue(&self) {
        self.state.fetch_sub((1 << QUEUE_SIZE_SHIFT), atomics::SeqCst);
    }

    // if(queue_size != 0 && lockers == 1) {--queue_size; true} else {--lockers; false}
    pub fn should_dequeue(&self) -> bool {
        // optimistically speculate we have no contention
        let mut a = self.state.compare_and_swap((1 << LOCKERS_SHIFT), 0, atomics::SeqCst);
        if a == (1 << LOCKERS_SHIFT) {return false;}

        loop {
            let (b, r) = if ((a & LOCKERS_MASK) == (1 << LOCKERS_SHIFT)
                && (a & QUEUE_SIZE_MASK) != 0) {
                (a - (1 << QUEUE_SIZE_SHIFT), true)
            } else {
                (a - (1 << LOCKERS_SHIFT), false)
            };
            let v = self.state.compare_and_swap(a, b, atomics::SeqCst);
            if a == v {return r;}
            a = v;
        }
    }

    // queue_size == 0 && lockers == 0
    pub fn can_try_lock(&self) -> bool {
        self.state.load(atomics::SeqCst) == 0
    }
}

// try_lock() {
//     if atomically {queue_size == 0 && lockers == 0} && lock.try_lock() {
//         if atomically {if queue_size == 0 {++lockers; true} else {false}} {
//             ok
//         } else {
//             lock.unlock()
//             fail
//         }
//     } else {
//         fail
//     }
// }
//
// lock() {
//     if try_lock() {
//         return guard;
//     }
//
//     if can_block && atomically {if queue_size == 0 {++lockers; true} else {false}} {
//         lock.lock();
//     } else {
//         q.push();
//         if atomically {++queue_size; if(lockers == 0) {++lockers; true} else {false}} {
//             // this never blocks indefinitely
//             // this is because lockers was 0, so we have no one having or trying to get the lock
//             // and we atomically set queue_size to a positive value, so no one will start blocking
//             lock.lock();
//             atomically {--queue_size}
//             t = q.pop();
//             if t != ourselves {
//                 t.wakeup();
//                 go to sleep
//             }
//         } else {
//             go to sleep
//         }
//     }
// }
//
// unlock() {
//     if atomically
//             {if(queue_size != 0 && lockers == 1) {--queue_size; true} else {--lockers; false}}
//      {
//         t = q.pop();
//         t.wakeup();
//     } else {
//         lock.unlock()
//     }
// }
impl StaticMutex {
    /// Try to acquire this lock, see `Mutex::try_lock`
    fn try_lock<'a>(&'a self) -> Option<Guard<'a>> {
        // note that we can't implement this by first calling should_lock()
        // and then try_lock(), because once should_lock() succeeds we have
        // committed to waking up tasks, and we can only do that by blocking on the mutex

        // also, this is the only place in the Mutex code where we aren't guaranteed that a
        // Task structure exists, and thus the task number limit doesn't limit this
        // however, we don't change self.state unless we manage to get the lock, so this
        // can only account for a single extra "task without ~Task", which is accounted by
        // having the Task limit be (1 << 16) - 2 or (1 << 32) - 2
        if self.state.can_try_lock() && unsafe { self.lock.trylock() } {
            // here we have the lock, but haven't told anyone about it
            // this means that a green task might be blocking expecting to get the lock
            // so if queue_size != 0 we abort and unlock, otherwise atomically increasing lockers

            // this is the same code used for the blocking lock(), because since we have the lock
            // already, we don't care have the problem of possibly "blocking" on other tasks
            if self.state.should_lock() {
                 Some(Guard{ lock: self })
            } else {
                // oops, we shouldn't have taken the lock because a task got queued in between
                // just unlock it and return failure, no one will know since we changed no state
                unsafe { self.lock.unlock(); }
                None
            }
        } else {
            None
        }
    }

    /// Acquires this lock, see `Mutex::lock`
    pub fn lock<'a>(&'a self) -> Guard<'a> {
        // Remember that an explicit goal of these mutexes is to be "just as
        // fast" as pthreads. Note that at some point our implementation
        // requires an answer to the question "can we block" and implies a hit
        // to OS TLS. In attempt to avoid this hit and to maintain efficiency in
        // the uncontended case (very important) we start off by hitting a
        // trylock on the OS mutex. If we succeed, then we're lucky!
        match self.try_lock() {
            Some(guard) => {return guard; },
            None => {}
        }

        let t: ~Task = Local::take();
        let can_block = t.can_block();
        if can_block && self.state.should_lock() {
            // Tasks which can block are super easy. These tasks just accept the
            // TLS hit we just made, and then call the blocking `lock()`
            // function. Turns out the TLS hit is essentially 0 on contention.
            Local::put(t);
            unsafe { self.lock.lock(); }
        } else {
            let mut our_node = q::Node::new(0);
            t.deschedule(1, |task| {
                our_node.data = unsafe { task.cast_to_uint() };
                unsafe { self.q.push(&mut our_node); }

                if self.state.queue_and_should_lock() {
                    // this code generally only gets executed in a race window, since typically
                    // either the trylock succeeds, and we return early, or we have someone else
                    // running (lockers != 0), so we take the other branch of this if and wait
                    // for someone else to wake us up
                    //
                    // in particular, this code only runs if someone unlocked the mutex between
                    // the try_lock and the self.state.queue_and_should_lock above
                    unsafe { self.lock.lock(); }
                    self.state.dequeue();

                    let node = unsafe { self.q.pop() }.expect("the queue is empty but queue_size was != 0");

                    // If we popped ourselves, then we just unblock. If it's someone
                    // else, we wake up the task and go to sleep
                    if node == &mut our_node as *mut q::Node<uint> {
                        Err(unsafe { BlockedTask::cast_from_uint(our_node.data) })
                    } else {
                        unsafe { BlockedTask::cast_from_uint((*node).data) }.wake().map(|t| t.reawaken());
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            });
        }

        Guard { lock: self }
    }

    fn unlock(&self) {
        // If we are the only locker and someone is queued, dequeue and wake them up
        // otherwise unlock, either to let another locker run, or to completely unlock the mutex

        // This allows to preserve fairness, by prioritizing tasks acquiring the OS mutex over
        // queued up task.
        // Note that once the queue is non-empty, everyone will queue, so fairness is preserved
        // in the other sense too.
        if self.state.should_dequeue() {
            let node = unsafe { self.q.pop() }.expect("the queue is empty but queue_size was != 0");
            unsafe { BlockedTask::cast_from_uint((*node).data) }.wake().map(|t| t.reawaken());
        } else {
            unsafe { self.lock.unlock(); }
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
            lock: StaticMutex {
                state: MutexState::new(),
                q: q::Queue::new(),
                lock: unsafe { mutex::Mutex::new() },
            }
        }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `Err(self)` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    pub fn try_lock<'a>(&'a self) -> Option<Guard<'a>> {
        self.lock.try_lock()
    }

    /// Acquires a mutex, blocking the current task until it is able to do so.
    ///
    /// This function will block the local task until it is availble to acquire
    /// the mutex. Upon returning, the task is the only task with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    pub fn lock<'a>(&'a self) -> Guard<'a> { self.lock.lock() }
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
}
