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
// This is the high-level implementation of the mutexes, but the nitty gritty
// details can be found in the code below.

use std::rt::local::Local;
use std::rt::task::{BlockedTask, Task};
use std::rt::thread::Thread;
use std::sync::atomics;
use std::unstable::mutex;

use q = sync::mpsc_intrusive;

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
    /// Internal queue that all green threads will be blocked on.
    priv q: q::Queue<uint>,
    /// Dubious flag about whether this mutex is held or not. You might be
    /// thinking "this is impossible to manage atomically," and you would be
    /// correct! Keep on reading!
    priv held: atomics::AtomicBool,
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
    held: atomics::INIT_ATOMIC_BOOL,
    q: q::Queue {
        head: atomics::INIT_ATOMIC_UINT,
        tail: 0 as *mut q::Node<uint>,
        stub: q::DummyNode {
            next: atomics::INIT_ATOMIC_UINT,
        }
    }
};

impl StaticMutex {
    /// Attempts to grab this lock, see `Mutex::try_lock`
    pub fn try_lock<'a>(&'a mut self) -> Option<Guard<'a>> {
        if unsafe { self.lock.trylock() } {
            self.held.store(true, atomics::Release); // see below
            Some(Guard{ lock: self })
        } else {
            None
        }
    }

    /// Acquires this lock, see `Mutex::lock`
    pub fn lock<'a>(&'a mut self) -> Guard<'a> {
        // Remember that an explicit goal of these mutexes is to be "just as
        // fast" as pthreads. Note that at some point our implementation
        // requires an answer to the question "can we block" and implies a hit
        // to OS TLS. In attempt to avoid this hit and to maintain efficiency in
        // the uncontended case (very important) we start off by hitting a
        // trylock on the OS mutex. If we succeed, then we're lucky!
        if unsafe { self.lock.trylock() } {
            self.held.store(true, atomics::Release); // see below
            return Guard{ lock: self }
        }

        let t: ~Task = Local::take();
        if t.can_block() {
            // Tasks which can block are super easy. These tasks just accept the
            // TLS hit we just made, and then call the blocking `lock()`
            // function. Turns out the TLS hit is essentially 0 on contention.
            Local::put(t);
            unsafe { self.lock.lock(); }
            self.held.store(true, atomics::Release); // see below
        } else {
            // And here's where we come to the "fun part" of this
            // implementation. Contention with a green task is fairly difficult
            // to resolve. The goal here is to push ourselves onto the internal
            // queue, but still be able to "cancel" our enqueue in case the lock
            // was dropped while we were doing our business.
            //
            // The pseudocode for this is:
            //
            //      let mut node = ...;
            //      push(node)
            //      if trylock() {
            //          wakeup(pop())
            //      } else {
            //          node.sleep()
            //      }
            //
            // And the pseudocode for the wakeup protocol is:
            //
            //      match pop() {
            //          Some(node) => node.wakeup(),
            //          None => lock.unlock()
            //      }
            //
            // Note that a contended green thread does *not* re-acquire the
            // mutex because ownership was silently transferred to it. You'll
            // note a fairly large race condition here, which is that whenever
            // the OS mutex is unlocked, "just before" it's unlocked a green
            // thread can fly in and block itself. This turns out to be a
            // fundamental problem with any sort of attempt to arbitrate among
            // the unlocker and a locking green thread.
            //
            // One possible solution for this is to attempt to re-acquire the
            // lock during the unlock procedure. This is less than ideal,
            // however, because it means that the memory of a mutex must be
            // guaranteed to be valid until *all unlocks* have returned. That's
            // normally the job of the mutex itself, so it can be seen that
            // touching a mutex again after it has been unlocked is an unwise
            // decision.
            //
            // Another alternative solution (and the one implemented) is more
            // distasteful, but functional. You'll notice that the struct
            // definition has a `held` flag, which is impossible to maintain
            // atomically. For our usage, the flag is set to `true` immediately
            // after a mutex is acquired and set to `false` at the *beginning*
            // of an unlock.
            //
            // By doing this, we're essentially instructing green threads to
            // "please spin" while another thread is in the middle of performing
            // the unlock procedure. Again, this is distasteful, but within the
            // constraints that we're working in I found it difficult to think
            // of other courses of action.
            let mut node = q::Node::new(0);
            t.deschedule(1, |task| {
                unsafe {
                    node.data = task.cast_to_uint();
                    self.q.push(&mut node);
                }

                let mut stolen = false;
                // Spinloop attempting to grab a mutex while someone's unlocking
                // the mutex. While it's not held and we fail the trylock, the
                // best thing we can do is hope that our yield will run the
                // unlocker before us (note that bounded waiting is shattered
                // here for green threads).
                while !self.held.load(atomics::SeqCst) {
                    if unsafe { self.lock.trylock() } {
                        self.held.store(true, atomics::Release);
                        stolen = true;
                        break
                    } else {
                        Thread::yield_now();
                    }
                }

                // If we managed to steal the lock, then we need to wake up a
                // thread. Note that we may not have acquired the mutex for
                // ourselves (we're not guaranteed to be the head of the queue).
                // The good news is that we *are* guaranteed to have a non-empty
                // queue. This is because if we acquired the mutex no one could
                // have transferred it to us (hence our own node must still be
                // on the queue).
                //
                // The queue itself can return `None` from a pop when there's
                // data on the queue (a known limitation of the queue), so here
                // you'll find the second spin loop (which is in theory even
                // rarer than the one above).
                if stolen {
                    let locker;
                    loop {
                        match unsafe { self.q.pop() } {
                            Some(t) => { locker = t; break }
                            None => Thread::yield_now()
                        }
                    }
                    Err(unsafe { BlockedTask::cast_from_uint((*locker).data) })
                } else {
                    Ok(())
                }
            });
            assert!(self.held.load(atomics::SeqCst));
        }

        Guard { lock: self }
    }

    fn unlock(&mut self) {
        // As documented above, we *initially* flag our mutex as unlocked in
        // order to allow green threads just starting to block to realize that
        // they shouldn't completely block.
        assert!(self.held.load(atomics::SeqCst));
        self.held.store(false, atomics::Release);

        // Remember that the queues we are using may return None when there is
        // indeed data on the queue. In this case, we can just safely ignore it.
        // The reason for this ignorance is that a value of `None` with data on
        // the queue means that the "head popper" hasn't finished yet. We've
        // already flagged our mutex as acquire-able, so the "head popper" will
        // see this and attempt to grab the mutex (or someone else will steal it
        // and this whole process will begin anew).
        match unsafe { self.q.pop() } {
            Some(t) => {
                self.held.store(true, atomics::Release);
                let task = unsafe { BlockedTask::cast_from_uint((*t).data) };
                task.wake().map(|t| t.reawaken());
            }
            None => unsafe { self.lock.unlock() }
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
                held: atomics::AtomicBool::new(false),
                q: q::Queue::new(),
                lock: unsafe { mutex::Mutex::new() },
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
    extern mod native;
    use super::{Mutex, StaticMutex, MUTEX_INIT};

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
        let _g = m.lock();
        m.lock();
    }
}
