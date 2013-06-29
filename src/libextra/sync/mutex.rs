// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::finally::Finally;
use std::cell::Cell;

use sync::semaphore::Semaphore;
use sync::unlock::Unlock;


/// A blocking, bounded-waiting, mutual exclusion lock.
#[deriving(Clone)]
pub struct Mutex {
    priv semaphore: Semaphore
}

impl Mutex {
    /// Create a mutex.
    #[inline]
    pub fn new() -> Mutex {
        Mutex { semaphore: Semaphore::new() }
    }

    /**
     * A convenience function to wrap the more complicated (but more
     * powerful lock method.) Obtains a lock, and then invokes the blk
     * argument.
     */
    #[inline]
    pub fn with_lock<U>(&self, blk: &fn() -> U) -> U {
        let _lock = self.lock();
        blk()
    }


    /**
     * Obtain a mutual exclusion lock on the mutex.
     *
     * No other code can obtain a lock until the returned lock is
     * released. If other code has obtained a lock this method blocks
     * until the lock is released, and then obtains a lock.
     */
    #[inline]
    pub fn lock<'r>(&'r self) -> Lock<'r> {
        self.acquire_lock();
        Lock { mutex: self }
    }

    #[inline]
    fn acquire_lock(&self) { self.semaphore.wait(); }

    #[inline]
    fn release_lock(&self) { self.semaphore.signal(); }
}


/// A handle that guarantees that an associated mutex is in a locked
/// (exclusively accessed) state.
pub struct Lock<'self> { priv mutex: &'self Mutex }

#[unsafe_destructor]
impl <'self> Drop for Lock<'self> {
    // Don't inline this due to issue #7793
    fn drop(&self) {
        self.mutex.release_lock()
    }
}

impl <'self> Unlock for Lock<'self> {
    #[inline]
    pub fn unlock<U>(&mut self, blk: ~once fn() -> U) -> U {
        self.mutex.release_lock();
        let cell = Cell::new(blk);
        do (|| cell.take()()).finally {
            self.mutex.acquire_lock()
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use std::comm;
    use std::task;
    use std::vec;

    use sync::wait_queue::WaitQueue;


    #[test]
    fn test_mutex_lock() {
        // Unsafely achieve shared state, and do the textbook
        // "load tmp = move ptr; inc tmp; store ptr <- tmp" dance.
        let (p, c) = comm::stream();
        let mutex = Mutex::new();
        let mutex_2 = mutex.clone();
        let mut sharedstate = ~0;
        {
            let ptr: *mut int = &mut *sharedstate;
            do task::spawn {
                let sharedstate: &mut int = unsafe { &mut *ptr };
                access_shared(sharedstate, &mutex_2, 10);
                c.send(());

            }
        }
        {
            access_shared(sharedstate, &mutex, 10);
            p.recv();

            assert_eq!(*sharedstate, 20);
        }

        fn access_shared(sharedstate: &mut int, m: &Mutex, n: uint) {
            for n.times {
                let _lock = m.lock();
                let oldval = *sharedstate;
                task::yield();
                *sharedstate = oldval + 1;
            }
        }
    }

    #[test]
    fn test_mutex_child_wakes_up_parent() {
        let mutex = Mutex::new();

        let mut lock = mutex.lock();
        let condition = WaitQueue::new();

        do task::spawn_with((
            mutex.clone(),
            condition.clone()
        )) |(mutex, condition)| {
            // Wait until parent's lock is released to avoid a race
            // condition
            do mutex.with_lock { }

            let woke_up_parent = condition.signal();
            assert!(woke_up_parent);
        }
        condition.wait_with(&mut lock);
    }

    fn test_mutex_parent_wakes_up_child() {
        let (port, chan) = comm::stream();

        let mutex = Mutex::new();

        let condition = WaitQueue::new();

        do task::spawn_with(condition.clone()) |condition| {
            let mut lock = mutex.lock();
            chan.send(());
            condition.wait_with(&mut lock);
            chan.send(());
        }

        port.recv(); // Wait until child gets in the mutex

        let woken = condition.signal();
        assert!(woken);

        port.recv(); // Wait until child wakes up
    }

    fn test_mutex_cond_broadcast_helper(num_waiters: uint) {
        let mutex = Mutex::new();
        let condition = WaitQueue::new();

        let ports = do vec::build_sized(num_waiters) |ports_push| {
            for num_waiters.times {
                let (port, chan) = comm::stream();
                ports_push(port);
                do task::spawn_with((
                    mutex.clone(),
                    condition.clone()
                )) |(mutex, condition)| {
                    let mut lock = mutex.lock();
                    chan.send(());
                    condition.wait_with(&mut lock);
                    chan.send(());
                }
            }
        };

        // wait until all children get in the mutex
        for ports.iter().advance |port| { port.recv(); }
        let num_woken = condition.broadcast();
        assert_eq!(num_woken, num_waiters);
        // wait until all children wake up
        for ports.iter().advance |port| { port.recv(); }
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
        let mutex = Mutex::new();
        let mutex2 = mutex.clone();

        let condition = WaitQueue::new();

        do task::try {
            mutex2.lock();
        };

        assert!(!condition.signal());
    }

    #[ignore(cfg(windows))]
    mod tests_for_kill_supporting_platforms {
        use super::super::*;

        use std::task;
        use std::comm;
        use std::cell::Cell;

        use sync::wait_queue::WaitQueue;


        #[test]
        fn test_mutex_killed_simple() {
            // Mutex must get automatically unlocked if failed/killed within.
            let mutex = Mutex::new();

            let mutex2 = mutex.clone();
            let result: Result<(),()> = do task::try {
                let _lock = mutex2.lock();
                fail!()
            };
            assert!(result.is_err());

            // child task must have finished by the time try returns
            mutex.lock();
        }
        #[test]
        fn test_mutex_killed_cond() {
            // Getting killed during cond wait must not corrupt the mutex while
            // unwinding (e.g. double unlock).
            let mutex = Mutex::new();
            let condition = WaitQueue::new();

            let mutex_2 = mutex.clone();
            let condition_2 = WaitQueue::new();
            let result: Result<(),()> = do task::try {
                let (p, c) = comm::stream();
                do task::spawn { // linked
                    p.recv(); // wait for sibling to get in the mutex
                    task::yield();
                    fail!();
                }

                let mut lock = mutex_2.lock();
                c.send(()); // tell sibling go ahead
                condition_2.wait_with(&mut lock); // block forever
            };
            assert!(result.is_err());

            // child task must have finished by the time try returns
            mutex.lock();

            let woken = condition.signal();
            assert!(!woken);
        }
        #[test]
        fn test_mutex_killed_broadcast() {
            let mutex = Mutex::new();

            let condition = WaitQueue::new();
            let (p, c) = comm::stream();

            let mutex_2 = mutex.clone();
            let condition_2 = condition.clone();
            let result: Result<(),()> = do task::try {
                let mut sibling_convos = ~[];
                for 2.times {
                    let (p,c) = comm::stream();
                    let c = Cell::new(c);
                    sibling_convos.push(p);
                    let mutex_i = mutex_2.clone();
                    let condition_i = condition_2.clone();
                    // spawn sibling task
                    do task::spawn { // linked
                        let mut lock = mutex_i.lock();
                        let c = c.take();
                        c.send(()); // tell sibling to go ahead
                        let _z = SendOnFailure(c);
                        condition_i.wait_with(&mut lock); // block forever
                    }
                }
                for sibling_convos.iter().advance |p| {
                    p.recv(); // wait for sibling to get in the mutex
                }
                mutex_2.lock();
                c.send(sibling_convos); // let parent wait on all children
                fail!();
            };
            assert!(result.is_err());
            // child task must have finished by the time try returns
            let r = p.recv();
            for r.iter().advance |p| { p.recv(); } // wait on all its siblings


            let woken = condition.broadcast();
            assert_eq!(woken, 0);

            struct SendOnFailure {
                c: comm::Chan<()>,
            }

            impl Drop for SendOnFailure {
                fn drop(&self) {
                    self.c.send(());
                }
            }

            fn SendOnFailure(c: comm::Chan<()>) -> SendOnFailure {
                SendOnFailure {
                    c: c
                }
            }
        }
        #[test] #[should_fail]
        fn test_mutex_different_conds() {
            let condition_1 = WaitQueue::new();
            let condition_2 = WaitQueue::new();
            let mutex = Mutex::new();

            let (p, c) = comm::stream();

            do task::spawn_with(mutex.clone()) |mutex| {
                let mut lock = mutex.lock();
                c.send(());
                condition_2.wait_with(&mut lock);
            }

            p.recv();

            assert!(condition_1.signal())
        }
    }
}