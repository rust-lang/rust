// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::sync::{UnsafeAtomicRcBox};
use std::unstable::finally::Finally;
use std::unstable::atomics;
use std::cell::Cell;
use std::task;

use sync::semaphore::Semaphore;
use sync::mutex::Mutex;
use sync::unlock::Unlock;


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
    read_mode: bool,
    // The only way the count flag is ever accessed is with xadd. Since it is
    // a read-modify-write operation, multiple xadds on different cores will
    // always be consistent with respect to each other, so a monotonic/relaxed
    // consistency ordering suffices (i.e., no extra barriers are needed).
    // FIXME(#6598): The atomics module has no relaxed ordering flag, so I use
    // acquire/release orderings superfluously. Change these someday.
    read_count: atomics::AtomicUint
}

/// A blocking, no-starvation, readers-writer lock.
#[deriving(Clone)]
pub struct RWLock {
    priv order_lock: Mutex,
    priv access_lock: Semaphore,
    priv state: UnsafeAtomicRcBox<RWLockInner>
}

impl RWLock {
    /// Create a readers-writer lock.
    #[inline]
    pub fn new() -> RWLock {
        RWLock {
            order_lock: Mutex::new(),
            access_lock: Semaphore::new(),
            state: UnsafeAtomicRcBox::new(RWLockInner {
                read_mode: false,
                read_count: atomics::AtomicUint::new(0)
            })
        }
    }

    /**
     * A convenience function to wrap the more complicated (but more
     * powerful read_lock method.) Obtains a read lock, and then
     * invokes the blk argument.
     */
    #[inline]
    pub fn with_read_lock<U>(&self, blk: &fn() -> U) -> U {
        let _lock = self.read_lock();
        blk()
    }

    /**
     * A convenience function to wrap the more complicated (but more
     * powerful write_lock method.) Obtains a write lock, and then
     * invokes the blk argument.
     */
    #[inline]
    pub fn with_write_lock<U>(&self, blk: &fn() -> U) -> U {
        let _lock = self.write_lock();
        blk()
    }


    /**
     * Obtain read access to the lock.
     *
     * Other tasks can hold read access to the lock at the same
     * time. Read access to a lock prevents tasks from acquiring write
     * access. If any task has write access to the lock this method
     * blocks until the write access is released.
     */
    #[inline]
    pub fn read_lock<'r>(&'r self) -> ReadLock <'r> {
        unsafe {
            self.acquire_read_lock();
            ReadLock { rwlock: self }
        }
    }

    #[inline]
    unsafe fn acquire_read_lock(&self) {
        do task::unkillable {
            let _lock = self.order_lock.lock();

            let state = &mut *self.state.get();
            let old_count = state.read_count.fetch_add(1, atomics::Acquire);
            if old_count == 0 {
                self.access_lock.wait();
                state.read_mode = true;
            }
        }
    }

    #[inline]
    unsafe fn release_read_lock(&self) {
        do task::unkillable {
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
                self.access_lock.signal();
            }
        }
    }


    /**
     * Obtain write access to the lock.
     *
     * No other tasks access the lock while the write access is
     * held. This method blocks until other tasks are done accessing
     * the lock.
     */
    #[inline]
    pub fn write_lock<'r>(&'r self) -> WriteLock<'r> {
        unsafe {
            self.acquire_write_lock();
            WriteLock { rwlock: self, downgraded: false }
        }
    }

    #[inline]
    unsafe fn acquire_write_lock(&self) {
        do task::unkillable {
            let _lock = self.order_lock.lock();
            self.access_lock.wait()
        }
    }

    #[inline]
    unsafe fn release_write_lock(&self) {
        do task::unkillable {
            self.access_lock.signal();
        }
    }

    #[inline]
    unsafe fn downgrade_write_lock(&self) {
        do task::unkillable {
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
                self.access_lock.signal();
            }
        }
    }
}


/// A handle on a rwlock value which guarantees read access to it.
pub struct ReadLock<'self> { priv rwlock: &'self RWLock }
#[unsafe_destructor]
impl <'self> Drop for ReadLock<'self> {
    // Don't inline this due to issue #7793
    fn drop(&self) {
        unsafe {
            self.rwlock.release_read_lock()
        }
    }
}

impl <'self> Unlock for ReadLock<'self> {
    #[inline]
    pub fn unlock<U>(&mut self, blk: ~once fn() -> U) -> U {
        unsafe {
            self.rwlock.release_read_lock();
            let cell = Cell::new(blk);
            do (|| cell.take()()).finally {
                self.rwlock.acquire_read_lock()
            }
        }
    }
}



/// A handle to a rwlock which guarantees write access to it.
pub struct WriteLock<'self> { priv rwlock: &'self RWLock, priv downgraded: bool }
#[unsafe_destructor]
impl <'self> Drop for WriteLock<'self> {
    // Don't inline this due to issue #7793
    fn drop(&self) {
        unsafe {
            if !self.downgraded {
                self.rwlock.release_write_lock()
            }
        }
    }
}

impl <'self> WriteLock<'self> {
    /// Downgrade a write lock to a read lock.
    #[inline]
    pub fn downgrade(self) -> ReadLock<'self> {
        unsafe {
            let mut writelock = self;

            writelock.rwlock.downgrade_write_lock();
            writelock.downgraded = true;

            ReadLock { rwlock: writelock.rwlock }
        }
    }
}

impl <'self> Unlock for WriteLock<'self> {
    #[inline]
    pub fn unlock<U>(&mut self, blk: ~once fn() -> U) -> U {
        unsafe {
            self.rwlock.release_write_lock();
            let cell = Cell::new(blk);
            do (|| cell.take()()).finally {
                self.rwlock.acquire_write_lock()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::comm;
    use std::task;
    use std::vec;

    use sync::unlock::Unlock;


    fn nest_read_locks(nesting: uint) {
        let rwlock = RWLock::new();

        let read_locks = do vec::build_sized(nesting) |push| {
            for nesting.times {
                push(rwlock.read_lock())
            }
        };

        // Destroy the read locks
        let _lock = read_locks;
    }

    #[test]
    fn test_rwlock_can_read_lock() { nest_read_locks(1) }

    #[test]
    fn test_rwlock_can_nest_read_locks() { nest_read_locks(2) }

    #[test]
    fn test_rwlock_can_nest_read_locks_more_than_once() { nest_read_locks(5) }


    pub enum RWLockMode { Read, Write, Downgrade }

    pub fn lock_rwlock_in_mode(x: &RWLock, mode: RWLockMode, blk: &fn()) {
        match mode {
            Read => do x.with_read_lock {
                blk()
            },
            Write => do x.with_write_lock {
                blk()
            },
            Downgrade => {
                let write_lock = x.write_lock();
                let _lock = write_lock.downgrade();
                blk()
            }
        }
    }
    fn test_rwlock_handshake(x: RWLock,
                             mode1: RWLockMode,
                             mode2: RWLockMode,
                             make_mode2_go_first: bool) {
        // Much like sem_multi_resource.
        let x2 = x.clone();
        let (p1, c1) = comm::stream();
        let (p2, c2) = comm::stream();
        do task::spawn {
            if !make_mode2_go_first {
                p2.recv(); // parent sends to us once it locks, or ...
            }
            do lock_rwlock_in_mode(&x2, mode2) {
                if make_mode2_go_first {
                    c1.send(()); // ... we send to it once we lock
                }
                p2.recv();
                c1.send(());
            }
        }
        if make_mode2_go_first {
            p1.recv(); // child sends to us once it locks, or ...
        }
        do lock_rwlock_in_mode(&x, mode1) {
            if !make_mode2_go_first {
                c2.send(()); // ... we send to it once we lock
            }
            c2.send(());
            p1.recv();
        }
    }
    #[test]
    fn test_rwlock_readers_and_readers() {
        test_rwlock_handshake(RWLock::new(), Read, Read, false);
        // The downgrader needs to get in before the reader gets in, otherwise
        // they cannot end up reading at the same time.
        test_rwlock_handshake(RWLock::new(), Downgrade, Read, false);
        test_rwlock_handshake(RWLock::new(), Read, Downgrade, true);
        // Two downgrade can never both end up reading at the same time.
    }
    #[test]
    fn test_rwlock_downgrade_unlock_read() {
        let x = RWLock::new();

        {
            let write_lock = x.write_lock();
            let _lock = write_lock.downgrade();
        }

        test_rwlock_handshake(x, Read, Read, false);
    }

    #[test]
    fn test_unlock_lets_others_acquire_lock() {
        let rwlock = RWLock::new();
        let (unlocked_port, unlocked_chan) = comm::oneshot();
        let (locked_port, locked_chan) = comm::oneshot();

        let mut write_lock = rwlock.write_lock();

        do task::spawn_with((
            rwlock.clone(),
            unlocked_port,
            locked_chan
        )) |(rwlock, unlocked_port, locked_chan)| {
            comm::recv_one(unlocked_port);
            let _write_lock = rwlock.write_lock();
            comm::send_one(locked_chan, ())
        }

        do write_lock.unlock {
            comm::send_one(unlocked_chan, ());
            comm::recv_one(locked_port)
        }
    }

    #[ignore(cfg(windows))]
    mod try_supporting_platforms_only {
        use super::super::*;
        use super::*;

        use std::task;


        fn rwlock_kill_helper(mode1: RWLockMode, mode2: RWLockMode) {
            // Mutex must get automatically unlocked if failed/killed within.
            let mutex = RWLock::new();

            let mutex_2 = mutex.clone();
            let result: Result<(),()> = do task::try {
                do lock_rwlock_in_mode(&mutex_2, mode1) {
                    fail!();
                }
            };
            assert!(result.is_err());

            // child task must have finished by the time try returns
            do lock_rwlock_in_mode(&mutex, mode2) { }
        }

        #[test]
        fn test_rwlock_reader_killed_writer() {
            rwlock_kill_helper(Read, Write)
        }
        #[test]
        fn test_rwlock_writer_killed_reader() {
            rwlock_kill_helper(Write, Read)
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
        fn test_rwlock_downgrader_killed_read() {
            rwlock_kill_helper(Downgrade, Read)
        }
        #[test]
        fn test_rwlock_read_killed_downgrader() {
            rwlock_kill_helper(Read, Downgrade)
        }
        #[test]
        fn test_rwlock_downgrader_killed_writer() {
            rwlock_kill_helper(Downgrade, Write)
        }
        #[test]
        fn test_rwlock_writer_killed_downgrader() {
            rwlock_kill_helper(Write, Downgrade)
        }
    }
}