// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::sync::UnsafeAtomicRcBox;
use std::cast;
use std::task;

use sync::rwlock::{RWLock, ReadLock, WriteLock};
use sync::unlock::Unlock;


struct RWArcInner<T> { rwlock: RWLock, contents: T, failed: bool }
impl <T> RWArcInner<T> {
    #[inline]
    fn new(initial_value: T) -> RWArcInner<T> {
        RWArcInner {
            rwlock: RWLock::new(),
            contents: initial_value,
            failed: false
        }
    }

    #[inline]
    fn assert_not_failed(&self) {
        if self.failed {
            fail!("Poisoned RWArc - another task failed inside!")
        }
    }

    #[inline]
    fn read_lock<'r>(&'r self) -> ReadLock<'r> {
        let read_lock = self.rwlock.read_lock();
        self.assert_not_failed();
        read_lock
    }

    #[inline]
    fn write_lock<'r>(&'r self) -> WriteLock<'r> {
        let write_lock = self.rwlock.write_lock();
        self.assert_not_failed();
        write_lock
    }
}


/**
 * A dual-mode atomically referenced counted value protected by a
 * reader-writer lock. The data can be accessed mutably or immutably,
 * and immutably-accessing tasks may run concurrently.
 */
pub struct RWArc<T> { priv contents: UnsafeAtomicRcBox<RWArcInner<T>> }
impl <T: Freeze + Send> RWArc<T> {
    /// Create an RWArc.
    #[inline]
    pub fn new(initial_value: T) -> RWArc<T> {
        let data = RWArcInner::new(initial_value);
        RWArc { contents: UnsafeAtomicRcBox::new(data) }
    }


    /**
     * A convenience function to wrap the more complicated (but more
     * powerful read_locked method.) Obtains a read lock, accesses the
     * value, and then invokes the blk argument.
     */
    #[inline]
    pub fn read<U>(&mut self, blk: &fn(&T) -> U) -> U {
        let mut locked = self.read_locked();
        blk(locked.get())
    }

    /**
     * A convenience function to wrap the more complicated (but more
     * powerful write_locked method.) Obtains a write lock, accesses
     * the value, and then invokes the blk argument.
     */
    #[inline]
    pub fn write<U>(&mut self, blk: &fn(&mut T) -> U) -> U {
        let mut locked = self.write_locked();
        blk(locked.get())
    }


    /**
     * Obtain permission to read from the underlying mutable data.
     *
     * # Failure
     * If the RWArc is poisoned then this method will fail.
     */
    #[inline]
    pub fn read_locked<'r>(&'r mut self) -> ReadLocked<'r, T> {
        unsafe {
            let state = &mut *self.contents.get();
            ReadLocked {
                rwarc: self,
                lock: state.read_lock()
            }
        }
    }

    /**
     * Access the underlying mutable data with mutual exclusion from
     * other tasks. The RWArc be locked until the access cookie is
     * released; all other tasks wishing to access the data will block
     * until the cookie is released.
     *
     * # Failure
     *
     * Failing while inside the RWArc will unlock the RWArc while
     * unwinding, so that other tasks won't block forever. It will
     * also poison the RWArc: any tasks that subsequently try to
     * access it (including those already blocked on the RWArc) will
     * also fail immediately.
     */
    #[inline]
    pub fn write_locked<'r>(&'r mut self) -> WriteLocked<'r, T> {
        unsafe {
            let state = &mut *self.contents.get();
            WriteLocked {
                inner: WriteLockedInner { rwarc: self },
                lock: state.write_lock()
            }
        }
    }

    #[inline]
    unsafe fn get<'r>(&'r mut self) -> &'r mut T {
        let shared_mut_value = &'r mut *self.contents.get();
        &'r mut shared_mut_value.contents
    }
}

impl <T: Freeze + Send> Clone for RWArc<T> {
    #[inline]
    fn clone(&self) -> RWArc<T> {
        RWArc { contents: self.contents.clone() }
    }
}

/// A handle to an atomically reference counted value that has a read
/// lock on it.
pub struct ReadLocked<'self, T> {
    priv lock: ReadLock<'self>,
    priv rwarc: &'self mut RWArc<T>
}

impl <'self, T: Freeze + Send> ReadLocked<'self, T> {
    /// Access the data of the read locked value
    #[inline]
    pub fn get(&'self mut self) -> &'self T {
        unsafe {
            let immut_pointer: &'self T = self.rwarc.get();
            immut_pointer
        }
    }
}

impl <'self, T: Send> Unlock for ReadLocked<'self, T> {
    #[inline]
    pub fn unlock<V>(&mut self, blk: ~once fn() -> V) -> V {
        let result = self.lock.unlock(blk);
        unsafe {
            let state = &*self.rwarc.contents.get();
            state.assert_not_failed();
        }
        result
    }
}


/// A handle to an atomically reference counted value that has a write
/// lock on it
pub struct WriteLocked<'self, T> {
    priv lock: WriteLock<'self>,
    priv inner: WriteLockedInner<'self, T>
}

// Can't just be a newtype due to issue 7899
struct WriteLockedInner<'self, T> { rwarc: &'self mut RWArc<T> }

#[unsafe_destructor]
impl <'self, T: Freeze + Send> Drop for WriteLockedInner<'self, T> {
    // Don't inline this due to issue #7793
    fn drop(&self) {
        unsafe {
            if task::failing() {
                let myself = cast::transmute_mut(self);
                let state = &mut *myself.rwarc.contents.get();
                state.failed = true
            }
        }
    }
}

impl <'self, T: Freeze + Send> WriteLocked<'self, T> {
    /// Access the data behind a write locked value.
    #[inline]
    pub fn get(&'self mut self) -> &'self mut T {
        unsafe {
            let mut_pointer: &'self mut T = self.inner.rwarc.get();
            mut_pointer
        }
    }

    /// Consumes a write locked value, and downgrades it to a read
    /// locked value.
    #[inline]
    pub fn downgrade(self) -> ReadLocked<'self, T> {
        let WriteLocked { inner, lock } = self;
        ReadLocked {
            rwarc: inner.rwarc,
            lock: lock.downgrade()
        }
    }
}

impl <'self, T: Freeze + Send> Unlock for WriteLocked<'self, T> {
    #[inline]
    pub fn unlock<V>(&mut self, blk: ~once fn() -> V) -> V {
        let result = self.lock.unlock(blk);
        unsafe {
            let state = &*self.inner.rwarc.contents.get();
            state.assert_not_failed();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::comm;
    use std::task;

    use sync::wait_queue::WaitQueue;


    #[test]
    fn test_readers_can_not_read_during_writes() {
        /*
        This is fundamentally a statistical test. There are a large
        number of ways writes can be combined with reads. This test
        attempts to explore that space of possibilities, and possible
        finds a few incorrect situations.
        */
        let number_of_writes_run: uint = 10;
        let readers: uint = 5;

        let mut arc: RWArc<int> = RWArc::new(0);

        // Spawn readers that try to catch the main thread in the act
        // of writing.
        let mut child_tasks = ~[];
        for readers.times {
            let mut builder = task::task();
            builder.future_result(|m| child_tasks.push(m));
            do builder.spawn_with(arc.clone()) |mut arc| {
                let mut num = arc.read_locked();
                assert!(*num.get() >= 0);
            }
        };

        // Write to the arc, and see if it will be caught.
        {
            let mut write_locked = arc.write_locked();
            let num = write_locked.get();
            for number_of_writes_run.times {
                let tmp = *num;
                *num = -1;
                task::yield();
                *num = tmp + 1;
            }
        }

        // Wait for child tasks to finish
        for child_tasks.iter().advance |r| {
            r.recv();
        }

        // Do a sanity check
        let mut num = arc.read_locked();
        assert_eq!(*num.get(), number_of_writes_run as int);
    }

    fn test_rw_write_cond_downgrade_read_race_helper() {
        // Tests that when a downgrader hands off the "reader cloud" lock
        // because of a contending reader, a writer can't race to get it
        // instead, which would result in readers_and_writers. This tests
        // the sync module rather than this one, but it's here because an
        // rwarc gives us extra shared state to help check for the race.
        // If you want to see this test fail, go to sync.rs and replace the
        // line in RWlock::write_cond() that looks like:
        //     "blk(&Condvar { order: opt_lock, ..*cond })"
        // with just "blk(cond)".
        let condition = WaitQueue::new();
        let mut arc = RWArc::new(true);
        let (wp, wc) = comm::stream();

        // writer task
        do task::spawn_with((
            arc.clone(),
            condition.clone()
        )) |mut (arc, condition)| {
            let mut write_locked = arc.write_locked();
            wc.send(()); // tell downgrader it's ok to go

            condition.wait_with(&mut write_locked);

            // The core of the test is here: the condvar reacquire path
            // must involve order_lock, so that it cannot race with a reader
            // trying to receive the "reader cloud lock hand-off".
            *write_locked.get() = false;
        }

        wp.recv(); // wait for writer to get in

        {
            let arc2 = arc.clone();
            let write_locked = arc.write_locked();

            // make writer contend in the cond-reacquire path
            condition.signal();

            // make a reader task to trigger the "reader cloud lock" handoff
            let (rp, rc) = comm::stream();
            do task::spawn_with(arc2) |mut arc| {
                rc.send(());
                let _lock = arc.read_locked();
            }
            rp.recv(); // wait for reader task to exist

            let mut read_locked = write_locked.downgrade();

            // if writer mistakenly got in, make sure it mutates state
            // before we assert on it
            for 5.times { task::yield() }
            // make sure writer didn't get in.
            assert!(*read_locked.get());
        }
    }

    #[test]
    fn test_rw_write_cond_downgrade_read_race() {
        // Ideally the above test case would have yield statements in it that
        // helped to expose the race nearly 100% of the time... but adding
        // yields in the intuitively-right locations made it even less likely,
        // and I wasn't sure why :( . This is a mediocre "next best" option.
        for 8.times { test_rw_write_cond_downgrade_read_race_helper() }
    }

    #[ignore(cfg(windows))]
    mod tests_for_fail_supported_platforms {
        use super::super::*;

        use std::cell::Cell;
        use std::task;


        #[test] #[should_fail]
        fn test_failed_writes_poison_reads() {
            let mut arc = RWArc::new(());
            let cell = Cell::new(arc.clone());

            let _: Result<(), ()> = do task::try {
                let mut arc = cell.take();
                let _write_locked = arc.write_locked();
                fail!();
            };

            arc.read_locked();
        }

        #[test] #[should_fail]
        fn test_failed_writes_poison_writes() {
            let mut arc = RWArc::new(());
            let cell = Cell::new(arc.clone());

            let _: Result<(), ()> = do task::try {
                let mut arc = cell.take();
                let _write_locked = arc.write_locked();
                fail!()
            };

            arc.write_locked();
        }

        #[test]
        fn test_failed_reads_do_not_poison_reads() {
            let mut arc = RWArc::new(());
            let cell = Cell::new(arc.clone());

            let _: Result<(), ()> = do task::try {
                let mut arc = cell.take();
                let _read_locked = arc.read_locked();
                fail!()
            };

            arc.read_locked();
        }

        #[test]
        fn test_failed_reads_do_not_poison_writes() {
            let mut arc = RWArc::new(());
            let cell = Cell::new(arc.clone());

            let _: Result<(), ()> = do task::try {
                let mut arc = cell.take();
                let _read_locked = arc.read_locked();
                fail!()
            };

            arc.write_locked();
        }

        #[test]
        fn test_failed_downgraded_reads_do_not_poison_writes() {
            let mut arc = RWArc::new(());
            let cell = Cell::new(arc.clone());

            let _: Result<(), ()> = do task::try {
                let mut arc = cell.take();
                let write_locked = arc.write_locked();
                let _read_locked = write_locked.downgrade();
                fail!()
            };

            arc.write_locked();
        }

        #[test] #[should_fail]
        fn test_arc_unlock_poison() {
            let mut arc = RWArc::new(());
            let arc2 = Cell::new(arc.clone());

            let mut write_locked = arc.write_locked();
            do write_locked.unlock {
                let arc3 = arc2;

                // Poison the arc
                let _: Result<(), ()> = do task::try {
                    let mut arc4 = arc3.take();
                    let _write_locked = arc4.write_locked();
                    fail!()
                };
            }

            // Should fail because of poison
        }
    }
}