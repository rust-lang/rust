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
use std::task;

use sync::mutex::{Mutex, Lock};
use sync::unlock::Unlock;


struct MutexArcInner<T> { mutex: Mutex, failed: bool, data: T }

impl <T> MutexArcInner<T> {
    fn assert_not_failed(&self) {
        if self.failed {
            fail!("Poisoned MutexArc - another task failed inside!")
        }
    }
}

/// An atomically reference counted variable with mutable data
/// protected by a blocking mutex.
pub struct MutexArc<T> { priv contents: UnsafeAtomicRcBox<MutexArcInner<T>> }

impl <T: Send> Clone for MutexArc<T> {
    #[inline]
    fn clone(&self) -> MutexArc<T> {
        MutexArc { contents: self.contents.clone() }
    }
}

impl <T: Send> MutexArc<T> {
    /// Create a mutex-protected ARC with the supplied data.
    #[inline]
    pub fn new(user_data: T) -> MutexArc<T> {
        let data = MutexArcInner {
            mutex: Mutex::new(),
            failed: false,
            data: user_data
        };
        MutexArc { contents: UnsafeAtomicRcBox::new(data) }
    }

    /**
     * Access the underlying mutable data with mutual exclusion from
     * other tasks.
     *
     * The mutex will be locked until the access cookie is released;
     * all other tasks wishing to access the data will block until the
     * cookie is released.
     *
     * # Safety notes
     * The reason this function is 'unsafe' is because it is possible
     * to construct a circular reference among multiple ARCs by
     * mutating the underlying data. This creates potential for
     * deadlock, but worse, this will guarantee a memory leak of all
     * involved ARCs. Using mutex ARCs inside of other ARCs is safe in
     * the absence of circular references.
     *
     * If you wish to nest mutex_arcs, one strategy for ensuring
     * safety at runtime is to add a "nesting level counter" inside
     * the stored data, and when traversing the arcs, assert that they
     * monotonically decrease.
     *
     * # Failure
     * Failing while inside the ARC will unlock the ARC while
     * unwinding, so that other tasks won't block forever. It will
     * also poison the ARC: any tasks that subsequently try to access
     * it (including those already blocked on the mutex) will also
     * fail immediately.
     */
    #[inline]
    pub unsafe fn locked<'r>(&'r self) -> Locked<'r, T> {
        let state = &*self.contents.get();

        let lock = state.mutex.lock();
        state.assert_not_failed();
        Locked { lock: lock, mutex_arc: self.clone() }
    }

    /**
     * A convenience function to wrap the more complicated (but more
     * powerful locked method.) Obtains a lock, accesses the value,
     * and then invokes the blk argument.
     */
    #[inline]
    pub unsafe fn get<U>(&self, blk: &fn(&mut T) -> U) -> U {
        let mut locked = self.locked();
        blk(locked.get())
    }
}

/// A value that guarantees exclusive access to a MutexArc until
/// destroyed
pub struct Locked<'self, T> { priv lock: Lock<'self>, priv mutex_arc: MutexArc<T> }

#[unsafe_destructor]
impl <'self, T: Send> Drop for Locked<'self, T> {
    // Don't inline this due to issue #7793
    pub fn drop(&self) {
        unsafe {
            /* There may be an assertion similar to
            assert!(!*self.failed)` that can be made here. This
            assertion might be false in case of cond.wait() */

            if task::failing() {
                let state = &mut *self.mutex_arc.contents.get();
                state.failed = true
            }
        }
    }
}

impl <'self, T: Send> Locked<'self, T> {
    /// Access the underlying locked data.
    #[inline]
    pub fn get(&'self mut self) -> &'self mut T {
        unsafe {
            let value = &'self mut *self.mutex_arc.contents.get();
            &'self mut value.data
        }
    }
}

impl <'self, T: Send> Unlock for Locked<'self, T> {
    #[inline]
    pub fn unlock<V>(&mut self, blk: ~once fn() -> V) -> V {
        let result = self.lock.unlock(blk);
        unsafe {
            let state = &*self.mutex_arc.contents.get();
            state.assert_not_failed()
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use std::task;
    use std::cell::Cell;


    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_failed_locks_poison_locks() {
        unsafe {
            let arc = MutexArc::new(());
            let arc2 = arc.clone();

            let _: Result<(), ()> = do task::try {
                let _locked = arc2.locked();
                fail!();
            };

            let _locked = arc.locked();
        }
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_arc_unlock_poison() {
        unsafe {
            let arc = MutexArc::new(());
            let arc2 = Cell::new(arc.clone());

            let mut locked = arc.locked();
            do locked.unlock {
                let arc3 = arc2;
                // Poison the arc
                let _: Result<(), ()> = do task::try {
                    let arc4 = arc3.take();
                    let _locked = arc4.locked();
                    fail!()
                };
            }

            // Should fail because of poison
        }
    }
}