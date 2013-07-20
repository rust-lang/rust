// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use std::unstable::sync::{Exclusive, exclusive};

use sync::wait_queue::WaitQueue;

/// A counting semaphore.
#[deriving(Clone)]
pub struct Semaphore {
    priv count: Exclusive<int>,
    priv waiters: WaitQueue
}

impl Semaphore {
    /// Create a counting semaphore that starts with a count of 1.
    #[inline]
    pub fn new() -> Semaphore {
        Semaphore::new_with_count(1)
    }

    /// Create a counting semaphore
    #[inline]
    pub fn new_with_count(count: int) -> Semaphore {
        Semaphore {
            count: exclusive(count),
            waiters: WaitQueue::new()
        }
    }

    #[inline]
    pub fn wait(&self) {
        unsafe {
            let maybe_wait_event = do self.count.with |c| {
                *c -= 1;
                if *c >= 0 { None } else {
                    Some(self.waiters.wait_event())
                }
            };

            match maybe_wait_event {
                // Ordering of waits matters so this must start inside
                // the exclusive lock
                None => {},
                Some(wait_event) => wait_event.wait()
            }
        }
    }

    /// Wake up a blocked task. Returns false if there was no blocked
    /// task.
    #[inline]
    pub fn signal(&self) -> bool {
        unsafe {
            let someone_was_waiting = do self.count.with |c| {
                *c += 1;
                *c > 0
            };

            // Ordering of signals doesn't matter so this can be
            // dragged out of the exclusive lock
            if someone_was_waiting { false } else {
                self.waiters.signal()
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::Cell;
    use std::comm;
    use std::task;

    #[test]
    fn test_sem_acquire_release() {
        let s = Semaphore::new();
        s.wait();
        s.signal();
        s.wait();
    }
    #[test]
    fn test_sem_runtime_friendly_blocking() {
        // Force the runtime to schedule two threads on the same sched_loop.
        // When one blocks, it should schedule the other one.
        do task::spawn_sched(task::ManualThreads(1)) {
            let s = Semaphore::new();
            let s2 = s.clone();
            let (p,c) = comm::stream();
            let child_data = Cell::new((s2, c));
            s.wait();
            let (s2, c) = child_data.take();
            do task::spawn {
                c.send(());
                s2.wait();
                s2.signal();
                c.send(());
            }
            p.recv(); // wait for child to come alive
            for 5.times { task::yield(); } // let the child contend
            s.signal();
            p.recv(); // wait for child to be done
        }
    }
}
