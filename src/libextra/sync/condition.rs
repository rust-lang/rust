// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sync::wait_queue::WaitQueue;
use sync::unlock::Unlock;


/// A wait queue that is strongly associated with a particular lock.
pub struct Condition<'self, T> {
    priv wait_queue: WaitQueue,
    priv lock: &'self mut T,
}

impl <'self, T: Unlock> Condition<'self, T> {
    /// Create a condition
    #[inline]
    pub fn new(lock: &'self mut T) -> Condition<'self, T> {
        Condition { wait_queue: WaitQueue::new(), lock: lock }
    }

    /// Wake up a blocked task. Returns false if there was no blocked
    /// task.
    #[inline]
    pub fn signal(&self) -> bool {
        self.wait_queue.signal()
    }

    /// Wake up all tasks waiting on the condition.
    #[inline]
    pub fn broadcast(&self) -> uint {
        self.wait_queue.broadcast()
    }

    /// Wait on the condition, and unlock the associated lock.
    #[inline]
    pub fn wait(&mut self) {
        self.wait_queue.wait_with(self.lock)
    }
}