// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::sync::{Exclusive, exclusive};
use std::cell::Cell;
use std::comm;

use sync::unlock::Unlock;


type SignalEnd = comm::ChanOne<()>;
type WaitEnd = comm::PortOne<()>;

/// A very basic primitive for synchronization, a queue of wait
/// events.
#[deriving(Clone)]
pub struct WaitQueue {
    // The exclusive lock is needed to prevent a race
    priv head: Exclusive<comm::Port<SignalEnd>>,
    priv tail: comm::SharedChan<SignalEnd>
}

/// A event that can be waited on.
pub struct WaitEvent {
    priv wait_end: WaitEnd
}

impl WaitEvent {
    /// Wait on the associated wait queue.
    pub fn wait(self) {
        self.wait_end.recv()
    }
}


impl WaitQueue {
    /// Create a wait queue.
    #[inline]
    pub fn new() -> WaitQueue {
        let (head, tail) = comm::stream();
        WaitQueue { head: exclusive(head), tail: comm::SharedChan::new(tail) }
    }

    /// Wake up a blocked task. Returns false if there was no blocked
    /// task.
    #[inline]
    pub fn signal(&self) -> bool {
        // Loop popping from the queue, and sending until we
        // succesfully wake up a task, or exhaust the queue.
        loop {
            let maybe_signal_end = unsafe {
                do self.head.with |head| {
                    // The peek is mandatory to make sure recv doesn't block.
                    if head.peek() { Some(head.recv()) } else { None }
                }
            };

            match maybe_signal_end {
                None => return false,
                Some(signal_end) => if comm::try_send_one(signal_end, ()) {
                    return true
                }
            }
        }
    }

    /// Wake up all tasks waiting on the wait queue.
    #[inline]
    pub fn broadcast(&self) -> uint {
        unsafe {
            do self.head.with |head| {
                // The peek is mandatory to make sure recv doesn't block.
                let mut count = 0;
                while head.peek() {
                    if comm::try_send_one(head.recv(), ()) {
                        count += 1;
                    }
                }
                count
            }
        }
    }

    /// Create a wait thunk.
    #[inline]
    pub fn wait_event(&self) -> WaitEvent {
        let (wait_end, signal_end) = comm::oneshot();
        self.tail.send(signal_end);
        WaitEvent { wait_end: wait_end }
    }

    /// Wait on the wait queue.
    #[inline]
    pub fn wait(&self) {
        self.wait_event().wait()
    }

    /// Wait on the wait queue with the unlockable value unlocked.
    #[inline]
    pub fn wait_with<T: Unlock>(&self, lock: &mut T) {
        let waiter = self.wait_event();
        let cell = Cell::new(waiter);
        do lock.unlock {
            cell.take().wait()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::task;
    use std::comm;
    use std::uint;

    use sync::unlock::Unlock;


    #[test]
    fn test_wait_queue_broadcast_wakes_all_waiters() {
        let threads = 10;

        let c = WaitQueue::new();

        let mut ds = ~[];
        for threads.times {
            let (p, d) = comm::stream();
            ds.push(p);
            do task::spawn_with((c.clone(), d)) |(c, d)| {
                c.wait_with(&mut SendOne(d))
            }
        }

        // Wait for all the tasks to wait
        for ds.consume_iter().advance |d| {
            d.recv()
        }

        // Check if all threads were waken up
        assert_eq!(threads, c.broadcast());
    }

    #[test]
    fn test_waits_queue_in_order() {
        let threads = 10;

        let c = WaitQueue::new();
        let (numbers_port, numbers_chan) = comm::stream();
        let numbers_chan = comm::SharedChan::new(numbers_chan);

        for uint::iterate(0, threads) |ii| {
            let (p, d) = comm::stream();
            do task::spawn_with((
                numbers_chan.clone(),
                c.clone(),
                d
            )) |(numbers_chan, c, d)| {
                c.wait_with(&mut SendOne(d));
                numbers_chan.send(ii)
            }

            // Wait for the task to wait
            p.recv()
        }

        // Check if the waits queued in order
        for uint::iterate(threads, 0) |ii| {
            // Each new thread signaled should send the right number
            assert!(c.signal());
            assert_eq!(numbers_port.recv(), ii)
        }
    }

    #[test]
    fn test_signal_wakes_a_single_task() {
        let c = WaitQueue::new();

        let (p, d) = comm::stream();
        do task::spawn_with((c.clone(), d)) |(c, d)| {
            c.wait_with(&mut SendOne(d))
        }

        // Wait for the task to complete
        p.recv();

        assert!(c.signal());
    }

    struct SendOne(comm::Chan<()>);
    impl Unlock for SendOne {
        fn unlock<U>(&mut self, blk: ~once fn() -> U) -> U {
            (**self).send(());
            blk()
        }
    }
}
