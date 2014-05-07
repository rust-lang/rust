// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Stream channels
///
/// This is the flavor of channels which are optimized for one sender and one
/// receiver. The sender will be upgraded to a shared channel if the channel is
/// cloned.
///
/// High level implementation details can be found in the comment of the parent
/// module.

use cmp;
use comm::Receiver;
use int;
use iter::Iterator;
use kinds::Send;
use ops::Drop;
use option::{Some, None};
use owned::Box;
use result::{Ok, Err, Result};
use rt::local::Local;
use rt::task::{Task, BlockedTask};
use rt::thread::Thread;
use spsc = sync::spsc_queue;
use sync::atomics;

static DISCONNECTED: int = int::MIN;
#[cfg(test)]
static MAX_STEALS: int = 5;
#[cfg(not(test))]
static MAX_STEALS: int = 1 << 20;

pub struct Packet<T> {
    queue: spsc::Queue<Message<T>>, // internal queue for all message

    cnt: atomics::AtomicInt, // How many items are on this channel
    steals: int, // How many times has a port received without blocking?
    to_wake: atomics::AtomicUint, // Task to wake up

    port_dropped: atomics::AtomicBool, // flag if the channel has been destroyed.
}

pub enum Failure<T> {
    Empty,
    Disconnected,
    Upgraded(Receiver<T>),
}

pub enum UpgradeResult {
    UpSuccess,
    UpDisconnected,
    UpWoke(BlockedTask),
}

pub enum SelectionResult<T> {
    SelSuccess,
    SelCanceled(BlockedTask),
    SelUpgraded(BlockedTask, Receiver<T>),
}

// Any message could contain an "upgrade request" to a new shared port, so the
// internal queue it's a queue of T, but rather Message<T>
enum Message<T> {
    Data(T),
    GoUp(Receiver<T>),
}

impl<T: Send> Packet<T> {
    pub fn new() -> Packet<T> {
        Packet {
            queue: spsc::Queue::new(128),

            cnt: atomics::AtomicInt::new(0),
            steals: 0,
            to_wake: atomics::AtomicUint::new(0),

            port_dropped: atomics::AtomicBool::new(false),
        }
    }


    pub fn send(&mut self, t: T) -> Result<(), T> {
        // If the other port has deterministically gone away, then definitely
        // must return the data back up the stack. Otherwise, the data is
        // considered as being sent.
        if self.port_dropped.load(atomics::SeqCst) { return Err(t) }

        match self.do_send(Data(t)) {
            UpSuccess | UpDisconnected => {},
            UpWoke(task) => { task.wake().map(|t| t.reawaken()); }
        }
        Ok(())
    }
    pub fn upgrade(&mut self, up: Receiver<T>) -> UpgradeResult {
        // If the port has gone away, then there's no need to proceed any
        // further.
        if self.port_dropped.load(atomics::SeqCst) { return UpDisconnected }

        self.do_send(GoUp(up))
    }

    fn do_send(&mut self, t: Message<T>) -> UpgradeResult {
        self.queue.push(t);
        match self.cnt.fetch_add(1, atomics::SeqCst) {
            // As described in the mod's doc comment, -1 == wakeup
            -1 => UpWoke(self.take_to_wake()),
            // As as described before, SPSC queues must be >= -2
            -2 => UpSuccess,

            // Be sure to preserve the disconnected state, and the return value
            // in this case is going to be whether our data was received or not.
            // This manifests itself on whether we have an empty queue or not.
            //
            // Primarily, are required to drain the queue here because the port
            // will never remove this data. We can only have at most one item to
            // drain (the port drains the rest).
            DISCONNECTED => {
                self.cnt.store(DISCONNECTED, atomics::SeqCst);
                let first = self.queue.pop();
                let second = self.queue.pop();
                assert!(second.is_none());

                match first {
                    Some(..) => UpSuccess,  // we failed to send the data
                    None => UpDisconnected, // we successfully sent data
                }
            }

            // Otherwise we just sent some data on a non-waiting queue, so just
            // make sure the world is sane and carry on!
            n => { assert!(n >= 0); UpSuccess }
        }
    }

    // Consumes ownership of the 'to_wake' field.
    fn take_to_wake(&mut self) -> BlockedTask {
        let task = self.to_wake.load(atomics::SeqCst);
        self.to_wake.store(0, atomics::SeqCst);
        assert!(task != 0);
        unsafe { BlockedTask::cast_from_uint(task) }
    }

    // Decrements the count on the channel for a sleeper, returning the sleeper
    // back if it shouldn't sleep. Note that this is the location where we take
    // steals into account.
    fn decrement(&mut self, task: BlockedTask) -> Result<(), BlockedTask> {
        assert_eq!(self.to_wake.load(atomics::SeqCst), 0);
        let n = unsafe { task.cast_to_uint() };
        self.to_wake.store(n, atomics::SeqCst);

        let steals = self.steals;
        self.steals = 0;

        match self.cnt.fetch_sub(1 + steals, atomics::SeqCst) {
            DISCONNECTED => { self.cnt.store(DISCONNECTED, atomics::SeqCst); }
            // If we factor in our steals and notice that the channel has no
            // data, we successfully sleep
            n => {
                assert!(n >= 0);
                if n - steals <= 0 { return Ok(()) }
            }
        }

        self.to_wake.store(0, atomics::SeqCst);
        Err(unsafe { BlockedTask::cast_from_uint(n) })
    }

    pub fn recv(&mut self) -> Result<T, Failure<T>> {
        // Optimistic preflight check (scheduling is expensive).
        match self.try_recv() {
            Err(Empty) => {}
            data => return data,
        }

        // Welp, our channel has no data. Deschedule the current task and
        // initiate the blocking protocol.
        let task: Box<Task> = Local::take();
        task.deschedule(1, |task| {
            self.decrement(task)
        });

        match self.try_recv() {
            // Messages which actually popped from the queue shouldn't count as
            // a steal, so offset the decrement here (we already have our
            // "steal" factored into the channel count above).
            data @ Ok(..) |
            data @ Err(Upgraded(..)) => {
                self.steals -= 1;
                data
            }

            data => data,
        }
    }

    pub fn try_recv(&mut self) -> Result<T, Failure<T>> {
        match self.queue.pop() {
            // If we stole some data, record to that effect (this will be
            // factored into cnt later on).
            //
            // Note that we don't allow steals to grow without bound in order to
            // prevent eventual overflow of either steals or cnt as an overflow
            // would have catastrophic results. Sometimes, steals > cnt, but
            // other times cnt > steals, so we don't know the relation between
            // steals and cnt. This code path is executed only rarely, so we do
            // a pretty slow operation, of swapping 0 into cnt, taking steals
            // down as much as possible (without going negative), and then
            // adding back in whatever we couldn't factor into steals.
            Some(data) => {
                if self.steals > MAX_STEALS {
                    match self.cnt.swap(0, atomics::SeqCst) {
                        DISCONNECTED => {
                            self.cnt.store(DISCONNECTED, atomics::SeqCst);
                        }
                        n => {
                            let m = cmp::min(n, self.steals);
                            self.steals -= m;
                            self.bump(n - m);
                        }
                    }
                    assert!(self.steals >= 0);
                }
                self.steals += 1;
                match data {
                    Data(t) => Ok(t),
                    GoUp(up) => Err(Upgraded(up)),
                }
            }

            None => {
                match self.cnt.load(atomics::SeqCst) {
                    n if n != DISCONNECTED => Err(Empty),

                    // This is a little bit of a tricky case. We failed to pop
                    // data above, and then we have viewed that the channel is
                    // disconnected. In this window more data could have been
                    // sent on the channel. It doesn't really make sense to
                    // return that the channel is disconnected when there's
                    // actually data on it, so be extra sure there's no data by
                    // popping one more time.
                    //
                    // We can ignore steals because the other end is
                    // disconnected and we'll never need to really factor in our
                    // steals again.
                    _ => {
                        match self.queue.pop() {
                            Some(Data(t)) => Ok(t),
                            Some(GoUp(up)) => Err(Upgraded(up)),
                            None => Err(Disconnected),
                        }
                    }
                }
            }
        }
    }

    pub fn drop_chan(&mut self) {
        // Dropping a channel is pretty simple, we just flag it as disconnected
        // and then wakeup a blocker if there is one.
        match self.cnt.swap(DISCONNECTED, atomics::SeqCst) {
            -1 => { self.take_to_wake().wake().map(|t| t.reawaken()); }
            DISCONNECTED => {}
            n => { assert!(n >= 0); }
        }
    }

    pub fn drop_port(&mut self) {
        // Dropping a port seems like a fairly trivial thing. In theory all we
        // need to do is flag that we're disconnected and then everything else
        // can take over (we don't have anyone to wake up).
        //
        // The catch for Ports is that we want to drop the entire contents of
        // the queue. There are multiple reasons for having this property, the
        // largest of which is that if another chan is waiting in this channel
        // (but not received yet), then waiting on that port will cause a
        // deadlock.
        //
        // So if we accept that we must now destroy the entire contents of the
        // queue, this code may make a bit more sense. The tricky part is that
        // we can't let any in-flight sends go un-dropped, we have to make sure
        // *everything* is dropped and nothing new will come onto the channel.

        // The first thing we do is set a flag saying that we're done for. All
        // sends are gated on this flag, so we're immediately guaranteed that
        // there are a bounded number of active sends that we'll have to deal
        // with.
        self.port_dropped.store(true, atomics::SeqCst);

        // Now that we're guaranteed to deal with a bounded number of senders,
        // we need to drain the queue. This draining process happens atomically
        // with respect to the "count" of the channel. If the count is nonzero
        // (with steals taken into account), then there must be data on the
        // channel. In this case we drain everything and then try again. We will
        // continue to fail while active senders send data while we're dropping
        // data, but eventually we're guaranteed to break out of this loop
        // (because there is a bounded number of senders).
        let mut steals = self.steals;
        while {
            let cnt = self.cnt.compare_and_swap(
                            steals, DISCONNECTED, atomics::SeqCst);
            cnt != DISCONNECTED && cnt != steals
        } {
            loop {
                match self.queue.pop() {
                    Some(..) => { steals += 1; }
                    None => break
                }
            }
        }

        // At this point in time, we have gated all future senders from sending,
        // and we have flagged the channel as being disconnected. The senders
        // still have some responsibility, however, because some sends may not
        // complete until after we flag the disconnection. There are more
        // details in the sending methods that see DISCONNECTED
    }

    ////////////////////////////////////////////////////////////////////////////
    // select implementation
    ////////////////////////////////////////////////////////////////////////////

    // Tests to see whether this port can receive without blocking. If Ok is
    // returned, then that's the answer. If Err is returned, then the returned
    // port needs to be queried instead (an upgrade happened)
    pub fn can_recv(&mut self) -> Result<bool, Receiver<T>> {
        // We peek at the queue to see if there's anything on it, and we use
        // this return value to determine if we should pop from the queue and
        // upgrade this channel immediately. If it looks like we've got an
        // upgrade pending, then go through the whole recv rigamarole to update
        // the internal state.
        match self.queue.peek() {
            Some(&GoUp(..)) => {
                match self.recv() {
                    Err(Upgraded(port)) => Err(port),
                    _ => unreachable!(),
                }
            }
            Some(..) => Ok(true),
            None => Ok(false)
        }
    }

    // increment the count on the channel (used for selection)
    fn bump(&mut self, amt: int) -> int {
        match self.cnt.fetch_add(amt, atomics::SeqCst) {
            DISCONNECTED => {
                self.cnt.store(DISCONNECTED, atomics::SeqCst);
                DISCONNECTED
            }
            n => n
        }
    }

    // Attempts to start selecting on this port. Like a oneshot, this can fail
    // immediately because of an upgrade.
    pub fn start_selection(&mut self, task: BlockedTask) -> SelectionResult<T> {
        match self.decrement(task) {
            Ok(()) => SelSuccess,
            Err(task) => {
                let ret = match self.queue.peek() {
                    Some(&GoUp(..)) => {
                        match self.queue.pop() {
                            Some(GoUp(port)) => SelUpgraded(task, port),
                            _ => unreachable!(),
                        }
                    }
                    Some(..) => SelCanceled(task),
                    None => SelCanceled(task),
                };
                // Undo our decrement above, and we should be guaranteed that the
                // previous value is positive because we're not going to sleep
                let prev = self.bump(1);
                assert!(prev == DISCONNECTED || prev >= 0);
                return ret;
            }
        }
    }

    // Removes a previous task from being blocked in this port
    pub fn abort_selection(&mut self,
                           was_upgrade: bool) -> Result<bool, Receiver<T>> {
        // If we're aborting selection after upgrading from a oneshot, then
        // we're guarantee that no one is waiting. The only way that we could
        // have seen the upgrade is if data was actually sent on the channel
        // half again. For us, this means that there is guaranteed to be data on
        // this channel. Furthermore, we're guaranteed that there was no
        // start_selection previously, so there's no need to modify `self.cnt`
        // at all.
        //
        // Hence, because of these invariants, we immediately return `Ok(true)`.
        // Note that the data may not actually be sent on the channel just yet.
        // The other end could have flagged the upgrade but not sent data to
        // this end. This is fine because we know it's a small bounded windows
        // of time until the data is actually sent.
        if was_upgrade {
            assert_eq!(self.steals, 0);
            assert_eq!(self.to_wake.load(atomics::SeqCst), 0);
            return Ok(true)
        }

        // We want to make sure that the count on the channel goes non-negative,
        // and in the stream case we can have at most one steal, so just assume
        // that we had one steal.
        let steals = 1;
        let prev = self.bump(steals + 1);

        // If we were previously disconnected, then we know for sure that there
        // is no task in to_wake, so just keep going
        let has_data = if prev == DISCONNECTED {
            assert_eq!(self.to_wake.load(atomics::SeqCst), 0);
            true // there is data, that data is that we're disconnected
        } else {
            let cur = prev + steals + 1;
            assert!(cur >= 0);

            // If the previous count was negative, then we just made things go
            // positive, hence we passed the -1 boundary and we're responsible
            // for removing the to_wake() field and trashing it.
            //
            // If the previous count was positive then we're in a tougher
            // situation. A possible race is that a sender just incremented
            // through -1 (meaning it's going to try to wake a task up), but it
            // hasn't yet read the to_wake. In order to prevent a future recv()
            // from waking up too early (this sender picking up the plastered
            // over to_wake), we spin loop here waiting for to_wake to be 0.
            // Note that this entire select() implementation needs an overhaul,
            // and this is *not* the worst part of it, so this is not done as a
            // final solution but rather out of necessity for now to get
            // something working.
            if prev < 0 {
                self.take_to_wake().trash();
            } else {
                while self.to_wake.load(atomics::SeqCst) != 0 {
                    Thread::yield_now();
                }
            }
            assert_eq!(self.steals, 0);
            self.steals = steals;

            // if we were previously positive, then there's surely data to
            // receive
            prev >= 0
        };

        // Now that we've determined that this queue "has data", we peek at the
        // queue to see if the data is an upgrade or not. If it's an upgrade,
        // then we need to destroy this port and abort selection on the
        // upgraded port.
        if has_data {
            match self.queue.peek() {
                Some(&GoUp(..)) => {
                    match self.queue.pop() {
                        Some(GoUp(port)) => Err(port),
                        _ => unreachable!(),
                    }
                }
                _ => Ok(true),
            }
        } else {
            Ok(false)
        }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Packet<T> {
    fn drop(&mut self) {
        // Note that this load is not only an assert for correctness about
        // disconnection, but also a proper fence before the read of
        // `to_wake`, so this assert cannot be removed with also removing
        // the `to_wake` assert.
        assert_eq!(self.cnt.load(atomics::SeqCst), DISCONNECTED);
        assert_eq!(self.to_wake.load(atomics::SeqCst), 0);
    }
}
