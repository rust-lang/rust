// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Shared channels
///
/// This is the flavor of channels which are not necessarily optimized for any
/// particular use case, but are the most general in how they are used. Shared
/// channels are cloneable allowing for multiple senders.
///
/// High level implementation details can be found in the comment of the parent
/// module. You'll also note that the implementation of the shared and stream
/// channels are quite similar, and this is no coincidence!

use cmp;
use int;
use iter::Iterator;
use kinds::Send;
use ops::Drop;
use option::{Some, None, Option};
use result::{Ok, Err, Result};
use rt::local::Local;
use rt::task::{Task, BlockedTask};
use rt::thread::Thread;
use sync::atomics;
use unstable::mutex::NativeMutex;
use slice::OwnedVector;

use mpsc = sync::mpsc_queue;

static DISCONNECTED: int = int::MIN;
static FUDGE: int = 1024;
#[cfg(test)]
static MAX_STEALS: int = 5;
#[cfg(not(test))]
static MAX_STEALS: int = 1 << 20;

pub struct Packet<T> {
    queue: mpsc::Queue<T>,
    cnt: atomics::AtomicInt, // How many items are on this channel
    steals: int, // How many times has a port received without blocking?
    to_wake: atomics::AtomicUint, // Task to wake up

    // The number of channels which are currently using this packet.
    channels: atomics::AtomicInt,

    // See the discussion in Port::drop and the channel send methods for what
    // these are used for
    port_dropped: atomics::AtomicBool,
    sender_drain: atomics::AtomicInt,

    // this lock protects various portions of this implementation during
    // select()
    select_lock: NativeMutex,
}

pub enum Failure {
    Empty,
    Disconnected,
}

impl<T: Send> Packet<T> {
    // Creation of a packet *must* be followed by a call to inherit_blocker
    pub fn new() -> Packet<T> {
        let p = Packet {
            queue: mpsc::Queue::new(),
            cnt: atomics::AtomicInt::new(0),
            steals: 0,
            to_wake: atomics::AtomicUint::new(0),
            channels: atomics::AtomicInt::new(2),
            port_dropped: atomics::AtomicBool::new(false),
            sender_drain: atomics::AtomicInt::new(0),
            select_lock: unsafe { NativeMutex::new() },
        };
        // see comments in inherit_blocker about why we grab this lock
        unsafe { p.select_lock.lock_noguard() }
        return p;
    }

    // This function is used at the creation of a shared packet to inherit a
    // previously blocked task. This is done to prevent spurious wakeups of
    // tasks in select().
    //
    // This can only be called at channel-creation time
    pub fn inherit_blocker(&mut self, task: Option<BlockedTask>) {
        match task {
            Some(task) => {
                assert_eq!(self.cnt.load(atomics::SeqCst), 0);
                assert_eq!(self.to_wake.load(atomics::SeqCst), 0);
                self.to_wake.store(unsafe { task.cast_to_uint() },
                                   atomics::SeqCst);
                self.cnt.store(-1, atomics::SeqCst);

                // This store is a little sketchy. What's happening here is
                // that we're transferring a blocker from a oneshot or stream
                // channel to this shared channel. In doing so, we never
                // spuriously wake them up and rather only wake them up at the
                // appropriate time. This implementation of shared channels
                // assumes that any blocking recv() will undo the increment of
                // steals performed in try_recv() once the recv is complete.
                // This thread that we're inheriting, however, is not in the
                // middle of recv. Hence, the first time we wake them up,
                // they're going to wake up from their old port, move on to the
                // upgraded port, and then call the block recv() function.
                //
                // When calling this function, they'll find there's data
                // immediately available, counting it as a steal. This in fact
                // wasn't a steal because we appropriately blocked them waiting
                // for data.
                //
                // To offset this bad increment, we initially set the steal
                // count to -1. You'll find some special code in
                // abort_selection() as well to ensure that this -1 steal count
                // doesn't escape too far.
                self.steals = -1;
            }
            None => {}
        }

        // When the shared packet is constructed, we grabbed this lock. The
        // purpose of this lock is to ensure that abort_selection() doesn't
        // interfere with this method. After we unlock this lock, we're
        // signifying that we're done modifying self.cnt and self.to_wake and
        // the port is ready for the world to continue using it.
        unsafe { self.select_lock.unlock_noguard() }
    }

    pub fn send(&mut self, t: T) -> bool {
        // See Port::drop for what's going on
        if self.port_dropped.load(atomics::SeqCst) { return false }

        // Note that the multiple sender case is a little tricker
        // semantically than the single sender case. The logic for
        // incrementing is "add and if disconnected store disconnected".
        // This could end up leading some senders to believe that there
        // wasn't a disconnect if in fact there was a disconnect. This means
        // that while one thread is attempting to re-store the disconnected
        // states, other threads could walk through merrily incrementing
        // this very-negative disconnected count. To prevent senders from
        // spuriously attempting to send when the channels is actually
        // disconnected, the count has a ranged check here.
        //
        // This is also done for another reason. Remember that the return
        // value of this function is:
        //
        //  `true` == the data *may* be received, this essentially has no
        //            meaning
        //  `false` == the data will *never* be received, this has a lot of
        //             meaning
        //
        // In the SPSC case, we have a check of 'queue.is_empty()' to see
        // whether the data was actually received, but this same condition
        // means nothing in a multi-producer context. As a result, this
        // preflight check serves as the definitive "this will never be
        // received". Once we get beyond this check, we have permanently
        // entered the realm of "this may be received"
        if self.cnt.load(atomics::SeqCst) < DISCONNECTED + FUDGE {
            return false
        }

        self.queue.push(t);
        match self.cnt.fetch_add(1, atomics::SeqCst) {
            -1 => {
                self.take_to_wake().wake().map(|t| t.reawaken());
            }

            // In this case, we have possibly failed to send our data, and
            // we need to consider re-popping the data in order to fully
            // destroy it. We must arbitrate among the multiple senders,
            // however, because the queues that we're using are
            // single-consumer queues. In order to do this, all exiting
            // pushers will use an atomic count in order to count those
            // flowing through. Pushers who see 0 are required to drain as
            // much as possible, and then can only exit when they are the
            // only pusher (otherwise they must try again).
            n if n < DISCONNECTED + FUDGE => {
                // see the comment in 'try' for a shared channel for why this
                // window of "not disconnected" is ok.
                self.cnt.store(DISCONNECTED, atomics::SeqCst);

                if self.sender_drain.fetch_add(1, atomics::SeqCst) == 0 {
                    loop {
                        // drain the queue, for info on the thread yield see the
                        // discussion in try_recv
                        loop {
                            match self.queue.pop() {
                                mpsc::Data(..) => {}
                                mpsc::Empty => break,
                                mpsc::Inconsistent => Thread::yield_now(),
                            }
                        }
                        // maybe we're done, if we're not the last ones
                        // here, then we need to go try again.
                        if self.sender_drain.fetch_sub(1, atomics::SeqCst) == 1 {
                            break
                        }
                    }

                    // At this point, there may still be data on the queue,
                    // but only if the count hasn't been incremented and
                    // some other sender hasn't finished pushing data just
                    // yet. That sender in question will drain its own data.
                }
            }

            // Can't make any assumptions about this case like in the SPSC case.
            _ => {}
        }

        true
    }

    pub fn recv(&mut self) -> Result<T, Failure> {
        // This code is essentially the exact same as that found in the stream
        // case (see stream.rs)
        match self.try_recv() {
            Err(Empty) => {}
            data => return data,
        }

        let task: ~Task = Local::take();
        task.deschedule(1, |task| {
            self.decrement(task)
        });

        match self.try_recv() {
            data @ Ok(..) => { self.steals -= 1; data }
            data => data,
        }
    }

    // Essentially the exact same thing as the stream decrement function.
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

    pub fn try_recv(&mut self) -> Result<T, Failure> {
        let ret = match self.queue.pop() {
            mpsc::Data(t) => Some(t),
            mpsc::Empty => None,

            // This is a bit of an interesting case. The channel is
            // reported as having data available, but our pop() has
            // failed due to the queue being in an inconsistent state.
            // This means that there is some pusher somewhere which has
            // yet to complete, but we are guaranteed that a pop will
            // eventually succeed. In this case, we spin in a yield loop
            // because the remote sender should finish their enqueue
            // operation "very quickly".
            //
            // Note that this yield loop does *not* attempt to do a green
            // yield (regardless of the context), but *always* performs an
            // OS-thread yield. The reasoning for this is that the pusher in
            // question which is causing the inconsistent state is
            // guaranteed to *not* be a blocked task (green tasks can't get
            // pre-empted), so it must be on a different OS thread. Also,
            // `try_recv` is normally a "guaranteed no rescheduling" context
            // in a green-thread situation. By yielding control of the
            // thread, we will hopefully allow time for the remote task on
            // the other OS thread to make progress.
            //
            // Avoiding this yield loop would require a different queue
            // abstraction which provides the guarantee that after M
            // pushes have succeeded, at least M pops will succeed. The
            // current queues guarantee that if there are N active
            // pushes, you can pop N times once all N have finished.
            mpsc::Inconsistent => {
                let data;
                loop {
                    Thread::yield_now();
                    match self.queue.pop() {
                        mpsc::Data(t) => { data = t; break }
                        mpsc::Empty => fail!("inconsistent => empty"),
                        mpsc::Inconsistent => {}
                    }
                }
                Some(data)
            }
        };
        match ret {
            // See the discussion in the stream implementation for why we
            // might decrement steals.
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
                Ok(data)
            }

            // See the discussion in the stream implementation for why we try
            // again.
            None => {
                match self.cnt.load(atomics::SeqCst) {
                    n if n != DISCONNECTED => Err(Empty),
                    _ => {
                        match self.queue.pop() {
                            mpsc::Data(t) => Ok(t),
                            mpsc::Empty => Err(Disconnected),
                            // with no senders, an inconsistency is impossible.
                            mpsc::Inconsistent => unreachable!(),
                        }
                    }
                }
            }
        }
    }

    // Prepares this shared packet for a channel clone, essentially just bumping
    // a refcount.
    pub fn clone_chan(&mut self) {
        self.channels.fetch_add(1, atomics::SeqCst);
    }

    // Decrement the reference count on a channel. This is called whenever a
    // Chan is dropped and may end up waking up a receiver. It's the receiver's
    // responsibility on the other end to figure out that we've disconnected.
    pub fn drop_chan(&mut self) {
        match self.channels.fetch_sub(1, atomics::SeqCst) {
            1 => {}
            n if n > 1 => return,
            n => fail!("bad number of channels left {}", n),
        }

        match self.cnt.swap(DISCONNECTED, atomics::SeqCst) {
            -1 => { self.take_to_wake().wake().map(|t| t.reawaken()); }
            DISCONNECTED => {}
            n => { assert!(n >= 0); }
        }
    }

    // See the long discussion inside of stream.rs for why the queue is drained,
    // and why it is done in this fashion.
    pub fn drop_port(&mut self) {
        self.port_dropped.store(true, atomics::SeqCst);
        let mut steals = self.steals;
        while {
            let cnt = self.cnt.compare_and_swap(
                            steals, DISCONNECTED, atomics::SeqCst);
            cnt != DISCONNECTED && cnt != steals
        } {
            // See the discussion in 'try_recv' for why we yield
            // control of this thread.
            loop {
                match self.queue.pop() {
                    mpsc::Data(..) => { steals += 1; }
                    mpsc::Empty | mpsc::Inconsistent => break,
                }
            }
        }
    }

    // Consumes ownership of the 'to_wake' field.
    fn take_to_wake(&mut self) -> BlockedTask {
        let task = self.to_wake.load(atomics::SeqCst);
        self.to_wake.store(0, atomics::SeqCst);
        assert!(task != 0);
        unsafe { BlockedTask::cast_from_uint(task) }
    }

    ////////////////////////////////////////////////////////////////////////////
    // select implementation
    ////////////////////////////////////////////////////////////////////////////

    // Helper function for select, tests whether this port can receive without
    // blocking (obviously not an atomic decision).
    //
    // This is different than the stream version because there's no need to peek
    // at the queue, we can just look at the local count.
    pub fn can_recv(&mut self) -> bool {
        let cnt = self.cnt.load(atomics::SeqCst);
        cnt == DISCONNECTED || cnt - self.steals > 0
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

    // Inserts the blocked task for selection on this port, returning it back if
    // the port already has data on it.
    //
    // The code here is the same as in stream.rs, except that it doesn't need to
    // peek at the channel to see if an upgrade is pending.
    pub fn start_selection(&mut self,
                           task: BlockedTask) -> Result<(), BlockedTask> {
        match self.decrement(task) {
            Ok(()) => Ok(()),
            Err(task) => {
                let prev = self.bump(1);
                assert!(prev == DISCONNECTED || prev >= 0);
                return Err(task);
            }
        }
    }

    // Cancels a previous task waiting on this port, returning whether there's
    // data on the port.
    //
    // This is similar to the stream implementation (hence fewer comments), but
    // uses a different value for the "steals" variable.
    pub fn abort_selection(&mut self, _was_upgrade: bool) -> bool {
        // Before we do anything else, we bounce on this lock. The reason for
        // doing this is to ensure that any upgrade-in-progress is gone and
        // done with. Without this bounce, we can race with inherit_blocker
        // about looking at and dealing with to_wake. Once we have acquired the
        // lock, we are guaranteed that inherit_blocker is done.
        unsafe {
            let _guard = self.select_lock.lock();
        }

        // Like the stream implementation, we want to make sure that the count
        // on the channel goes non-negative. We don't know how negative the
        // stream currently is, so instead of using a steal value of 1, we load
        // the channel count and figure out what we should do to make it
        // positive.
        let steals = {
            let cnt = self.cnt.load(atomics::SeqCst);
            if cnt < 0 && cnt != DISCONNECTED {-cnt} else {0}
        };
        let prev = self.bump(steals + 1);

        if prev == DISCONNECTED {
            assert_eq!(self.to_wake.load(atomics::SeqCst), 0);
            true
        } else {
            let cur = prev + steals + 1;
            assert!(cur >= 0);
            if prev < 0 {
                self.take_to_wake().trash();
            } else {
                while self.to_wake.load(atomics::SeqCst) != 0 {
                    Thread::yield_now();
                }
            }
            // if the number of steals is -1, it was the pre-emptive -1 steal
            // count from when we inherited a blocker. This is fine because
            // we're just going to overwrite it with a real value.
            assert!(self.steals == 0 || self.steals == -1);
            self.steals = steals;
            prev >= 0
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
        assert_eq!(self.channels.load(atomics::SeqCst), 0);
    }
}
