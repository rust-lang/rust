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

pub use self::Failure::*;

use core::intrinsics::abort;
use core::isize;
use core::usize;

use sync::atomic::{AtomicUsize, AtomicIsize, Ordering};
use sync::mpsc::blocking::{self, SignalToken};
use sync::mpsc::mpsc_queue as mpsc;
use sync::mpsc::select::StartResult::*;
use sync::mpsc::select::StartResult;
use sync::{Mutex, MutexGuard};
use thread;
use time::Instant;

const DISCONNECTED: usize = usize::MAX;
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

struct CellDisconnected;

struct SignalTokenCell {
    // Atomic holder of 0, DISCONNECTED or SignalToken
    token: AtomicUsize,
}

impl Drop for SignalTokenCell {
    fn drop(&mut self) {
        self.take_token();
    }
}

impl SignalTokenCell {
    fn new() -> SignalTokenCell {
        SignalTokenCell {
            token: AtomicUsize::new(0)
        }
    }

    fn load_is_disconnected(&self) -> bool {
        self.token.load(Ordering::Relaxed) == DISCONNECTED
    }

    /// Do not overwrite DISCONNECTED or another token
    fn store_if_empty(&self, token: SignalToken) {
        let ptr = unsafe { token.cast_to_usize() };
        if self.token.compare_and_swap(0, ptr, Ordering::SeqCst) != 0 {
            unsafe { SignalToken::cast_from_usize(ptr); }
        }
    }

    /// Store token unless it is disconnected overwriting another token if any
    fn store_unless_disconnected(&self, token: SignalToken) -> Result<(), CellDisconnected> {
        let ptr = unsafe { token.cast_to_usize() };
        let mut curr = self.token.load(Ordering::Relaxed);
        loop {
            if curr == DISCONNECTED {
                unsafe { SignalToken::cast_from_usize(ptr); }
                return Err(CellDisconnected);
            }
            let prev = self.token.compare_and_swap(curr, ptr, Ordering::SeqCst);
            if prev == curr {
                if prev != 0 {
                    unsafe { SignalToken::cast_from_usize(prev); }
                }
                return Ok(());
            }
            curr = prev;
        }
    }

    fn store_disconnected(&self) -> Option<SignalToken> {
        let ptr = self.token.swap(DISCONNECTED, Ordering::SeqCst);
        if ptr != 0 && ptr != DISCONNECTED {
            Some(unsafe { SignalToken::cast_from_usize(ptr) })
        } else {
            None
        }
    }

    fn take_token(&self) -> Option<SignalToken> {
        let mut curr = self.token.load(Ordering::SeqCst);
        loop {
            if curr == 0 || curr == DISCONNECTED {
                return None;
            }

            let prev = self.token.compare_and_swap(curr, 0, Ordering::SeqCst);
            if prev == curr {
                return Some(unsafe { SignalToken::cast_from_usize(curr) })
            }

            curr = prev;
        }
    }
}


pub struct Packet<T> {
    queue: mpsc::Queue<T>,
    to_wake: SignalTokenCell, // SignalToken for wake up

    // The number of channels which are currently using this packet.
    channels: AtomicUsize,

    // See the discussion in Port::drop and the channel send methods for what
    // these are used for
    sender_drain: AtomicIsize,

    // this lock protects various portions of this implementation during
    // select()
    select_lock: Mutex<()>,
}

pub enum Failure {
    Empty,
    Disconnected,
}

impl<T> Packet<T> {
    // Creation of a packet *must* be followed by a call to postinit_lock
    // and later by inherit_blocker
    pub fn new() -> Packet<T> {
        Packet {
            queue: mpsc::Queue::new(),
            to_wake: SignalTokenCell::new(),
            channels: AtomicUsize::new(2),
            sender_drain: AtomicIsize::new(0),
            select_lock: Mutex::new(()),
        }
    }

    // This function should be used after newly created Packet
    // was wrapped with an Arc
    // In other case mutex data will be duplicated while cloning
    // and that could cause problems on platforms where it is
    // represented by opaque data structure
    pub fn postinit_lock(&self) -> MutexGuard<()> {
        self.select_lock.lock().unwrap()
    }

    // This function is used at the creation of a shared packet to inherit a
    // previously blocked thread. This is done to prevent spurious wakeups of
    // threads in select().
    //
    // This can only be called at channel-creation time
    pub fn inherit_blocker(&self,
                           token: Option<SignalToken>,
                           guard: MutexGuard<()>) {
        token.map(|token| {
            // To not overwrite signal token
            // installed after receiver timed out and started again.
            self.to_wake.store_if_empty(token);
        });

        // When the shared packet is constructed, we grabbed this lock. The
        // purpose of this lock is to ensure that abort_selection() doesn't
        // interfere with this method. After we unlock this lock, we're
        // signifying that we're done modifying self.cnt and self.to_wake and
        // the port is ready for the world to continue using it.
        drop(guard);
    }

    fn drain_queue_after_disconnected(&self) {
        assert!(self.to_wake.load_is_disconnected());

        // In this case, we have possibly failed to send our data, and
        // we need to consider re-popping the data in order to fully
        // destroy it. We must arbitrate among the multiple senders,
        // however, because the queues that we're using are
        // single-consumer queues. In order to do this, all exiting
        // pushers will use an atomic count in order to count those
        // flowing through. Pushers who see 0 are required to drain as
        // much as possible, and then can only exit when they are the
        // only pusher (otherwise they must try again).
        if self.sender_drain.fetch_add(1, Ordering::SeqCst) == 0 {
            loop {
                // drain the queue, for info on the thread yield see the
                // discussion in try_recv
                loop {
                    match self.queue.pop() {
                        mpsc::Data(..) => {}
                        mpsc::Empty => break,
                        mpsc::Inconsistent => thread::yield_now(),
                    }
                }
                // maybe we're done, if we're not the last ones
                // here, then we need to go try again.
                if self.sender_drain.fetch_sub(1, Ordering::SeqCst) == 1 {
                    break
                }
            }

            // At this point, there may still be data on the queue,
            // but only if the count hasn't been incremented and
            // some other sender hasn't finished pushing data just
            // yet. That sender in question will drain its own data.
        }
    }

    pub fn send(&self, t: T) -> Result<(), T> {
        if self.to_wake.load_is_disconnected() {
            return Err(t);
        }

        self.queue.push(t);

        if let Some(token) = self.to_wake.take_token() {
            token.signal();
        }

        // Disconnected means receiver has beed dropped just now.
        // So it does not do recv, but it can still do drain under the same lock.
        if self.to_wake.load_is_disconnected() {
            self.drain_queue_after_disconnected();
        }

        Ok(())
    }

    pub fn recv(&self, deadline: Option<Instant>) -> Result<T, Failure> {
        // This code is essentially the exact same as that found in the stream
        // case (see stream.rs)
        match self.try_recv() {
            Err(Empty) => {}
            data => return data,
        }

        let (wait_token, signal_token) = blocking::tokens();

        // Ignore disconnected, because disconnected is checked in next try_recv
        drop(self.to_wake.store_unless_disconnected(signal_token));

        match self.try_recv() {
            Err(Empty) => {}
            data => return data,
        }

        match deadline {
            Some(deadline) => {
                wait_token.wait_max_until(deadline);
            },
            None => wait_token.wait(),
        }

        // Release memory
        self.to_wake.take_token();

        self.try_recv()
    }

    pub fn try_recv(&self) -> Result<T, Failure> {
        // Disconnected flag must be loaded before queue pop to properly handle
        // race like this:
        //
        // recv: queue.pop -> empty
        // send: queue.push
        // send: drop_chan
        // recv: check disconnected flag
        let disconnected = self.to_wake.load_is_disconnected();

        let ret = match self.queue.pop() {
            mpsc::Data(t) => Some(t),
            mpsc::Empty => None,

            // This is a bit of an interesting case. The channel is reported as
            // having data available, but our pop() has failed due to the queue
            // being in an inconsistent state.  This means that there is some
            // pusher somewhere which has yet to complete, but we are guaranteed
            // that a pop will eventually succeed. In this case, we spin in a
            // yield loop because the remote sender should finish their enqueue
            // operation "very quickly".
            //
            // Avoiding this yield loop would require a different queue
            // abstraction which provides the guarantee that after M pushes have
            // succeeded, at least M pops will succeed. The current queues
            // guarantee that if there are N active pushes, you can pop N times
            // once all N have finished.
            mpsc::Inconsistent => {
                let data;
                loop {
                    thread::yield_now();
                    match self.queue.pop() {
                        mpsc::Data(t) => { data = t; break }
                        mpsc::Empty => panic!("inconsistent => empty"),
                        mpsc::Inconsistent => {}
                    }
                }
                Some(data)
            }
        };
        match ret {
            Some(data) => Ok(data),
            None => {
                if disconnected {
                    Err(Disconnected)
                } else {
                    Err(Empty)
                }
            }
        }
    }

    // Prepares this shared packet for a channel clone, essentially just bumping
    // a refcount.
    pub fn clone_chan(&self) {
        let old_count = self.channels.fetch_add(1, Ordering::SeqCst);

        // See comments on Arc::clone() on why we do this (for `mem::forget`).
        if old_count > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }
    }

    // Decrement the reference count on a channel. This is called whenever a
    // Chan is dropped and may end up waking up a receiver. It's the receiver's
    // responsibility on the other end to figure out that we've disconnected.
    pub fn drop_chan(&self) {
        match self.channels.fetch_sub(1, Ordering::SeqCst) {
            1 => {}
            n if n > 1 => return,
            n => panic!("bad number of channels left {}", n),
        }

        if let Some(signal) = self.to_wake.store_disconnected() {
            signal.signal();
        }
    }

    // See the long discussion inside of stream.rs for why the queue is drained,
    // and why it is done in this fashion.
    pub fn drop_port(&self) {
        self.to_wake.store_disconnected();

        // Must drain under lock, because sender may also drain in `send`.
        self.drain_queue_after_disconnected();
    }

    ////////////////////////////////////////////////////////////////////////////
    // select implementation
    ////////////////////////////////////////////////////////////////////////////

    // Helper function for select, tests whether this port can receive without
    // blocking (obviously not an atomic decision).
    //
    // This is different than the stream version because there's no need to peek
    // at the queue, we can just look at the local count.
    pub fn can_recv(&self) -> bool {
        self.queue.can_pop()
    }

    // Inserts the signal token for selection on this port, returning true if
    // blocking should proceed.
    //
    // The code here is the same as in stream.rs, except that it doesn't need to
    // peek at the channel to see if an upgrade is pending.
    pub fn start_selection(&self, token: SignalToken) -> StartResult {
        if self.can_recv() {
            StartResult::Abort
        } else {
            match self.to_wake.store_unless_disconnected(token) {
                Ok(()) => Installed,
                Err(CellDisconnected) => StartResult::Abort,
            }
        }
    }

    // Cancels a previous thread waiting on this port, returning whether there's
    // data on the port.
    pub fn abort_selection(&self, _was_upgrade: bool) -> bool {
        // Before we do anything else, we bounce on this lock. The reason for
        // doing this is to ensure that any upgrade-in-progress is gone and
        // done with. Without this bounce, we can race with inherit_blocker
        // about looking at and dealing with to_wake. Once we have acquired the
        // lock, we are guaranteed that inherit_blocker is done.
        {
            let _guard = self.select_lock.lock().unwrap();
        }

        self.to_wake.take_token();

        self.can_recv()
    }
}

impl<T> Drop for Packet<T> {
    fn drop(&mut self) {
        // Note that this load is not only an assert for correctness about
        // disconnection, but also a proper fence before the read of
        // `to_wake`, so this assert cannot be removed with also removing
        // the `to_wake` assert.
        assert_eq!(self.channels.load(Ordering::SeqCst), 0);
    }
}
