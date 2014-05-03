// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Communication primitives for concurrent tasks
//!
//! Rust makes it very difficult to share data among tasks to prevent race
//! conditions and to improve parallelism, but there is often a need for
//! communication between concurrent tasks. The primitives defined in this
//! module are the building blocks for synchronization in rust.
//!
//! This module provides message-based communication over channels, concretely
//! defined among three types:
//!
//! * `Sender`
//! * `SyncSender`
//! * `Receiver`
//!
//! A `Sender` or `SyncSender` is used to send data to a `Receiver`. Both
//! senders are clone-able such that many tasks can send simultaneously to one
//! receiver.  These channels are *task blocking*, not *thread blocking*. This
//! means that if one task is blocked on a channel, other tasks can continue to
//! make progress.
//!
//! Rust channels come in one of two flavors:
//!
//! 1. An asynchronous, infinitely buffered channel. The `channel()` function
//!    will return a `(Sender, Receiver)` tuple where all sends will be
//!    **asynchronous** (they never block). The channel conceptually has an
//!    infinite buffer.
//!
//! 2. A synchronous, bounded channel. The `sync_channel()` function will return
//!    a `(SyncSender, Receiver)` tuple where the storage for pending messages
//!    is a pre-allocated buffer of a fixed size. All sends will be
//!    **synchronous** by blocking until there is buffer space available. Note
//!    that a bound of 0 is allowed, causing the channel to become a
//!    "rendezvous" channel where each sender atomically hands off a message to
//!    a receiver.
//!
//! ## Failure Propagation
//!
//! In addition to being a core primitive for communicating in rust, channels
//! are the points at which failure is propagated among tasks.  Whenever the one
//! half of channel is closed, the other half will have its next operation
//! `fail!`. The purpose of this is to allow propagation of failure among tasks
//! that are linked to one another via channels.
//!
//! There are methods on both of senders and receivers to perform their
//! respective operations without failing, however.
//!
//! ## Runtime Requirements
//!
//! The channel types defined in this module generally have very few runtime
//! requirements in order to operate. The major requirement they have is for a
//! local rust `Task` to be available if any *blocking* operation is performed.
//!
//! If a local `Task` is not available (for example an FFI callback), then the
//! `send` operation is safe on a `Sender` (as well as a `send_opt`) as well as
//! the `try_send` method on a `SyncSender`, but no other operations are
//! guaranteed to be safe.
//!
//! Additionally, channels can interoperate between runtimes. If one task in a
//! program is running on libnative and another is running on libgreen, they can
//! still communicate with one another using channels.
//!
//! # Example
//!
//! Simple usage:
//!
//! ```
//! // Create a simple streaming channel
//! let (tx, rx) = channel();
//! spawn(proc() {
//!     tx.send(10);
//! });
//! assert_eq!(rx.recv(), 10);
//! ```
//!
//! Shared usage:
//!
//! ```
//! // Create a shared channel which can be sent along from many tasks
//! let (tx, rx) = channel();
//! for i in range(0, 10) {
//!     let tx = tx.clone();
//!     spawn(proc() {
//!         tx.send(i);
//!     })
//! }
//!
//! for _ in range(0, 10) {
//!     let j = rx.recv();
//!     assert!(0 <= j && j < 10);
//! }
//! ```
//!
//! Propagating failure:
//!
//! ```should_fail
//! // The call to recv() will fail!() because the channel has already hung
//! // up (or been deallocated)
//! let (tx, rx) = channel::<int>();
//! drop(tx);
//! rx.recv();
//! ```
//!
//! Synchronous channels:
//!
//! ```
//! let (tx, rx) = sync_channel(0);
//! spawn(proc() {
//!     // This will wait for the parent task to start receiving
//!     tx.send(53);
//! });
//! rx.recv();
//! ```

// A description of how Rust's channel implementation works
//
// Channels are supposed to be the basic building block for all other
// concurrent primitives that are used in Rust. As a result, the channel type
// needs to be highly optimized, flexible, and broad enough for use everywhere.
//
// The choice of implementation of all channels is to be built on lock-free data
// structures. The channels themselves are then consequently also lock-free data
// structures. As always with lock-free code, this is a very "here be dragons"
// territory, especially because I'm unaware of any academic papers which have
// gone into great length about channels of these flavors.
//
// ## Flavors of channels
//
// From the perspective of a consumer of this library, there is only one flavor
// of channel. This channel can be used as a stream and cloned to allow multiple
// senders. Under the hood, however, there are actually three flavors of
// channels in play.
//
// * Oneshots - these channels are highly optimized for the one-send use case.
//              They contain as few atomics as possible and involve one and
//              exactly one allocation.
// * Streams - these channels are optimized for the non-shared use case. They
//             use a different concurrent queue which is more tailored for this
//             use case. The initial allocation of this flavor of channel is not
//             optimized.
// * Shared - this is the most general form of channel that this module offers,
//            a channel with multiple senders. This type is as optimized as it
//            can be, but the previous two types mentioned are much faster for
//            their use-cases.
//
// ## Concurrent queues
//
// The basic idea of Rust's Sender/Receiver types is that send() never blocks, but
// recv() obviously blocks. This means that under the hood there must be some
// shared and concurrent queue holding all of the actual data.
//
// With two flavors of channels, two flavors of queues are also used. We have
// chosen to use queues from a well-known author which are abbreviated as SPSC
// and MPSC (single producer, single consumer and multiple producer, single
// consumer). SPSC queues are used for streams while MPSC queues are used for
// shared channels.
//
// ### SPSC optimizations
//
// The SPSC queue found online is essentially a linked list of nodes where one
// half of the nodes are the "queue of data" and the other half of nodes are a
// cache of unused nodes. The unused nodes are used such that an allocation is
// not required on every push() and a free doesn't need to happen on every
// pop().
//
// As found online, however, the cache of nodes is of an infinite size. This
// means that if a channel at one point in its life had 50k items in the queue,
// then the queue will always have the capacity for 50k items. I believed that
// this was an unnecessary limitation of the implementation, so I have altered
// the queue to optionally have a bound on the cache size.
//
// By default, streams will have an unbounded SPSC queue with a small-ish cache
// size. The hope is that the cache is still large enough to have very fast
// send() operations while not too large such that millions of channels can
// coexist at once.
//
// ### MPSC optimizations
//
// Right now the MPSC queue has not been optimized. Like the SPSC queue, it uses
// a linked list under the hood to earn its unboundedness, but I have not put
// forth much effort into having a cache of nodes similar to the SPSC queue.
//
// For now, I believe that this is "ok" because shared channels are not the most
// common type, but soon we may wish to revisit this queue choice and determine
// another candidate for backend storage of shared channels.
//
// ## Overview of the Implementation
//
// Now that there's a little background on the concurrent queues used, it's
// worth going into much more detail about the channels themselves. The basic
// pseudocode for a send/recv are:
//
//
//      send(t)                             recv()
//        queue.push(t)                       return if queue.pop()
//        if increment() == -1                deschedule {
//          wakeup()                            if decrement() > 0
//                                                cancel_deschedule()
//                                            }
//                                            queue.pop()
//
// As mentioned before, there are no locks in this implementation, only atomic
// instructions are used.
//
// ### The internal atomic counter
//
// Every channel has a shared counter with each half to keep track of the size
// of the queue. This counter is used to abort descheduling by the receiver and
// to know when to wake up on the sending side.
//
// As seen in the pseudocode, senders will increment this count and receivers
// will decrement the count. The theory behind this is that if a sender sees a
// -1 count, it will wake up the receiver, and if the receiver sees a 1+ count,
// then it doesn't need to block.
//
// The recv() method has a beginning call to pop(), and if successful, it needs
// to decrement the count. It is a crucial implementation detail that this
// decrement does *not* happen to the shared counter. If this were the case,
// then it would be possible for the counter to be very negative when there were
// no receivers waiting, in which case the senders would have to determine when
// it was actually appropriate to wake up a receiver.
//
// Instead, the "steal count" is kept track of separately (not atomically
// because it's only used by receivers), and then the decrement() call when
// descheduling will lump in all of the recent steals into one large decrement.
//
// The implication of this is that if a sender sees a -1 count, then there's
// guaranteed to be a waiter waiting!
//
// ## Native Implementation
//
// A major goal of these channels is to work seamlessly on and off the runtime.
// All of the previous race conditions have been worded in terms of
// scheduler-isms (which is obviously not available without the runtime).
//
// For now, native usage of channels (off the runtime) will fall back onto
// mutexes/cond vars for descheduling/atomic decisions. The no-contention path
// is still entirely lock-free, the "deschedule" blocks above are surrounded by
// a mutex and the "wakeup" blocks involve grabbing a mutex and signaling on a
// condition variable.
//
// ## Select
//
// Being able to support selection over channels has greatly influenced this
// design, and not only does selection need to work inside the runtime, but also
// outside the runtime.
//
// The implementation is fairly straightforward. The goal of select() is not to
// return some data, but only to return which channel can receive data without
// blocking. The implementation is essentially the entire blocking procedure
// followed by an increment as soon as its woken up. The cancellation procedure
// involves an increment and swapping out of to_wake to acquire ownership of the
// task to unblock.
//
// Sadly this current implementation requires multiple allocations, so I have
// seen the throughput of select() be much worse than it should be. I do not
// believe that there is anything fundamental which needs to change about these
// channels, however, in order to support a more efficient select().
//
// # Conclusion
//
// And now that you've seen all the races that I found and attempted to fix,
// here's the code for you to find some more!

use cast;
use cell::Cell;
use clone::Clone;
use iter::Iterator;
use kinds::Send;
use kinds::marker;
use mem;
use ops::Drop;
use option::{Some, None, Option};
use result::{Ok, Err, Result};
use rt::local::Local;
use rt::task::{Task, BlockedTask};
use sync::arc::UnsafeArc;

pub use comm::select::{Select, Handle};

macro_rules! test (
    { fn $name:ident() $b:block $(#[$a:meta])*} => (
        mod $name {
            #![allow(unused_imports)]

            use native;
            use comm::*;
            use prelude::*;
            use super::*;
            use super::super::*;
            use task;

            fn f() $b

            $(#[$a])* #[test] fn uv() { f() }
            $(#[$a])* #[test] fn native() {
                use native;
                let (tx, rx) = channel();
                native::task::spawn(proc() { tx.send(f()) });
                rx.recv();
            }
        }
    )
)

mod select;
mod oneshot;
mod stream;
mod shared;
mod sync;

// Use a power of 2 to allow LLVM to optimize to something that's not a
// division, this is hit pretty regularly.
static RESCHED_FREQ: int = 256;

/// The receiving-half of Rust's channel type. This half can only be owned by
/// one task
pub struct Receiver<T> {
    inner: Flavor<T>,
    receives: Cell<uint>,
    // can't share in an arc
    marker: marker::NoShare,
}

/// An iterator over messages on a receiver, this iterator will block
/// whenever `next` is called, waiting for a new message, and `None` will be
/// returned when the corresponding channel has hung up.
pub struct Messages<'a, T> {
    rx: &'a Receiver<T>
}

/// The sending-half of Rust's asynchronous channel type. This half can only be
/// owned by one task, but it can be cloned to send to other tasks.
pub struct Sender<T> {
    inner: Flavor<T>,
    sends: Cell<uint>,
    // can't share in an arc
    marker: marker::NoShare,
}

/// The sending-half of Rust's synchronous channel type. This half can only be
/// owned by one task, but it can be cloned to send to other tasks.
pub struct SyncSender<T> {
    inner: UnsafeArc<sync::Packet<T>>,
    // can't share in an arc
    marker: marker::NoShare,
}

/// This enumeration is the list of the possible reasons that try_recv could not
/// return data when called.
#[deriving(Eq, Clone, Show)]
pub enum TryRecvError {
    /// This channel is currently empty, but the sender(s) have not yet
    /// disconnected, so data may yet become available.
    Empty,
    /// This channel's sending half has become disconnected, and there will
    /// never be any more data received on this channel
    Disconnected,
}

/// This enumeration is the list of the possible error outcomes for the
/// `SyncSender::try_send` method.
#[deriving(Eq, Clone, Show)]
pub enum TrySendError<T> {
    /// The data could not be sent on the channel because it would require that
    /// the callee block to send the data.
    ///
    /// If this is a buffered channel, then the buffer is full at this time. If
    /// this is not a buffered channel, then there is no receiver available to
    /// acquire the data.
    Full(T),
    /// This channel's receiving half has disconnected, so the data could not be
    /// sent. The data is returned back to the callee in this case.
    RecvDisconnected(T),
}

enum Flavor<T> {
    Oneshot(UnsafeArc<oneshot::Packet<T>>),
    Stream(UnsafeArc<stream::Packet<T>>),
    Shared(UnsafeArc<shared::Packet<T>>),
    Sync(UnsafeArc<sync::Packet<T>>),
}

/// Creates a new asynchronous channel, returning the sender/receiver halves.
///
/// All data sent on the sender will become available on the receiver, and no
/// send will block the calling task (this channel has an "infinite buffer").
///
/// # Example
///
/// ```
/// let (tx, rx) = channel();
///
/// // Spawn off an expensive computation
/// spawn(proc() {
/// #   fn expensive_computation() {}
///     tx.send(expensive_computation());
/// });
///
/// // Do some useful work for awhile
///
/// // Let's see what that answer was
/// println!("{}", rx.recv());
/// ```
pub fn channel<T: Send>() -> (Sender<T>, Receiver<T>) {
    let (a, b) = UnsafeArc::new2(oneshot::Packet::new());
    (Sender::new(Oneshot(b)), Receiver::new(Oneshot(a)))
}

/// Creates a new synchronous, bounded channel.
///
/// Like asynchronous channels, the `Receiver` will block until a message
/// becomes available. These channels differ greatly in the semantics of the
/// sender from asynchronous channels, however.
///
/// This channel has an internal buffer on which messages will be queued. When
/// the internal buffer becomes full, future sends will *block* waiting for the
/// buffer to open up. Note that a buffer size of 0 is valid, in which case this
/// becomes  "rendezvous channel" where each send will not return until a recv
/// is paired with it.
///
/// As with asynchronous channels, all senders will fail in `send` if the
/// `Receiver` has been destroyed.
///
/// # Example
///
/// ```
/// let (tx, rx) = sync_channel(1);
///
/// // this returns immediately
/// tx.send(1);
///
/// spawn(proc() {
///     // this will block until the previous message has been received
///     tx.send(2);
/// });
///
/// assert_eq!(rx.recv(), 1);
/// assert_eq!(rx.recv(), 2);
/// ```
pub fn sync_channel<T: Send>(bound: uint) -> (SyncSender<T>, Receiver<T>) {
    let (a, b) = UnsafeArc::new2(sync::Packet::new(bound));
    (SyncSender::new(a), Receiver::new(Sync(b)))
}

////////////////////////////////////////////////////////////////////////////////
// Sender
////////////////////////////////////////////////////////////////////////////////

impl<T: Send> Sender<T> {
    fn new(inner: Flavor<T>) -> Sender<T> {
        Sender { inner: inner, sends: Cell::new(0), marker: marker::NoShare }
    }

    /// Sends a value along this channel to be received by the corresponding
    /// receiver.
    ///
    /// Rust channels are infinitely buffered so this method will never block.
    ///
    /// # Failure
    ///
    /// This function will fail if the other end of the channel has hung up.
    /// This means that if the corresponding receiver has fallen out of scope,
    /// this function will trigger a fail message saying that a message is
    /// being sent on a closed channel.
    ///
    /// Note that if this function does *not* fail, it does not mean that the
    /// data will be successfully received. All sends are placed into a queue,
    /// so it is possible for a send to succeed (the other end is alive), but
    /// then the other end could immediately disconnect.
    ///
    /// The purpose of this functionality is to propagate failure among tasks.
    /// If failure is not desired, then consider using the `send_opt` method
    pub fn send(&self, t: T) {
        if self.send_opt(t).is_err() {
            fail!("sending on a closed channel");
        }
    }

    /// Attempts to send a value on this channel, returning it back if it could
    /// not be sent.
    ///
    /// A successful send occurs when it is determined that the other end of
    /// the channel has not hung up already. An unsuccessful send would be one
    /// where the corresponding receiver has already been deallocated. Note
    /// that a return value of `Err` means that the data will never be
    /// received, but a return value of `Ok` does *not* mean that the data
    /// will be received.  It is possible for the corresponding receiver to
    /// hang up immediately after this function returns `Ok`.
    ///
    /// Like `send`, this method will never block.
    ///
    /// # Failure
    ///
    /// This method will never fail, it will return the message back to the
    /// caller if the other end is disconnected
    ///
    /// # Example
    ///
    /// ```
    /// let (tx, rx) = channel();
    ///
    /// // This send is always successful
    /// assert_eq!(tx.send_opt(1), Ok(()));
    ///
    /// // This send will fail because the receiver is gone
    /// drop(rx);
    /// assert_eq!(tx.send_opt(1), Err(1));
    /// ```
    pub fn send_opt(&self, t: T) -> Result<(), T> {
        // In order to prevent starvation of other tasks in situations where
        // a task sends repeatedly without ever receiving, we occassionally
        // yield instead of doing a send immediately.
        //
        // Don't unconditionally attempt to yield because the TLS overhead can
        // be a bit much, and also use `try_take` instead of `take` because
        // there's no reason that this send shouldn't be usable off the
        // runtime.
        let cnt = self.sends.get() + 1;
        self.sends.set(cnt);
        if cnt % (RESCHED_FREQ as uint) == 0 {
            let task: Option<~Task> = Local::try_take();
            task.map(|t| t.maybe_yield());
        }

        let (new_inner, ret) = match self.inner {
            Oneshot(ref p) => {
                let p = p.get();
                unsafe {
                    if !(*p).sent() {
                        return (*p).send(t);
                    } else {
                        let (a, b) = UnsafeArc::new2(stream::Packet::new());
                        match (*p).upgrade(Receiver::new(Stream(b))) {
                            oneshot::UpSuccess => {
                                let ret = (*a.get()).send(t);
                                (a, ret)
                            }
                            oneshot::UpDisconnected => (a, Err(t)),
                            oneshot::UpWoke(task) => {
                                // This send cannot fail because the task is
                                // asleep (we're looking at it), so the receiver
                                // can't go away.
                                (*a.get()).send(t).ok().unwrap();
                                task.wake().map(|t| t.reawaken());
                                (a, Ok(()))
                            }
                        }
                    }
                }
            }
            Stream(ref p) => return unsafe { (*p.get()).send(t) },
            Shared(ref p) => return unsafe { (*p.get()).send(t) },
            Sync(..) => unreachable!(),
        };

        unsafe {
            let mut tmp = Sender::new(Stream(new_inner));
            mem::swap(&mut cast::transmute_mut(self).inner, &mut tmp.inner);
        }
        return ret;
    }
}

impl<T: Send> Clone for Sender<T> {
    fn clone(&self) -> Sender<T> {
        let (packet, sleeper) = match self.inner {
            Oneshot(ref p) => {
                let (a, b) = UnsafeArc::new2(shared::Packet::new());
                match unsafe { (*p.get()).upgrade(Receiver::new(Shared(a))) } {
                    oneshot::UpSuccess | oneshot::UpDisconnected => (b, None),
                    oneshot::UpWoke(task) => (b, Some(task))
                }
            }
            Stream(ref p) => {
                let (a, b) = UnsafeArc::new2(shared::Packet::new());
                match unsafe { (*p.get()).upgrade(Receiver::new(Shared(a))) } {
                    stream::UpSuccess | stream::UpDisconnected => (b, None),
                    stream::UpWoke(task) => (b, Some(task)),
                }
            }
            Shared(ref p) => {
                unsafe { (*p.get()).clone_chan(); }
                return Sender::new(Shared(p.clone()));
            }
            Sync(..) => unreachable!(),
        };

        unsafe {
            (*packet.get()).inherit_blocker(sleeper);

            let mut tmp = Sender::new(Shared(packet.clone()));
            mem::swap(&mut cast::transmute_mut(self).inner, &mut tmp.inner);
        }
        Sender::new(Shared(packet))
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Sender<T> {
    fn drop(&mut self) {
        match self.inner {
            Oneshot(ref mut p) => unsafe { (*p.get()).drop_chan(); },
            Stream(ref mut p) => unsafe { (*p.get()).drop_chan(); },
            Shared(ref mut p) => unsafe { (*p.get()).drop_chan(); },
            Sync(..) => unreachable!(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// SyncSender
////////////////////////////////////////////////////////////////////////////////

impl<T: Send> SyncSender<T> {
    fn new(inner: UnsafeArc<sync::Packet<T>>) -> SyncSender<T> {
        SyncSender { inner: inner, marker: marker::NoShare }
    }

    /// Sends a value on this synchronous channel.
    ///
    /// This function will *block* until space in the internal buffer becomes
    /// available or a receiver is available to hand off the message to.
    ///
    /// Note that a successful send does *not* guarantee that the receiver will
    /// ever see the data if there is a buffer on this channel. Messages may be
    /// enqueued in the internal buffer for the receiver to receive at a later
    /// time. If the buffer size is 0, however, it can be guaranteed that the
    /// receiver has indeed received the data if this function returns success.
    ///
    /// # Failure
    ///
    /// Similarly to `Sender::send`, this function will fail if the
    /// corresponding `Receiver` for this channel has disconnected. This
    /// behavior is used to propagate failure among tasks.
    ///
    /// If failure is not desired, you can achieve the same semantics with the
    /// `SyncSender::send_opt` method which will not fail if the receiver
    /// disconnects.
    pub fn send(&self, t: T) {
        if self.send_opt(t).is_err() {
            fail!("sending on a closed channel");
        }
    }

    /// Send a value on a channel, returning it back if the receiver
    /// disconnected
    ///
    /// This method will *block* to send the value `t` on the channel, but if
    /// the value could not be sent due to the receiver disconnecting, the value
    /// is returned back to the callee. This function is similar to `try_send`,
    /// except that it will block if the channel is currently full.
    ///
    /// # Failure
    ///
    /// This function cannot fail.
    pub fn send_opt(&self, t: T) -> Result<(), T> {
        unsafe { (*self.inner.get()).send(t) }
    }

    /// Attempts to send a value on this channel without blocking.
    ///
    /// This method differs from `send_opt` by returning immediately if the
    /// channel's buffer is full or no receiver is waiting to acquire some
    /// data. Compared with `send_opt`, this function has two failure cases
    /// instead of one (one for disconnection, one for a full buffer).
    ///
    /// See `SyncSender::send` for notes about guarantees of whether the
    /// receiver has received the data or not if this function is successful.
    ///
    /// # Failure
    ///
    /// This function cannot fail
    pub fn try_send(&self, t: T) -> Result<(), TrySendError<T>> {
        unsafe { (*self.inner.get()).try_send(t) }
    }
}

impl<T: Send> Clone for SyncSender<T> {
    fn clone(&self) -> SyncSender<T> {
        unsafe { (*self.inner.get()).clone_chan(); }
        return SyncSender::new(self.inner.clone());
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for SyncSender<T> {
    fn drop(&mut self) {
        unsafe { (*self.inner.get()).drop_chan(); }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Receiver
////////////////////////////////////////////////////////////////////////////////

impl<T: Send> Receiver<T> {
    fn new(inner: Flavor<T>) -> Receiver<T> {
        Receiver { inner: inner, receives: Cell::new(0), marker: marker::NoShare }
    }

    /// Blocks waiting for a value on this receiver
    ///
    /// This function will block if necessary to wait for a corresponding send
    /// on the channel from its paired `Sender` structure. This receiver will
    /// be woken up when data is ready, and the data will be returned.
    ///
    /// # Failure
    ///
    /// Similar to channels, this method will trigger a task failure if the
    /// other end of the channel has hung up (been deallocated). The purpose of
    /// this is to propagate failure among tasks.
    ///
    /// If failure is not desired, then there are two options:
    ///
    /// * If blocking is still desired, the `recv_opt` method will return `None`
    ///   when the other end hangs up
    ///
    /// * If blocking is not desired, then the `try_recv` method will attempt to
    ///   peek at a value on this receiver.
    pub fn recv(&self) -> T {
        match self.recv_opt() {
            Ok(t) => t,
            Err(()) => fail!("receiving on a closed channel"),
        }
    }

    /// Attempts to return a pending value on this receiver without blocking
    ///
    /// This method will never block the caller in order to wait for data to
    /// become available. Instead, this will always return immediately with a
    /// possible option of pending data on the channel.
    ///
    /// This is useful for a flavor of "optimistic check" before deciding to
    /// block on a receiver.
    ///
    /// This function cannot fail.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        // If a thread is spinning in try_recv, we should take the opportunity
        // to reschedule things occasionally. See notes above in scheduling on
        // sends for why this doesn't always hit TLS, and also for why this uses
        // `try_take` instead of `take`.
        let cnt = self.receives.get() + 1;
        self.receives.set(cnt);
        if cnt % (RESCHED_FREQ as uint) == 0 {
            let task: Option<~Task> = Local::try_take();
            task.map(|t| t.maybe_yield());
        }

        loop {
            let mut new_port = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(oneshot::Empty) => return Err(Empty),
                        Err(oneshot::Disconnected) => return Err(Disconnected),
                        Err(oneshot::Upgraded(rx)) => rx,
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(stream::Empty) => return Err(Empty),
                        Err(stream::Disconnected) => return Err(Disconnected),
                        Err(stream::Upgraded(rx)) => rx,
                    }
                }
                Shared(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(shared::Empty) => return Err(Empty),
                        Err(shared::Disconnected) => return Err(Disconnected),
                    }
                }
                Sync(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(sync::Empty) => return Err(Empty),
                        Err(sync::Disconnected) => return Err(Disconnected),
                    }
                }
            };
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    /// Attempt to wait for a value on this receiver, but does not fail if the
    /// corresponding channel has hung up.
    ///
    /// This implementation of iterators for ports will always block if there is
    /// not data available on the receiver, but it will not fail in the case
    /// that the channel has been deallocated.
    ///
    /// In other words, this function has the same semantics as the `recv`
    /// method except for the failure aspect.
    ///
    /// If the channel has hung up, then `Err` is returned. Otherwise `Ok` of
    /// the value found on the receiver is returned.
    pub fn recv_opt(&self) -> Result<T, ()> {
        loop {
            let mut new_port = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Ok(t),
                        Err(oneshot::Empty) => return unreachable!(),
                        Err(oneshot::Disconnected) => return Err(()),
                        Err(oneshot::Upgraded(rx)) => rx,
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Ok(t),
                        Err(stream::Empty) => return unreachable!(),
                        Err(stream::Disconnected) => return Err(()),
                        Err(stream::Upgraded(rx)) => rx,
                    }
                }
                Shared(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Ok(t),
                        Err(shared::Empty) => return unreachable!(),
                        Err(shared::Disconnected) => return Err(()),
                    }
                }
                Sync(ref p) => return unsafe { (*p.get()).recv() }
            };
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    /// Returns an iterator which will block waiting for messages, but never
    /// `fail!`. It will return `None` when the channel has hung up.
    pub fn iter<'a>(&'a self) -> Messages<'a, T> {
        Messages { rx: self }
    }
}

impl<T: Send> select::Packet for Receiver<T> {
    fn can_recv(&self) -> bool {
        loop {
            let mut new_port = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).can_recv() } {
                        Ok(ret) => return ret,
                        Err(upgrade) => upgrade,
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).can_recv() } {
                        Ok(ret) => return ret,
                        Err(upgrade) => upgrade,
                    }
                }
                Shared(ref p) => {
                    return unsafe { (*p.get()).can_recv() };
                }
                Sync(ref p) => {
                    return unsafe { (*p.get()).can_recv() };
                }
            };
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    fn start_selection(&self, mut task: BlockedTask) -> Result<(), BlockedTask>{
        loop {
            let (t, mut new_port) = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).start_selection(task) } {
                        oneshot::SelSuccess => return Ok(()),
                        oneshot::SelCanceled(task) => return Err(task),
                        oneshot::SelUpgraded(t, rx) => (t, rx),
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).start_selection(task) } {
                        stream::SelSuccess => return Ok(()),
                        stream::SelCanceled(task) => return Err(task),
                        stream::SelUpgraded(t, rx) => (t, rx),
                    }
                }
                Shared(ref p) => {
                    return unsafe { (*p.get()).start_selection(task) };
                }
                Sync(ref p) => {
                    return unsafe { (*p.get()).start_selection(task) };
                }
            };
            task = t;
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    fn abort_selection(&self) -> bool {
        let mut was_upgrade = false;
        loop {
            let result = match self.inner {
                Oneshot(ref p) => unsafe { (*p.get()).abort_selection() },
                Stream(ref p) => unsafe {
                    (*p.get()).abort_selection(was_upgrade)
                },
                Shared(ref p) => return unsafe {
                    (*p.get()).abort_selection(was_upgrade)
                },
                Sync(ref p) => return unsafe {
                    (*p.get()).abort_selection()
                },
            };
            let mut new_port = match result { Ok(b) => return b, Err(p) => p };
            was_upgrade = true;
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }
}

impl<'a, T: Send> Iterator<T> for Messages<'a, T> {
    fn next(&mut self) -> Option<T> { self.rx.recv_opt().ok() }
}

#[unsafe_destructor]
impl<T: Send> Drop for Receiver<T> {
    fn drop(&mut self) {
        match self.inner {
            Oneshot(ref mut p) => unsafe { (*p.get()).drop_port(); },
            Stream(ref mut p) => unsafe { (*p.get()).drop_port(); },
            Shared(ref mut p) => unsafe { (*p.get()).drop_port(); },
            Sync(ref mut p) => unsafe { (*p.get()).drop_port(); },
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use native;
    use os;
    use super::*;

    pub fn stress_factor() -> uint {
        match os::getenv("RUST_TEST_STRESS") {
            Some(val) => from_str::<uint>(val).unwrap(),
            None => 1,
        }
    }

    test!(fn smoke() {
        let (tx, rx) = channel();
        tx.send(1);
        assert_eq!(rx.recv(), 1);
    })

    test!(fn drop_full() {
        let (tx, _rx) = channel();
        tx.send(box 1);
    })

    test!(fn drop_full_shared() {
        let (tx, _rx) = channel();
        drop(tx.clone());
        drop(tx.clone());
        tx.send(box 1);
    })

    test!(fn smoke_shared() {
        let (tx, rx) = channel();
        tx.send(1);
        assert_eq!(rx.recv(), 1);
        let tx = tx.clone();
        tx.send(1);
        assert_eq!(rx.recv(), 1);
    })

    test!(fn smoke_threads() {
        let (tx, rx) = channel();
        spawn(proc() {
            tx.send(1);
        });
        assert_eq!(rx.recv(), 1);
    })

    test!(fn smoke_port_gone() {
        let (tx, rx) = channel();
        drop(rx);
        tx.send(1);
    } #[should_fail])

    test!(fn smoke_shared_port_gone() {
        let (tx, rx) = channel();
        drop(rx);
        tx.send(1);
    } #[should_fail])

    test!(fn smoke_shared_port_gone2() {
        let (tx, rx) = channel();
        drop(rx);
        let tx2 = tx.clone();
        drop(tx);
        tx2.send(1);
    } #[should_fail])

    test!(fn port_gone_concurrent() {
        let (tx, rx) = channel();
        spawn(proc() {
            rx.recv();
        });
        loop { tx.send(1) }
    } #[should_fail])

    test!(fn port_gone_concurrent_shared() {
        let (tx, rx) = channel();
        let tx2 = tx.clone();
        spawn(proc() {
            rx.recv();
        });
        loop {
            tx.send(1);
            tx2.send(1);
        }
    } #[should_fail])

    test!(fn smoke_chan_gone() {
        let (tx, rx) = channel::<int>();
        drop(tx);
        rx.recv();
    } #[should_fail])

    test!(fn smoke_chan_gone_shared() {
        let (tx, rx) = channel::<()>();
        let tx2 = tx.clone();
        drop(tx);
        drop(tx2);
        rx.recv();
    } #[should_fail])

    test!(fn chan_gone_concurrent() {
        let (tx, rx) = channel();
        spawn(proc() {
            tx.send(1);
            tx.send(1);
        });
        loop { rx.recv(); }
    } #[should_fail])

    test!(fn stress() {
        let (tx, rx) = channel();
        spawn(proc() {
            for _ in range(0, 10000) { tx.send(1); }
        });
        for _ in range(0, 10000) {
            assert_eq!(rx.recv(), 1);
        }
    })

    test!(fn stress_shared() {
        static AMT: uint = 10000;
        static NTHREADS: uint = 8;
        let (tx, rx) = channel::<int>();
        let (dtx, drx) = channel::<()>();

        spawn(proc() {
            for _ in range(0, AMT * NTHREADS) {
                assert_eq!(rx.recv(), 1);
            }
            match rx.try_recv() {
                Ok(..) => fail!(),
                _ => {}
            }
            dtx.send(());
        });

        for _ in range(0, NTHREADS) {
            let tx = tx.clone();
            spawn(proc() {
                for _ in range(0, AMT) { tx.send(1); }
            });
        }
        drop(tx);
        drx.recv();
    })

    #[test]
    fn send_from_outside_runtime() {
        let (tx1, rx1) = channel::<()>();
        let (tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();
        let tx4 = tx3.clone();
        spawn(proc() {
            tx1.send(());
            for _ in range(0, 40) {
                assert_eq!(rx2.recv(), 1);
            }
            tx3.send(());
        });
        rx1.recv();
        native::task::spawn(proc() {
            for _ in range(0, 40) {
                tx2.send(1);
            }
            tx4.send(());
        });
        rx3.recv();
        rx3.recv();
    }

    #[test]
    fn recv_from_outside_runtime() {
        let (tx, rx) = channel::<int>();
        let (dtx, drx) = channel();
        native::task::spawn(proc() {
            for _ in range(0, 40) {
                assert_eq!(rx.recv(), 1);
            }
            dtx.send(());
        });
        for _ in range(0, 40) {
            tx.send(1);
        }
        drx.recv();
    }

    #[test]
    fn no_runtime() {
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();
        let tx4 = tx3.clone();
        native::task::spawn(proc() {
            assert_eq!(rx1.recv(), 1);
            tx2.send(2);
            tx4.send(());
        });
        native::task::spawn(proc() {
            tx1.send(1);
            assert_eq!(rx2.recv(), 2);
            tx3.send(());
        });
        rx3.recv();
        rx3.recv();
    }

    test!(fn oneshot_single_thread_close_port_first() {
        // Simple test of closing without sending
        let (_tx, rx) = channel::<int>();
        drop(rx);
    })

    test!(fn oneshot_single_thread_close_chan_first() {
        // Simple test of closing without sending
        let (tx, _rx) = channel::<int>();
        drop(tx);
    })

    test!(fn oneshot_single_thread_send_port_close() {
        // Testing that the sender cleans up the payload if receiver is closed
        let (tx, rx) = channel::<~int>();
        drop(rx);
        tx.send(box 0);
    } #[should_fail])

    test!(fn oneshot_single_thread_recv_chan_close() {
        // Receiving on a closed chan will fail
        let res = task::try(proc() {
            let (tx, rx) = channel::<int>();
            drop(tx);
            rx.recv();
        });
        // What is our res?
        assert!(res.is_err());
    })

    test!(fn oneshot_single_thread_send_then_recv() {
        let (tx, rx) = channel::<~int>();
        tx.send(box 10);
        assert!(rx.recv() == box 10);
    })

    test!(fn oneshot_single_thread_try_send_open() {
        let (tx, rx) = channel::<int>();
        assert!(tx.send_opt(10).is_ok());
        assert!(rx.recv() == 10);
    })

    test!(fn oneshot_single_thread_try_send_closed() {
        let (tx, rx) = channel::<int>();
        drop(rx);
        assert!(tx.send_opt(10).is_err());
    })

    test!(fn oneshot_single_thread_try_recv_open() {
        let (tx, rx) = channel::<int>();
        tx.send(10);
        assert!(rx.recv_opt() == Ok(10));
    })

    test!(fn oneshot_single_thread_try_recv_closed() {
        let (tx, rx) = channel::<int>();
        drop(tx);
        assert!(rx.recv_opt() == Err(()));
    })

    test!(fn oneshot_single_thread_peek_data() {
        let (tx, rx) = channel::<int>();
        assert_eq!(rx.try_recv(), Err(Empty))
        tx.send(10);
        assert_eq!(rx.try_recv(), Ok(10));
    })

    test!(fn oneshot_single_thread_peek_close() {
        let (tx, rx) = channel::<int>();
        drop(tx);
        assert_eq!(rx.try_recv(), Err(Disconnected));
        assert_eq!(rx.try_recv(), Err(Disconnected));
    })

    test!(fn oneshot_single_thread_peek_open() {
        let (_tx, rx) = channel::<int>();
        assert_eq!(rx.try_recv(), Err(Empty));
    })

    test!(fn oneshot_multi_task_recv_then_send() {
        let (tx, rx) = channel::<~int>();
        spawn(proc() {
            assert!(rx.recv() == box 10);
        });

        tx.send(box 10);
    })

    test!(fn oneshot_multi_task_recv_then_close() {
        let (tx, rx) = channel::<~int>();
        spawn(proc() {
            drop(tx);
        });
        let res = task::try(proc() {
            assert!(rx.recv() == box 10);
        });
        assert!(res.is_err());
    })

    test!(fn oneshot_multi_thread_close_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = channel::<int>();
            spawn(proc() {
                drop(rx);
            });
            drop(tx);
        }
    })

    test!(fn oneshot_multi_thread_send_close_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = channel::<int>();
            spawn(proc() {
                drop(rx);
            });
            let _ = task::try(proc() {
                tx.send(1);
            });
        }
    })

    test!(fn oneshot_multi_thread_recv_close_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = channel::<int>();
            spawn(proc() {
                let res = task::try(proc() {
                    rx.recv();
                });
                assert!(res.is_err());
            });
            spawn(proc() {
                spawn(proc() {
                    drop(tx);
                });
            });
        }
    })

    test!(fn oneshot_multi_thread_send_recv_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = channel();
            spawn(proc() {
                tx.send(box 10);
            });
            spawn(proc() {
                assert!(rx.recv() == box 10);
            });
        }
    })

    test!(fn stream_send_recv_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = channel();

            send(tx, 0);
            recv(rx, 0);

            fn send(tx: Sender<~int>, i: int) {
                if i == 10 { return }

                spawn(proc() {
                    tx.send(box i);
                    send(tx, i + 1);
                });
            }

            fn recv(rx: Receiver<~int>, i: int) {
                if i == 10 { return }

                spawn(proc() {
                    assert!(rx.recv() == box i);
                    recv(rx, i + 1);
                });
            }
        }
    })

    test!(fn recv_a_lot() {
        // Regression test that we don't run out of stack in scheduler context
        let (tx, rx) = channel();
        for _ in range(0, 10000) { tx.send(()); }
        for _ in range(0, 10000) { rx.recv(); }
    })

    test!(fn shared_chan_stress() {
        let (tx, rx) = channel();
        let total = stress_factor() + 100;
        for _ in range(0, total) {
            let tx = tx.clone();
            spawn(proc() {
                tx.send(());
            });
        }

        for _ in range(0, total) {
            rx.recv();
        }
    })

    test!(fn test_nested_recv_iter() {
        let (tx, rx) = channel::<int>();
        let (total_tx, total_rx) = channel::<int>();

        spawn(proc() {
            let mut acc = 0;
            for x in rx.iter() {
                acc += x;
            }
            total_tx.send(acc);
        });

        tx.send(3);
        tx.send(1);
        tx.send(2);
        drop(tx);
        assert_eq!(total_rx.recv(), 6);
    })

    test!(fn test_recv_iter_break() {
        let (tx, rx) = channel::<int>();
        let (count_tx, count_rx) = channel();

        spawn(proc() {
            let mut count = 0;
            for x in rx.iter() {
                if count >= 3 {
                    break;
                } else {
                    count += x;
                }
            }
            count_tx.send(count);
        });

        tx.send(2);
        tx.send(2);
        tx.send(2);
        let _ = tx.send_opt(2);
        drop(tx);
        assert_eq!(count_rx.recv(), 4);
    })

    test!(fn try_recv_states() {
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<()>();
        let (tx3, rx3) = channel::<()>();
        spawn(proc() {
            rx2.recv();
            tx1.send(1);
            tx3.send(());
            rx2.recv();
            drop(tx1);
            tx3.send(());
        });

        assert_eq!(rx1.try_recv(), Err(Empty));
        tx2.send(());
        rx3.recv();
        assert_eq!(rx1.try_recv(), Ok(1));
        assert_eq!(rx1.try_recv(), Err(Empty));
        tx2.send(());
        rx3.recv();
        assert_eq!(rx1.try_recv(), Err(Disconnected));
    })

    // This bug used to end up in a livelock inside of the Receiver destructor
    // because the internal state of the Shared packet was corrupted
    test!(fn destroy_upgraded_shared_port_when_sender_still_active() {
        let (tx, rx) = channel();
        let (tx2, rx2) = channel();
        spawn(proc() {
            rx.recv(); // wait on a oneshot
            drop(rx);  // destroy a shared
            tx2.send(());
        });
        // make sure the other task has gone to sleep
        for _ in range(0, 5000) { task::deschedule(); }

        // upgrade to a shared chan and send a message
        let t = tx.clone();
        drop(tx);
        t.send(());

        // wait for the child task to exit before we exit
        rx2.recv();
    })

    test!(fn sends_off_the_runtime() {
        use rt::thread::Thread;

        let (tx, rx) = channel();
        let t = Thread::start(proc() {
            for _ in range(0, 1000) {
                tx.send(());
            }
        });
        for _ in range(0, 1000) {
            rx.recv();
        }
        t.join();
    })

    test!(fn try_recvs_off_the_runtime() {
        use rt::thread::Thread;

        let (tx, rx) = channel();
        let (cdone, pdone) = channel();
        let t = Thread::start(proc() {
            let mut hits = 0;
            while hits < 10 {
                match rx.try_recv() {
                    Ok(()) => { hits += 1; }
                    Err(Empty) => { Thread::yield_now(); }
                    Err(Disconnected) => return,
                }
            }
            cdone.send(());
        });
        for _ in range(0, 10) {
            tx.send(());
        }
        t.join();
        pdone.recv();
    })
}

#[cfg(test)]
mod sync_tests {
    use prelude::*;
    use os;

    pub fn stress_factor() -> uint {
        match os::getenv("RUST_TEST_STRESS") {
            Some(val) => from_str::<uint>(val).unwrap(),
            None => 1,
        }
    }

    test!(fn smoke() {
        let (tx, rx) = sync_channel(1);
        tx.send(1);
        assert_eq!(rx.recv(), 1);
    })

    test!(fn drop_full() {
        let (tx, _rx) = sync_channel(1);
        tx.send(box 1);
    })

    test!(fn smoke_shared() {
        let (tx, rx) = sync_channel(1);
        tx.send(1);
        assert_eq!(rx.recv(), 1);
        let tx = tx.clone();
        tx.send(1);
        assert_eq!(rx.recv(), 1);
    })

    test!(fn smoke_threads() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() {
            tx.send(1);
        });
        assert_eq!(rx.recv(), 1);
    })

    test!(fn smoke_port_gone() {
        let (tx, rx) = sync_channel(0);
        drop(rx);
        tx.send(1);
    } #[should_fail])

    test!(fn smoke_shared_port_gone2() {
        let (tx, rx) = sync_channel(0);
        drop(rx);
        let tx2 = tx.clone();
        drop(tx);
        tx2.send(1);
    } #[should_fail])

    test!(fn port_gone_concurrent() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() {
            rx.recv();
        });
        loop { tx.send(1) }
    } #[should_fail])

    test!(fn port_gone_concurrent_shared() {
        let (tx, rx) = sync_channel(0);
        let tx2 = tx.clone();
        spawn(proc() {
            rx.recv();
        });
        loop {
            tx.send(1);
            tx2.send(1);
        }
    } #[should_fail])

    test!(fn smoke_chan_gone() {
        let (tx, rx) = sync_channel::<int>(0);
        drop(tx);
        rx.recv();
    } #[should_fail])

    test!(fn smoke_chan_gone_shared() {
        let (tx, rx) = sync_channel::<()>(0);
        let tx2 = tx.clone();
        drop(tx);
        drop(tx2);
        rx.recv();
    } #[should_fail])

    test!(fn chan_gone_concurrent() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() {
            tx.send(1);
            tx.send(1);
        });
        loop { rx.recv(); }
    } #[should_fail])

    test!(fn stress() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() {
            for _ in range(0, 10000) { tx.send(1); }
        });
        for _ in range(0, 10000) {
            assert_eq!(rx.recv(), 1);
        }
    })

    test!(fn stress_shared() {
        static AMT: uint = 1000;
        static NTHREADS: uint = 8;
        let (tx, rx) = sync_channel::<int>(0);
        let (dtx, drx) = sync_channel::<()>(0);

        spawn(proc() {
            for _ in range(0, AMT * NTHREADS) {
                assert_eq!(rx.recv(), 1);
            }
            match rx.try_recv() {
                Ok(..) => fail!(),
                _ => {}
            }
            dtx.send(());
        });

        for _ in range(0, NTHREADS) {
            let tx = tx.clone();
            spawn(proc() {
                for _ in range(0, AMT) { tx.send(1); }
            });
        }
        drop(tx);
        drx.recv();
    })

    test!(fn oneshot_single_thread_close_port_first() {
        // Simple test of closing without sending
        let (_tx, rx) = sync_channel::<int>(0);
        drop(rx);
    })

    test!(fn oneshot_single_thread_close_chan_first() {
        // Simple test of closing without sending
        let (tx, _rx) = sync_channel::<int>(0);
        drop(tx);
    })

    test!(fn oneshot_single_thread_send_port_close() {
        // Testing that the sender cleans up the payload if receiver is closed
        let (tx, rx) = sync_channel::<~int>(0);
        drop(rx);
        tx.send(box 0);
    } #[should_fail])

    test!(fn oneshot_single_thread_recv_chan_close() {
        // Receiving on a closed chan will fail
        let res = task::try(proc() {
            let (tx, rx) = sync_channel::<int>(0);
            drop(tx);
            rx.recv();
        });
        // What is our res?
        assert!(res.is_err());
    })

    test!(fn oneshot_single_thread_send_then_recv() {
        let (tx, rx) = sync_channel::<~int>(1);
        tx.send(box 10);
        assert!(rx.recv() == box 10);
    })

    test!(fn oneshot_single_thread_try_send_open() {
        let (tx, rx) = sync_channel::<int>(1);
        assert_eq!(tx.try_send(10), Ok(()));
        assert!(rx.recv() == 10);
    })

    test!(fn oneshot_single_thread_try_send_closed() {
        let (tx, rx) = sync_channel::<int>(0);
        drop(rx);
        assert_eq!(tx.try_send(10), Err(RecvDisconnected(10)));
    })

    test!(fn oneshot_single_thread_try_send_closed2() {
        let (tx, _rx) = sync_channel::<int>(0);
        assert_eq!(tx.try_send(10), Err(Full(10)));
    })

    test!(fn oneshot_single_thread_try_recv_open() {
        let (tx, rx) = sync_channel::<int>(1);
        tx.send(10);
        assert!(rx.recv_opt() == Ok(10));
    })

    test!(fn oneshot_single_thread_try_recv_closed() {
        let (tx, rx) = sync_channel::<int>(0);
        drop(tx);
        assert!(rx.recv_opt() == Err(()));
    })

    test!(fn oneshot_single_thread_peek_data() {
        let (tx, rx) = sync_channel::<int>(1);
        assert_eq!(rx.try_recv(), Err(Empty))
        tx.send(10);
        assert_eq!(rx.try_recv(), Ok(10));
    })

    test!(fn oneshot_single_thread_peek_close() {
        let (tx, rx) = sync_channel::<int>(0);
        drop(tx);
        assert_eq!(rx.try_recv(), Err(Disconnected));
        assert_eq!(rx.try_recv(), Err(Disconnected));
    })

    test!(fn oneshot_single_thread_peek_open() {
        let (_tx, rx) = sync_channel::<int>(0);
        assert_eq!(rx.try_recv(), Err(Empty));
    })

    test!(fn oneshot_multi_task_recv_then_send() {
        let (tx, rx) = sync_channel::<~int>(0);
        spawn(proc() {
            assert!(rx.recv() == box 10);
        });

        tx.send(box 10);
    })

    test!(fn oneshot_multi_task_recv_then_close() {
        let (tx, rx) = sync_channel::<~int>(0);
        spawn(proc() {
            drop(tx);
        });
        let res = task::try(proc() {
            assert!(rx.recv() == box 10);
        });
        assert!(res.is_err());
    })

    test!(fn oneshot_multi_thread_close_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = sync_channel::<int>(0);
            spawn(proc() {
                drop(rx);
            });
            drop(tx);
        }
    })

    test!(fn oneshot_multi_thread_send_close_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = sync_channel::<int>(0);
            spawn(proc() {
                drop(rx);
            });
            let _ = task::try(proc() {
                tx.send(1);
            });
        }
    })

    test!(fn oneshot_multi_thread_recv_close_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = sync_channel::<int>(0);
            spawn(proc() {
                let res = task::try(proc() {
                    rx.recv();
                });
                assert!(res.is_err());
            });
            spawn(proc() {
                spawn(proc() {
                    drop(tx);
                });
            });
        }
    })

    test!(fn oneshot_multi_thread_send_recv_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = sync_channel(0);
            spawn(proc() {
                tx.send(box 10);
            });
            spawn(proc() {
                assert!(rx.recv() == box 10);
            });
        }
    })

    test!(fn stream_send_recv_stress() {
        for _ in range(0, stress_factor()) {
            let (tx, rx) = sync_channel(0);

            send(tx, 0);
            recv(rx, 0);

            fn send(tx: SyncSender<~int>, i: int) {
                if i == 10 { return }

                spawn(proc() {
                    tx.send(box i);
                    send(tx, i + 1);
                });
            }

            fn recv(rx: Receiver<~int>, i: int) {
                if i == 10 { return }

                spawn(proc() {
                    assert!(rx.recv() == box i);
                    recv(rx, i + 1);
                });
            }
        }
    })

    test!(fn recv_a_lot() {
        // Regression test that we don't run out of stack in scheduler context
        let (tx, rx) = sync_channel(10000);
        for _ in range(0, 10000) { tx.send(()); }
        for _ in range(0, 10000) { rx.recv(); }
    })

    test!(fn shared_chan_stress() {
        let (tx, rx) = sync_channel(0);
        let total = stress_factor() + 100;
        for _ in range(0, total) {
            let tx = tx.clone();
            spawn(proc() {
                tx.send(());
            });
        }

        for _ in range(0, total) {
            rx.recv();
        }
    })

    test!(fn test_nested_recv_iter() {
        let (tx, rx) = sync_channel::<int>(0);
        let (total_tx, total_rx) = sync_channel::<int>(0);

        spawn(proc() {
            let mut acc = 0;
            for x in rx.iter() {
                acc += x;
            }
            total_tx.send(acc);
        });

        tx.send(3);
        tx.send(1);
        tx.send(2);
        drop(tx);
        assert_eq!(total_rx.recv(), 6);
    })

    test!(fn test_recv_iter_break() {
        let (tx, rx) = sync_channel::<int>(0);
        let (count_tx, count_rx) = sync_channel(0);

        spawn(proc() {
            let mut count = 0;
            for x in rx.iter() {
                if count >= 3 {
                    break;
                } else {
                    count += x;
                }
            }
            count_tx.send(count);
        });

        tx.send(2);
        tx.send(2);
        tx.send(2);
        let _ = tx.try_send(2);
        drop(tx);
        assert_eq!(count_rx.recv(), 4);
    })

    test!(fn try_recv_states() {
        let (tx1, rx1) = sync_channel::<int>(1);
        let (tx2, rx2) = sync_channel::<()>(1);
        let (tx3, rx3) = sync_channel::<()>(1);
        spawn(proc() {
            rx2.recv();
            tx1.send(1);
            tx3.send(());
            rx2.recv();
            drop(tx1);
            tx3.send(());
        });

        assert_eq!(rx1.try_recv(), Err(Empty));
        tx2.send(());
        rx3.recv();
        assert_eq!(rx1.try_recv(), Ok(1));
        assert_eq!(rx1.try_recv(), Err(Empty));
        tx2.send(());
        rx3.recv();
        assert_eq!(rx1.try_recv(), Err(Disconnected));
    })

    // This bug used to end up in a livelock inside of the Receiver destructor
    // because the internal state of the Shared packet was corrupted
    test!(fn destroy_upgraded_shared_port_when_sender_still_active() {
        let (tx, rx) = sync_channel(0);
        let (tx2, rx2) = sync_channel(0);
        spawn(proc() {
            rx.recv(); // wait on a oneshot
            drop(rx);  // destroy a shared
            tx2.send(());
        });
        // make sure the other task has gone to sleep
        for _ in range(0, 5000) { task::deschedule(); }

        // upgrade to a shared chan and send a message
        let t = tx.clone();
        drop(tx);
        t.send(());

        // wait for the child task to exit before we exit
        rx2.recv();
    })

    test!(fn try_recvs_off_the_runtime() {
        use std::rt::thread::Thread;

        let (tx, rx) = sync_channel(0);
        let (cdone, pdone) = channel();
        let t = Thread::start(proc() {
            let mut hits = 0;
            while hits < 10 {
                match rx.try_recv() {
                    Ok(()) => { hits += 1; }
                    Err(Empty) => { Thread::yield_now(); }
                    Err(Disconnected) => return,
                }
            }
            cdone.send(());
        });
        for _ in range(0, 10) {
            tx.send(());
        }
        t.join();
        pdone.recv();
    })

    test!(fn send_opt1() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() { rx.recv(); });
        assert_eq!(tx.send_opt(1), Ok(()));
    })

    test!(fn send_opt2() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() { drop(rx); });
        assert_eq!(tx.send_opt(1), Err(1));
    })

    test!(fn send_opt3() {
        let (tx, rx) = sync_channel(1);
        assert_eq!(tx.send_opt(1), Ok(()));
        spawn(proc() { drop(rx); });
        assert_eq!(tx.send_opt(1), Err(1));
    })

    test!(fn send_opt4() {
        let (tx, rx) = sync_channel(0);
        let tx2 = tx.clone();
        let (done, donerx) = channel();
        let done2 = done.clone();
        spawn(proc() {
            assert_eq!(tx.send_opt(1), Err(1));
            done.send(());
        });
        spawn(proc() {
            assert_eq!(tx2.send_opt(2), Err(2));
            done2.send(());
        });
        drop(rx);
        donerx.recv();
        donerx.recv();
    })

    test!(fn try_send1() {
        let (tx, _rx) = sync_channel(0);
        assert_eq!(tx.try_send(1), Err(Full(1)));
    })

    test!(fn try_send2() {
        let (tx, _rx) = sync_channel(1);
        assert_eq!(tx.try_send(1), Ok(()));
        assert_eq!(tx.try_send(1), Err(Full(1)));
    })

    test!(fn try_send3() {
        let (tx, rx) = sync_channel(1);
        assert_eq!(tx.try_send(1), Ok(()));
        drop(rx);
        assert_eq!(tx.try_send(1), Err(RecvDisconnected(1)));
    })

    test!(fn try_send4() {
        let (tx, rx) = sync_channel(0);
        spawn(proc() {
            for _ in range(0, 1000) { task::deschedule(); }
            assert_eq!(tx.try_send(1), Ok(()));
        });
        assert_eq!(rx.recv(), 1);
    } #[ignore(reason = "flaky on libnative")])
}
