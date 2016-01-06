// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Multi-producer, single-consumer FIFO queue communication primitives.
//!
//! This module provides message-based communication over channels, concretely
//! defined among three types:
//!
//! * `Sender`
//! * `SyncSender`
//! * `Receiver`
//!
//! A `Sender` or `SyncSender` is used to send data to a `Receiver`. Both
//! senders are clone-able (multi-producer) such that many threads can send
//! simultaneously to one receiver (single-consumer).
//!
//! These channels come in two flavors:
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
//! ## Disconnection
//!
//! The send and receive operations on channels will all return a `Result`
//! indicating whether the operation succeeded or not. An unsuccessful operation
//! is normally indicative of the other half of a channel having "hung up" by
//! being dropped in its corresponding thread.
//!
//! Once half of a channel has been deallocated, most operations can no longer
//! continue to make progress, so `Err` will be returned. Many applications will
//! continue to `unwrap()` the results returned from this module, instigating a
//! propagation of failure among threads if one unexpectedly dies.
//!
//! # Examples
//!
//! Simple usage:
//!
//! ```
//! use std::thread;
//! use std::sync::mpsc::channel;
//!
//! // Create a simple streaming channel
//! let (tx, rx) = channel();
//! thread::spawn(move|| {
//!     tx.send(10).unwrap();
//! });
//! assert_eq!(rx.recv().unwrap(), 10);
//! ```
//!
//! Shared usage:
//!
//! ```
//! use std::thread;
//! use std::sync::mpsc::channel;
//!
//! // Create a shared channel that can be sent along from many threads
//! // where tx is the sending half (tx for transmission), and rx is the receiving
//! // half (rx for receiving).
//! let (tx, rx) = channel();
//! for i in 0..10 {
//!     let tx = tx.clone();
//!     thread::spawn(move|| {
//!         tx.send(i).unwrap();
//!     });
//! }
//!
//! for _ in 0..10 {
//!     let j = rx.recv().unwrap();
//!     assert!(0 <= j && j < 10);
//! }
//! ```
//!
//! Propagating panics:
//!
//! ```
//! use std::sync::mpsc::channel;
//!
//! // The call to recv() will return an error because the channel has already
//! // hung up (or been deallocated)
//! let (tx, rx) = channel::<i32>();
//! drop(tx);
//! assert!(rx.recv().is_err());
//! ```
//!
//! Synchronous channels:
//!
//! ```
//! use std::thread;
//! use std::sync::mpsc::sync_channel;
//!
//! let (tx, rx) = sync_channel::<i32>(0);
//! thread::spawn(move|| {
//!     // This will wait for the parent thread to start receiving
//!     tx.send(53).unwrap();
//! });
//! rx.recv().unwrap();
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

// A description of how Rust's channel implementation works
//
// Channels are supposed to be the basic building block for all other
// concurrent primitives that are used in Rust. As a result, the channel type
// needs to be highly optimized, flexible, and broad enough for use everywhere.
//
// The choice of implementation of all channels is to be built on lock-free data
// structures. The channels themselves are then consequently also lock-free data
// structures. As always with lock-free code, this is a very "here be dragons"
// territory, especially because I'm unaware of any academic papers that have
// gone into great length about channels of these flavors.
//
// ## Flavors of channels
//
// From the perspective of a consumer of this library, there is only one flavor
// of channel. This channel can be used as a stream and cloned to allow multiple
// senders. Under the hood, however, there are actually three flavors of
// channels in play.
//
// * Flavor::Oneshots - these channels are highly optimized for the one-send use case.
//              They contain as few atomics as possible and involve one and
//              exactly one allocation.
// * Streams - these channels are optimized for the non-shared use case. They
//             use a different concurrent queue that is more tailored for this
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
// chosen to use queues from a well-known author that are abbreviated as SPSC
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
// thread to unblock.
//
// Sadly this current implementation requires multiple allocations, so I have
// seen the throughput of select() be much worse than it should be. I do not
// believe that there is anything fundamental that needs to change about these
// channels, however, in order to support a more efficient select().
//
// # Conclusion
//
// And now that you've seen all the races that I found and attempted to fix,
// here's the code for you to find some more!

use sync::Arc;
use error;
use fmt;
use mem;
use cell::UnsafeCell;
use marker::Reflect;

#[unstable(feature = "mpsc_select", issue = "27800")]
pub use self::select::{Select, Handle};
use self::select::StartResult;
use self::select::StartResult::*;
use self::blocking::SignalToken;

mod blocking;
mod oneshot;
mod select;
mod shared;
mod stream;
mod sync;
mod mpsc_queue;
mod spsc_queue;

/// The receiving-half of Rust's channel type. This half can only be owned by
/// one thread
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Receiver<T> {
    inner: UnsafeCell<Flavor<T>>,
}

// The receiver port can be sent from place to place, so long as it
// is not used to receive non-sendable things.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for Receiver<T> { }

/// An iterator over messages on a receiver, this iterator will block
/// whenever `next` is called, waiting for a new message, and `None` will be
/// returned when the corresponding channel has hung up.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    rx: &'a Receiver<T>
}

/// An owning iterator over messages on a receiver, this iterator will block
/// whenever `next` is called, waiting for a new message, and `None` will be
/// returned when the corresponding channel has hung up.
#[stable(feature = "receiver_into_iter", since = "1.1.0")]
pub struct IntoIter<T> {
    rx: Receiver<T>
}

/// The sending-half of Rust's asynchronous channel type. This half can only be
/// owned by one thread, but it can be cloned to send to other threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Sender<T> {
    inner: UnsafeCell<Flavor<T>>,
}

// The send port can be sent from place to place, so long as it
// is not used to send non-sendable things.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for Sender<T> { }

/// The sending-half of Rust's synchronous channel type. This half can only be
/// owned by one thread, but it can be cloned to send to other threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SyncSender<T> {
    inner: Arc<UnsafeCell<sync::Packet<T>>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for SyncSender<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> !Sync for SyncSender<T> {}

/// An error returned from the `send` function on channels.
///
/// A `send` operation can only fail if the receiving end of a channel is
/// disconnected, implying that the data could never be received. The error
/// contains the data being sent as a payload so it can be recovered.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct SendError<T>(#[stable(feature = "rust1", since = "1.0.0")] pub T);

/// An error returned from the `recv` function on a `Receiver`.
///
/// The `recv` operation can only fail if the sending half of a channel is
/// disconnected, implying that no further messages will ever be received.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RecvError;

/// This enumeration is the list of the possible reasons that `try_recv` could
/// not return data when called.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum TryRecvError {
    /// This channel is currently empty, but the sender(s) have not yet
    /// disconnected, so data may yet become available.
    #[stable(feature = "rust1", since = "1.0.0")]
    Empty,

    /// This channel's sending half has become disconnected, and there will
    /// never be any more data received on this channel
    #[stable(feature = "rust1", since = "1.0.0")]
    Disconnected,
}

/// This enumeration is the list of the possible error outcomes for the
/// `SyncSender::try_send` method.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TrySendError<T> {
    /// The data could not be sent on the channel because it would require that
    /// the callee block to send the data.
    ///
    /// If this is a buffered channel, then the buffer is full at this time. If
    /// this is not a buffered channel, then there is no receiver available to
    /// acquire the data.
    #[stable(feature = "rust1", since = "1.0.0")]
    Full(T),

    /// This channel's receiving half has disconnected, so the data could not be
    /// sent. The data is returned back to the callee in this case.
    #[stable(feature = "rust1", since = "1.0.0")]
    Disconnected(T),
}

enum Flavor<T> {
    Oneshot(Arc<UnsafeCell<oneshot::Packet<T>>>),
    Stream(Arc<UnsafeCell<stream::Packet<T>>>),
    Shared(Arc<UnsafeCell<shared::Packet<T>>>),
    Sync(Arc<UnsafeCell<sync::Packet<T>>>),
}

#[doc(hidden)]
trait UnsafeFlavor<T> {
    fn inner_unsafe(&self) -> &UnsafeCell<Flavor<T>>;
    unsafe fn inner_mut<'a>(&'a self) -> &'a mut Flavor<T> {
        &mut *self.inner_unsafe().get()
    }
    unsafe fn inner<'a>(&'a self) -> &'a Flavor<T> {
        &*self.inner_unsafe().get()
    }
}
impl<T> UnsafeFlavor<T> for Sender<T> {
    fn inner_unsafe(&self) -> &UnsafeCell<Flavor<T>> {
        &self.inner
    }
}
impl<T> UnsafeFlavor<T> for Receiver<T> {
    fn inner_unsafe(&self) -> &UnsafeCell<Flavor<T>> {
        &self.inner
    }
}

/// Creates a new asynchronous channel, returning the sender/receiver halves.
///
/// All data sent on the sender will become available on the receiver, and no
/// send will block the calling thread (this channel has an "infinite buffer").
///
/// # Examples
///
/// ```
/// use std::sync::mpsc::channel;
/// use std::thread;
///
/// // tx is the sending half (tx for transmission), and rx is the receiving
/// // half (rx for receiving).
/// let (tx, rx) = channel();
///
/// // Spawn off an expensive computation
/// thread::spawn(move|| {
/// #   fn expensive_computation() {}
///     tx.send(expensive_computation()).unwrap();
/// });
///
/// // Do some useful work for awhile
///
/// // Let's see what that answer was
/// println!("{:?}", rx.recv().unwrap());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let a = Arc::new(UnsafeCell::new(oneshot::Packet::new()));
    (Sender::new(Flavor::Oneshot(a.clone())), Receiver::new(Flavor::Oneshot(a)))
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
/// As with asynchronous channels, all senders will panic in `send` if the
/// `Receiver` has been destroyed.
///
/// # Examples
///
/// ```
/// use std::sync::mpsc::sync_channel;
/// use std::thread;
///
/// let (tx, rx) = sync_channel(1);
///
/// // this returns immediately
/// tx.send(1).unwrap();
///
/// thread::spawn(move|| {
///     // this will block until the previous message has been received
///     tx.send(2).unwrap();
/// });
///
/// assert_eq!(rx.recv().unwrap(), 1);
/// assert_eq!(rx.recv().unwrap(), 2);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn sync_channel<T>(bound: usize) -> (SyncSender<T>, Receiver<T>) {
    let a = Arc::new(UnsafeCell::new(sync::Packet::new(bound)));
    (SyncSender::new(a.clone()), Receiver::new(Flavor::Sync(a)))
}

////////////////////////////////////////////////////////////////////////////////
// Sender
////////////////////////////////////////////////////////////////////////////////

impl<T> Sender<T> {
    fn new(inner: Flavor<T>) -> Sender<T> {
        Sender {
            inner: UnsafeCell::new(inner),
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
    /// This method will never block the current thread.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::mpsc::channel;
    ///
    /// let (tx, rx) = channel();
    ///
    /// // This send is always successful
    /// tx.send(1).unwrap();
    ///
    /// // This send will fail because the receiver is gone
    /// drop(rx);
    /// assert_eq!(tx.send(1).err().unwrap().0, 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send(&self, t: T) -> Result<(), SendError<T>> {
        let (new_inner, ret) = match *unsafe { self.inner() } {
            Flavor::Oneshot(ref p) => {
                unsafe {
                    let p = p.get();
                    if !(*p).sent() {
                        return (*p).send(t).map_err(SendError);
                    } else {
                        let a =
                            Arc::new(UnsafeCell::new(stream::Packet::new()));
                        let rx = Receiver::new(Flavor::Stream(a.clone()));
                        match (*p).upgrade(rx) {
                            oneshot::UpSuccess => {
                                let ret = (*a.get()).send(t);
                                (a, ret)
                            }
                            oneshot::UpDisconnected => (a, Err(t)),
                            oneshot::UpWoke(token) => {
                                // This send cannot panic because the thread is
                                // asleep (we're looking at it), so the receiver
                                // can't go away.
                                (*a.get()).send(t).ok().unwrap();
                                token.signal();
                                (a, Ok(()))
                            }
                        }
                    }
                }
            }
            Flavor::Stream(ref p) => return unsafe {
                (*p.get()).send(t).map_err(SendError)
            },
            Flavor::Shared(ref p) => return unsafe {
                (*p.get()).send(t).map_err(SendError)
            },
            Flavor::Sync(..) => unreachable!(),
        };

        unsafe {
            let tmp = Sender::new(Flavor::Stream(new_inner));
            mem::swap(self.inner_mut(), tmp.inner_mut());
        }
        ret.map_err(SendError)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Sender<T> {
    fn clone(&self) -> Sender<T> {
        let (packet, sleeper, guard) = match *unsafe { self.inner() } {
            Flavor::Oneshot(ref p) => {
                let a = Arc::new(UnsafeCell::new(shared::Packet::new()));
                unsafe {
                    let guard = (*a.get()).postinit_lock();
                    let rx = Receiver::new(Flavor::Shared(a.clone()));
                    match (*p.get()).upgrade(rx) {
                        oneshot::UpSuccess |
                        oneshot::UpDisconnected => (a, None, guard),
                        oneshot::UpWoke(task) => (a, Some(task), guard)
                    }
                }
            }
            Flavor::Stream(ref p) => {
                let a = Arc::new(UnsafeCell::new(shared::Packet::new()));
                unsafe {
                    let guard = (*a.get()).postinit_lock();
                    let rx = Receiver::new(Flavor::Shared(a.clone()));
                    match (*p.get()).upgrade(rx) {
                        stream::UpSuccess |
                        stream::UpDisconnected => (a, None, guard),
                        stream::UpWoke(task) => (a, Some(task), guard),
                    }
                }
            }
            Flavor::Shared(ref p) => {
                unsafe { (*p.get()).clone_chan(); }
                return Sender::new(Flavor::Shared(p.clone()));
            }
            Flavor::Sync(..) => unreachable!(),
        };

        unsafe {
            (*packet.get()).inherit_blocker(sleeper, guard);

            let tmp = Sender::new(Flavor::Shared(packet.clone()));
            mem::swap(self.inner_mut(), tmp.inner_mut());
        }
        Sender::new(Flavor::Shared(packet))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        match *unsafe { self.inner_mut() } {
            Flavor::Oneshot(ref p) => unsafe { (*p.get()).drop_chan(); },
            Flavor::Stream(ref p) => unsafe { (*p.get()).drop_chan(); },
            Flavor::Shared(ref p) => unsafe { (*p.get()).drop_chan(); },
            Flavor::Sync(..) => unreachable!(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// SyncSender
////////////////////////////////////////////////////////////////////////////////

impl<T> SyncSender<T> {
    fn new(inner: Arc<UnsafeCell<sync::Packet<T>>>) -> SyncSender<T> {
        SyncSender { inner: inner }
    }

    /// Sends a value on this synchronous channel.
    ///
    /// This function will *block* until space in the internal buffer becomes
    /// available or a receiver is available to hand off the message to.
    ///
    /// Note that a successful send does *not* guarantee that the receiver will
    /// ever see the data if there is a buffer on this channel. Items may be
    /// enqueued in the internal buffer for the receiver to receive at a later
    /// time. If the buffer size is 0, however, it can be guaranteed that the
    /// receiver has indeed received the data if this function returns success.
    ///
    /// This function will never panic, but it may return `Err` if the
    /// `Receiver` has disconnected and is no longer able to receive
    /// information.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send(&self, t: T) -> Result<(), SendError<T>> {
        unsafe { (*self.inner.get()).send(t).map_err(SendError) }
    }

    /// Attempts to send a value on this channel without blocking.
    ///
    /// This method differs from `send` by returning immediately if the
    /// channel's buffer is full or no receiver is waiting to acquire some
    /// data. Compared with `send`, this function has two failure cases
    /// instead of one (one for disconnection, one for a full buffer).
    ///
    /// See `SyncSender::send` for notes about guarantees of whether the
    /// receiver has received the data or not if this function is successful.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_send(&self, t: T) -> Result<(), TrySendError<T>> {
        unsafe { (*self.inner.get()).try_send(t) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for SyncSender<T> {
    fn clone(&self) -> SyncSender<T> {
        unsafe { (*self.inner.get()).clone_chan(); }
        SyncSender::new(self.inner.clone())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for SyncSender<T> {
    fn drop(&mut self) {
        unsafe { (*self.inner.get()).drop_chan(); }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Receiver
////////////////////////////////////////////////////////////////////////////////

impl<T> Receiver<T> {
    fn new(inner: Flavor<T>) -> Receiver<T> {
        Receiver { inner: UnsafeCell::new(inner) }
    }

    /// Attempts to return a pending value on this receiver without blocking
    ///
    /// This method will never block the caller in order to wait for data to
    /// become available. Instead, this will always return immediately with a
    /// possible option of pending data on the channel.
    ///
    /// This is useful for a flavor of "optimistic check" before deciding to
    /// block on a receiver.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        loop {
            let new_port = match *unsafe { self.inner() } {
                Flavor::Oneshot(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(oneshot::Empty) => return Err(TryRecvError::Empty),
                        Err(oneshot::Disconnected) => {
                            return Err(TryRecvError::Disconnected)
                        }
                        Err(oneshot::Upgraded(rx)) => rx,
                    }
                }
                Flavor::Stream(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(stream::Empty) => return Err(TryRecvError::Empty),
                        Err(stream::Disconnected) => {
                            return Err(TryRecvError::Disconnected)
                        }
                        Err(stream::Upgraded(rx)) => rx,
                    }
                }
                Flavor::Shared(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(shared::Empty) => return Err(TryRecvError::Empty),
                        Err(shared::Disconnected) => {
                            return Err(TryRecvError::Disconnected)
                        }
                    }
                }
                Flavor::Sync(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Ok(t),
                        Err(sync::Empty) => return Err(TryRecvError::Empty),
                        Err(sync::Disconnected) => {
                            return Err(TryRecvError::Disconnected)
                        }
                    }
                }
            };
            unsafe {
                mem::swap(self.inner_mut(),
                          new_port.inner_mut());
            }
        }
    }

    /// Attempts to wait for a value on this receiver, returning an error if the
    /// corresponding channel has hung up.
    ///
    /// This function will always block the current thread if there is no data
    /// available and it's possible for more data to be sent. Once a message is
    /// sent to the corresponding `Sender`, then this receiver will wake up and
    /// return that message.
    ///
    /// If the corresponding `Sender` has disconnected, or it disconnects while
    /// this call is blocking, this call will wake up and return `Err` to
    /// indicate that no more messages can ever be received on this channel.
    /// However, since channels are buffered, messages sent before the disconnect
    /// will still be properly received.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::mpsc;
    /// use std::thread;
    ///
    /// let (send, recv) = mpsc::channel();
    /// let handle = thread::spawn(move || {
    ///     send.send(1u8).unwrap();
    /// });
    ///
    /// handle.join().unwrap();
    ///
    /// assert_eq!(Ok(1), recv.recv());
    /// ```
    ///
    /// Buffering behavior:
    ///
    /// ```
    /// use std::sync::mpsc;
    /// use std::thread;
    /// use std::sync::mpsc::RecvError;
    ///
    /// let (send, recv) = mpsc::channel();
    /// let handle = thread::spawn(move || {
    ///     send.send(1u8).unwrap();
    ///     send.send(2).unwrap();
    ///     send.send(3).unwrap();
    ///     drop(send);
    /// });
    ///
    /// // wait for the thread to join so we ensure the sender is dropped
    /// handle.join().unwrap();
    ///
    /// assert_eq!(Ok(1), recv.recv());
    /// assert_eq!(Ok(2), recv.recv());
    /// assert_eq!(Ok(3), recv.recv());
    /// assert_eq!(Err(RecvError), recv.recv());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn recv(&self) -> Result<T, RecvError> {
        loop {
            let new_port = match *unsafe { self.inner() } {
                Flavor::Oneshot(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Ok(t),
                        Err(oneshot::Empty) => return unreachable!(),
                        Err(oneshot::Disconnected) => return Err(RecvError),
                        Err(oneshot::Upgraded(rx)) => rx,
                    }
                }
                Flavor::Stream(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Ok(t),
                        Err(stream::Empty) => return unreachable!(),
                        Err(stream::Disconnected) => return Err(RecvError),
                        Err(stream::Upgraded(rx)) => rx,
                    }
                }
                Flavor::Shared(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Ok(t),
                        Err(shared::Empty) => return unreachable!(),
                        Err(shared::Disconnected) => return Err(RecvError),
                    }
                }
                Flavor::Sync(ref p) => return unsafe {
                    (*p.get()).recv().map_err(|()| RecvError)
                }
            };
            unsafe {
                mem::swap(self.inner_mut(), new_port.inner_mut());
            }
        }
    }

    /// Returns an iterator that will block waiting for messages, but never
    /// `panic!`. It will return `None` when the channel has hung up.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter { rx: self }
    }
}

impl<T> select::Packet for Receiver<T> {
    fn can_recv(&self) -> bool {
        loop {
            let new_port = match *unsafe { self.inner() } {
                Flavor::Oneshot(ref p) => {
                    match unsafe { (*p.get()).can_recv() } {
                        Ok(ret) => return ret,
                        Err(upgrade) => upgrade,
                    }
                }
                Flavor::Stream(ref p) => {
                    match unsafe { (*p.get()).can_recv() } {
                        Ok(ret) => return ret,
                        Err(upgrade) => upgrade,
                    }
                }
                Flavor::Shared(ref p) => {
                    return unsafe { (*p.get()).can_recv() };
                }
                Flavor::Sync(ref p) => {
                    return unsafe { (*p.get()).can_recv() };
                }
            };
            unsafe {
                mem::swap(self.inner_mut(),
                          new_port.inner_mut());
            }
        }
    }

    fn start_selection(&self, mut token: SignalToken) -> StartResult {
        loop {
            let (t, new_port) = match *unsafe { self.inner() } {
                Flavor::Oneshot(ref p) => {
                    match unsafe { (*p.get()).start_selection(token) } {
                        oneshot::SelSuccess => return Installed,
                        oneshot::SelCanceled => return Abort,
                        oneshot::SelUpgraded(t, rx) => (t, rx),
                    }
                }
                Flavor::Stream(ref p) => {
                    match unsafe { (*p.get()).start_selection(token) } {
                        stream::SelSuccess => return Installed,
                        stream::SelCanceled => return Abort,
                        stream::SelUpgraded(t, rx) => (t, rx),
                    }
                }
                Flavor::Shared(ref p) => {
                    return unsafe { (*p.get()).start_selection(token) };
                }
                Flavor::Sync(ref p) => {
                    return unsafe { (*p.get()).start_selection(token) };
                }
            };
            token = t;
            unsafe {
                mem::swap(self.inner_mut(), new_port.inner_mut());
            }
        }
    }

    fn abort_selection(&self) -> bool {
        let mut was_upgrade = false;
        loop {
            let result = match *unsafe { self.inner() } {
                Flavor::Oneshot(ref p) => unsafe { (*p.get()).abort_selection() },
                Flavor::Stream(ref p) => unsafe {
                    (*p.get()).abort_selection(was_upgrade)
                },
                Flavor::Shared(ref p) => return unsafe {
                    (*p.get()).abort_selection(was_upgrade)
                },
                Flavor::Sync(ref p) => return unsafe {
                    (*p.get()).abort_selection()
                },
            };
            let new_port = match result { Ok(b) => return b, Err(p) => p };
            was_upgrade = true;
            unsafe {
                mem::swap(self.inner_mut(),
                          new_port.inner_mut());
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> { self.rx.recv().ok() }
}

#[stable(feature = "receiver_into_iter", since = "1.1.0")]
impl<'a, T> IntoIterator for &'a Receiver<T> {
    type Item = T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> { self.iter() }
}

#[stable(feature = "receiver_into_iter", since = "1.1.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> { self.rx.recv().ok() }
}

#[stable(feature = "receiver_into_iter", since = "1.1.0")]
impl <T> IntoIterator for Receiver<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter { rx: self }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        match *unsafe { self.inner_mut() } {
            Flavor::Oneshot(ref p) => unsafe { (*p.get()).drop_port(); },
            Flavor::Stream(ref p) => unsafe { (*p.get()).drop_port(); },
            Flavor::Shared(ref p) => unsafe { (*p.get()).drop_port(); },
            Flavor::Sync(ref p) => unsafe { (*p.get()).drop_port(); },
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Debug for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "SendError(..)".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "sending on a closed channel".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Send + Reflect> error::Error for SendError<T> {
    fn description(&self) -> &str {
        "sending on a closed channel"
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Debug for TrySendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TrySendError::Full(..) => "Full(..)".fmt(f),
            TrySendError::Disconnected(..) => "Disconnected(..)".fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Display for TrySendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TrySendError::Full(..) => {
                "sending on a full channel".fmt(f)
            }
            TrySendError::Disconnected(..) => {
                "sending on a closed channel".fmt(f)
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Send + Reflect> error::Error for TrySendError<T> {

    fn description(&self) -> &str {
        match *self {
            TrySendError::Full(..) => {
                "sending on a full channel"
            }
            TrySendError::Disconnected(..) => {
                "sending on a closed channel"
            }
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for RecvError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "receiving on a closed channel".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for RecvError {

    fn description(&self) -> &str {
        "receiving on a closed channel"
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for TryRecvError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TryRecvError::Empty => {
                "receiving on an empty channel".fmt(f)
            }
            TryRecvError::Disconnected => {
                "receiving on a closed channel".fmt(f)
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for TryRecvError {

    fn description(&self) -> &str {
        match *self {
            TryRecvError::Empty => {
                "receiving on an empty channel"
            }
            TryRecvError::Disconnected => {
                "receiving on a closed channel"
            }
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use env;
    use super::*;
    use thread;

    pub fn stress_factor() -> usize {
        match env::var("RUST_TEST_STRESS") {
            Ok(val) => val.parse().unwrap(),
            Err(..) => 1,
        }
    }

    #[test]
    fn smoke() {
        let (tx, rx) = channel::<i32>();
        tx.send(1).unwrap();
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn drop_full() {
        let (tx, _rx) = channel::<Box<isize>>();
        tx.send(box 1).unwrap();
    }

    #[test]
    fn drop_full_shared() {
        let (tx, _rx) = channel::<Box<isize>>();
        drop(tx.clone());
        drop(tx.clone());
        tx.send(box 1).unwrap();
    }

    #[test]
    fn smoke_shared() {
        let (tx, rx) = channel::<i32>();
        tx.send(1).unwrap();
        assert_eq!(rx.recv().unwrap(), 1);
        let tx = tx.clone();
        tx.send(1).unwrap();
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn smoke_threads() {
        let (tx, rx) = channel::<i32>();
        let _t = thread::spawn(move|| {
            tx.send(1).unwrap();
        });
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn smoke_port_gone() {
        let (tx, rx) = channel::<i32>();
        drop(rx);
        assert!(tx.send(1).is_err());
    }

    #[test]
    fn smoke_shared_port_gone() {
        let (tx, rx) = channel::<i32>();
        drop(rx);
        assert!(tx.send(1).is_err())
    }

    #[test]
    fn smoke_shared_port_gone2() {
        let (tx, rx) = channel::<i32>();
        drop(rx);
        let tx2 = tx.clone();
        drop(tx);
        assert!(tx2.send(1).is_err());
    }

    #[test]
    fn port_gone_concurrent() {
        let (tx, rx) = channel::<i32>();
        let _t = thread::spawn(move|| {
            rx.recv().unwrap();
        });
        while tx.send(1).is_ok() {}
    }

    #[test]
    fn port_gone_concurrent_shared() {
        let (tx, rx) = channel::<i32>();
        let tx2 = tx.clone();
        let _t = thread::spawn(move|| {
            rx.recv().unwrap();
        });
        while tx.send(1).is_ok() && tx2.send(1).is_ok() {}
    }

    #[test]
    fn smoke_chan_gone() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn smoke_chan_gone_shared() {
        let (tx, rx) = channel::<()>();
        let tx2 = tx.clone();
        drop(tx);
        drop(tx2);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn chan_gone_concurrent() {
        let (tx, rx) = channel::<i32>();
        let _t = thread::spawn(move|| {
            tx.send(1).unwrap();
            tx.send(1).unwrap();
        });
        while rx.recv().is_ok() {}
    }

    #[test]
    fn stress() {
        let (tx, rx) = channel::<i32>();
        let t = thread::spawn(move|| {
            for _ in 0..10000 { tx.send(1).unwrap(); }
        });
        for _ in 0..10000 {
            assert_eq!(rx.recv().unwrap(), 1);
        }
        t.join().ok().unwrap();
    }

    #[test]
    fn stress_shared() {
        const AMT: u32 = 10000;
        const NTHREADS: u32 = 8;
        let (tx, rx) = channel::<i32>();

        let t = thread::spawn(move|| {
            for _ in 0..AMT * NTHREADS {
                assert_eq!(rx.recv().unwrap(), 1);
            }
            match rx.try_recv() {
                Ok(..) => panic!(),
                _ => {}
            }
        });

        for _ in 0..NTHREADS {
            let tx = tx.clone();
            thread::spawn(move|| {
                for _ in 0..AMT { tx.send(1).unwrap(); }
            });
        }
        drop(tx);
        t.join().ok().unwrap();
    }

    #[test]
    fn send_from_outside_runtime() {
        let (tx1, rx1) = channel::<()>();
        let (tx2, rx2) = channel::<i32>();
        let t1 = thread::spawn(move|| {
            tx1.send(()).unwrap();
            for _ in 0..40 {
                assert_eq!(rx2.recv().unwrap(), 1);
            }
        });
        rx1.recv().unwrap();
        let t2 = thread::spawn(move|| {
            for _ in 0..40 {
                tx2.send(1).unwrap();
            }
        });
        t1.join().ok().unwrap();
        t2.join().ok().unwrap();
    }

    #[test]
    fn recv_from_outside_runtime() {
        let (tx, rx) = channel::<i32>();
        let t = thread::spawn(move|| {
            for _ in 0..40 {
                assert_eq!(rx.recv().unwrap(), 1);
            }
        });
        for _ in 0..40 {
            tx.send(1).unwrap();
        }
        t.join().ok().unwrap();
    }

    #[test]
    fn no_runtime() {
        let (tx1, rx1) = channel::<i32>();
        let (tx2, rx2) = channel::<i32>();
        let t1 = thread::spawn(move|| {
            assert_eq!(rx1.recv().unwrap(), 1);
            tx2.send(2).unwrap();
        });
        let t2 = thread::spawn(move|| {
            tx1.send(1).unwrap();
            assert_eq!(rx2.recv().unwrap(), 2);
        });
        t1.join().ok().unwrap();
        t2.join().ok().unwrap();
    }

    #[test]
    fn oneshot_single_thread_close_port_first() {
        // Simple test of closing without sending
        let (_tx, rx) = channel::<i32>();
        drop(rx);
    }

    #[test]
    fn oneshot_single_thread_close_chan_first() {
        // Simple test of closing without sending
        let (tx, _rx) = channel::<i32>();
        drop(tx);
    }

    #[test]
    fn oneshot_single_thread_send_port_close() {
        // Testing that the sender cleans up the payload if receiver is closed
        let (tx, rx) = channel::<Box<i32>>();
        drop(rx);
        assert!(tx.send(box 0).is_err());
    }

    #[test]
    fn oneshot_single_thread_recv_chan_close() {
        // Receiving on a closed chan will panic
        let res = thread::spawn(move|| {
            let (tx, rx) = channel::<i32>();
            drop(tx);
            rx.recv().unwrap();
        }).join();
        // What is our res?
        assert!(res.is_err());
    }

    #[test]
    fn oneshot_single_thread_send_then_recv() {
        let (tx, rx) = channel::<Box<i32>>();
        tx.send(box 10).unwrap();
        assert!(rx.recv().unwrap() == box 10);
    }

    #[test]
    fn oneshot_single_thread_try_send_open() {
        let (tx, rx) = channel::<i32>();
        assert!(tx.send(10).is_ok());
        assert!(rx.recv().unwrap() == 10);
    }

    #[test]
    fn oneshot_single_thread_try_send_closed() {
        let (tx, rx) = channel::<i32>();
        drop(rx);
        assert!(tx.send(10).is_err());
    }

    #[test]
    fn oneshot_single_thread_try_recv_open() {
        let (tx, rx) = channel::<i32>();
        tx.send(10).unwrap();
        assert!(rx.recv() == Ok(10));
    }

    #[test]
    fn oneshot_single_thread_try_recv_closed() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn oneshot_single_thread_peek_data() {
        let (tx, rx) = channel::<i32>();
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        tx.send(10).unwrap();
        assert_eq!(rx.try_recv(), Ok(10));
    }

    #[test]
    fn oneshot_single_thread_peek_close() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn oneshot_single_thread_peek_open() {
        let (_tx, rx) = channel::<i32>();
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn oneshot_multi_task_recv_then_send() {
        let (tx, rx) = channel::<Box<i32>>();
        let _t = thread::spawn(move|| {
            assert!(rx.recv().unwrap() == box 10);
        });

        tx.send(box 10).unwrap();
    }

    #[test]
    fn oneshot_multi_task_recv_then_close() {
        let (tx, rx) = channel::<Box<i32>>();
        let _t = thread::spawn(move|| {
            drop(tx);
        });
        let res = thread::spawn(move|| {
            assert!(rx.recv().unwrap() == box 10);
        }).join();
        assert!(res.is_err());
    }

    #[test]
    fn oneshot_multi_thread_close_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = channel::<i32>();
            let _t = thread::spawn(move|| {
                drop(rx);
            });
            drop(tx);
        }
    }

    #[test]
    fn oneshot_multi_thread_send_close_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = channel::<i32>();
            let _t = thread::spawn(move|| {
                drop(rx);
            });
            let _ = thread::spawn(move|| {
                tx.send(1).unwrap();
            }).join();
        }
    }

    #[test]
    fn oneshot_multi_thread_recv_close_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = channel::<i32>();
            thread::spawn(move|| {
                let res = thread::spawn(move|| {
                    rx.recv().unwrap();
                }).join();
                assert!(res.is_err());
            });
            let _t = thread::spawn(move|| {
                thread::spawn(move|| {
                    drop(tx);
                });
            });
        }
    }

    #[test]
    fn oneshot_multi_thread_send_recv_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = channel::<Box<isize>>();
            let _t = thread::spawn(move|| {
                tx.send(box 10).unwrap();
            });
            assert!(rx.recv().unwrap() == box 10);
        }
    }

    #[test]
    fn stream_send_recv_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = channel();

            send(tx, 0);
            recv(rx, 0);

            fn send(tx: Sender<Box<i32>>, i: i32) {
                if i == 10 { return }

                thread::spawn(move|| {
                    tx.send(box i).unwrap();
                    send(tx, i + 1);
                });
            }

            fn recv(rx: Receiver<Box<i32>>, i: i32) {
                if i == 10 { return }

                thread::spawn(move|| {
                    assert!(rx.recv().unwrap() == box i);
                    recv(rx, i + 1);
                });
            }
        }
    }

    #[test]
    fn recv_a_lot() {
        // Regression test that we don't run out of stack in scheduler context
        let (tx, rx) = channel();
        for _ in 0..10000 { tx.send(()).unwrap(); }
        for _ in 0..10000 { rx.recv().unwrap(); }
    }

    #[test]
    fn shared_chan_stress() {
        let (tx, rx) = channel();
        let total = stress_factor() + 100;
        for _ in 0..total {
            let tx = tx.clone();
            thread::spawn(move|| {
                tx.send(()).unwrap();
            });
        }

        for _ in 0..total {
            rx.recv().unwrap();
        }
    }

    #[test]
    fn test_nested_recv_iter() {
        let (tx, rx) = channel::<i32>();
        let (total_tx, total_rx) = channel::<i32>();

        let _t = thread::spawn(move|| {
            let mut acc = 0;
            for x in rx.iter() {
                acc += x;
            }
            total_tx.send(acc).unwrap();
        });

        tx.send(3).unwrap();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        assert_eq!(total_rx.recv().unwrap(), 6);
    }

    #[test]
    fn test_recv_iter_break() {
        let (tx, rx) = channel::<i32>();
        let (count_tx, count_rx) = channel();

        let _t = thread::spawn(move|| {
            let mut count = 0;
            for x in rx.iter() {
                if count >= 3 {
                    break;
                } else {
                    count += x;
                }
            }
            count_tx.send(count).unwrap();
        });

        tx.send(2).unwrap();
        tx.send(2).unwrap();
        tx.send(2).unwrap();
        let _ = tx.send(2);
        drop(tx);
        assert_eq!(count_rx.recv().unwrap(), 4);
    }

    #[test]
    fn test_recv_into_iter_owned() {
        let mut iter = {
          let (tx, rx) = channel::<i32>();
          tx.send(1).unwrap();
          tx.send(2).unwrap();

          rx.into_iter()
        };
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().is_none(), true);
    }

    #[test]
    fn test_recv_into_iter_borrowed() {
        let (tx, rx) = channel::<i32>();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        let mut iter = (&rx).into_iter();
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().is_none(), true);
    }

    #[test]
    fn try_recv_states() {
        let (tx1, rx1) = channel::<i32>();
        let (tx2, rx2) = channel::<()>();
        let (tx3, rx3) = channel::<()>();
        let _t = thread::spawn(move|| {
            rx2.recv().unwrap();
            tx1.send(1).unwrap();
            tx3.send(()).unwrap();
            rx2.recv().unwrap();
            drop(tx1);
            tx3.send(()).unwrap();
        });

        assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
        tx2.send(()).unwrap();
        rx3.recv().unwrap();
        assert_eq!(rx1.try_recv(), Ok(1));
        assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
        tx2.send(()).unwrap();
        rx3.recv().unwrap();
        assert_eq!(rx1.try_recv(), Err(TryRecvError::Disconnected));
    }

    // This bug used to end up in a livelock inside of the Receiver destructor
    // because the internal state of the Shared packet was corrupted
    #[test]
    fn destroy_upgraded_shared_port_when_sender_still_active() {
        let (tx, rx) = channel();
        let (tx2, rx2) = channel();
        let _t = thread::spawn(move|| {
            rx.recv().unwrap(); // wait on a oneshot
            drop(rx);  // destroy a shared
            tx2.send(()).unwrap();
        });
        // make sure the other thread has gone to sleep
        for _ in 0..5000 { thread::yield_now(); }

        // upgrade to a shared chan and send a message
        let t = tx.clone();
        drop(tx);
        t.send(()).unwrap();

        // wait for the child thread to exit before we exit
        rx2.recv().unwrap();
    }
}

#[cfg(test)]
mod sync_tests {
    use prelude::v1::*;

    use env;
    use thread;
    use super::*;

    pub fn stress_factor() -> usize {
        match env::var("RUST_TEST_STRESS") {
            Ok(val) => val.parse().unwrap(),
            Err(..) => 1,
        }
    }

    #[test]
    fn smoke() {
        let (tx, rx) = sync_channel::<i32>(1);
        tx.send(1).unwrap();
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn drop_full() {
        let (tx, _rx) = sync_channel::<Box<isize>>(1);
        tx.send(box 1).unwrap();
    }

    #[test]
    fn smoke_shared() {
        let (tx, rx) = sync_channel::<i32>(1);
        tx.send(1).unwrap();
        assert_eq!(rx.recv().unwrap(), 1);
        let tx = tx.clone();
        tx.send(1).unwrap();
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn smoke_threads() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move|| {
            tx.send(1).unwrap();
        });
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn smoke_port_gone() {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(rx);
        assert!(tx.send(1).is_err());
    }

    #[test]
    fn smoke_shared_port_gone2() {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(rx);
        let tx2 = tx.clone();
        drop(tx);
        assert!(tx2.send(1).is_err());
    }

    #[test]
    fn port_gone_concurrent() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move|| {
            rx.recv().unwrap();
        });
        while tx.send(1).is_ok() {}
    }

    #[test]
    fn port_gone_concurrent_shared() {
        let (tx, rx) = sync_channel::<i32>(0);
        let tx2 = tx.clone();
        let _t = thread::spawn(move|| {
            rx.recv().unwrap();
        });
        while tx.send(1).is_ok() && tx2.send(1).is_ok() {}
    }

    #[test]
    fn smoke_chan_gone() {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(tx);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn smoke_chan_gone_shared() {
        let (tx, rx) = sync_channel::<()>(0);
        let tx2 = tx.clone();
        drop(tx);
        drop(tx2);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn chan_gone_concurrent() {
        let (tx, rx) = sync_channel::<i32>(0);
        thread::spawn(move|| {
            tx.send(1).unwrap();
            tx.send(1).unwrap();
        });
        while rx.recv().is_ok() {}
    }

    #[test]
    fn stress() {
        let (tx, rx) = sync_channel::<i32>(0);
        thread::spawn(move|| {
            for _ in 0..10000 { tx.send(1).unwrap(); }
        });
        for _ in 0..10000 {
            assert_eq!(rx.recv().unwrap(), 1);
        }
    }

    #[test]
    fn stress_shared() {
        const AMT: u32 = 1000;
        const NTHREADS: u32 = 8;
        let (tx, rx) = sync_channel::<i32>(0);
        let (dtx, drx) = sync_channel::<()>(0);

        thread::spawn(move|| {
            for _ in 0..AMT * NTHREADS {
                assert_eq!(rx.recv().unwrap(), 1);
            }
            match rx.try_recv() {
                Ok(..) => panic!(),
                _ => {}
            }
            dtx.send(()).unwrap();
        });

        for _ in 0..NTHREADS {
            let tx = tx.clone();
            thread::spawn(move|| {
                for _ in 0..AMT { tx.send(1).unwrap(); }
            });
        }
        drop(tx);
        drx.recv().unwrap();
    }

    #[test]
    fn oneshot_single_thread_close_port_first() {
        // Simple test of closing without sending
        let (_tx, rx) = sync_channel::<i32>(0);
        drop(rx);
    }

    #[test]
    fn oneshot_single_thread_close_chan_first() {
        // Simple test of closing without sending
        let (tx, _rx) = sync_channel::<i32>(0);
        drop(tx);
    }

    #[test]
    fn oneshot_single_thread_send_port_close() {
        // Testing that the sender cleans up the payload if receiver is closed
        let (tx, rx) = sync_channel::<Box<i32>>(0);
        drop(rx);
        assert!(tx.send(box 0).is_err());
    }

    #[test]
    fn oneshot_single_thread_recv_chan_close() {
        // Receiving on a closed chan will panic
        let res = thread::spawn(move|| {
            let (tx, rx) = sync_channel::<i32>(0);
            drop(tx);
            rx.recv().unwrap();
        }).join();
        // What is our res?
        assert!(res.is_err());
    }

    #[test]
    fn oneshot_single_thread_send_then_recv() {
        let (tx, rx) = sync_channel::<Box<i32>>(1);
        tx.send(box 10).unwrap();
        assert!(rx.recv().unwrap() == box 10);
    }

    #[test]
    fn oneshot_single_thread_try_send_open() {
        let (tx, rx) = sync_channel::<i32>(1);
        assert_eq!(tx.try_send(10), Ok(()));
        assert!(rx.recv().unwrap() == 10);
    }

    #[test]
    fn oneshot_single_thread_try_send_closed() {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(rx);
        assert_eq!(tx.try_send(10), Err(TrySendError::Disconnected(10)));
    }

    #[test]
    fn oneshot_single_thread_try_send_closed2() {
        let (tx, _rx) = sync_channel::<i32>(0);
        assert_eq!(tx.try_send(10), Err(TrySendError::Full(10)));
    }

    #[test]
    fn oneshot_single_thread_try_recv_open() {
        let (tx, rx) = sync_channel::<i32>(1);
        tx.send(10).unwrap();
        assert!(rx.recv() == Ok(10));
    }

    #[test]
    fn oneshot_single_thread_try_recv_closed() {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(tx);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn oneshot_single_thread_peek_data() {
        let (tx, rx) = sync_channel::<i32>(1);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        tx.send(10).unwrap();
        assert_eq!(rx.try_recv(), Ok(10));
    }

    #[test]
    fn oneshot_single_thread_peek_close() {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(tx);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn oneshot_single_thread_peek_open() {
        let (_tx, rx) = sync_channel::<i32>(0);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn oneshot_multi_task_recv_then_send() {
        let (tx, rx) = sync_channel::<Box<i32>>(0);
        let _t = thread::spawn(move|| {
            assert!(rx.recv().unwrap() == box 10);
        });

        tx.send(box 10).unwrap();
    }

    #[test]
    fn oneshot_multi_task_recv_then_close() {
        let (tx, rx) = sync_channel::<Box<i32>>(0);
        let _t = thread::spawn(move|| {
            drop(tx);
        });
        let res = thread::spawn(move|| {
            assert!(rx.recv().unwrap() == box 10);
        }).join();
        assert!(res.is_err());
    }

    #[test]
    fn oneshot_multi_thread_close_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = sync_channel::<i32>(0);
            let _t = thread::spawn(move|| {
                drop(rx);
            });
            drop(tx);
        }
    }

    #[test]
    fn oneshot_multi_thread_send_close_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = sync_channel::<i32>(0);
            let _t = thread::spawn(move|| {
                drop(rx);
            });
            let _ = thread::spawn(move || {
                tx.send(1).unwrap();
            }).join();
        }
    }

    #[test]
    fn oneshot_multi_thread_recv_close_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = sync_channel::<i32>(0);
            let _t = thread::spawn(move|| {
                let res = thread::spawn(move|| {
                    rx.recv().unwrap();
                }).join();
                assert!(res.is_err());
            });
            let _t = thread::spawn(move|| {
                thread::spawn(move|| {
                    drop(tx);
                });
            });
        }
    }

    #[test]
    fn oneshot_multi_thread_send_recv_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = sync_channel::<Box<i32>>(0);
            let _t = thread::spawn(move|| {
                tx.send(box 10).unwrap();
            });
            assert!(rx.recv().unwrap() == box 10);
        }
    }

    #[test]
    fn stream_send_recv_stress() {
        for _ in 0..stress_factor() {
            let (tx, rx) = sync_channel::<Box<i32>>(0);

            send(tx, 0);
            recv(rx, 0);

            fn send(tx: SyncSender<Box<i32>>, i: i32) {
                if i == 10 { return }

                thread::spawn(move|| {
                    tx.send(box i).unwrap();
                    send(tx, i + 1);
                });
            }

            fn recv(rx: Receiver<Box<i32>>, i: i32) {
                if i == 10 { return }

                thread::spawn(move|| {
                    assert!(rx.recv().unwrap() == box i);
                    recv(rx, i + 1);
                });
            }
        }
    }

    #[test]
    fn recv_a_lot() {
        // Regression test that we don't run out of stack in scheduler context
        let (tx, rx) = sync_channel(10000);
        for _ in 0..10000 { tx.send(()).unwrap(); }
        for _ in 0..10000 { rx.recv().unwrap(); }
    }

    #[test]
    fn shared_chan_stress() {
        let (tx, rx) = sync_channel(0);
        let total = stress_factor() + 100;
        for _ in 0..total {
            let tx = tx.clone();
            thread::spawn(move|| {
                tx.send(()).unwrap();
            });
        }

        for _ in 0..total {
            rx.recv().unwrap();
        }
    }

    #[test]
    fn test_nested_recv_iter() {
        let (tx, rx) = sync_channel::<i32>(0);
        let (total_tx, total_rx) = sync_channel::<i32>(0);

        let _t = thread::spawn(move|| {
            let mut acc = 0;
            for x in rx.iter() {
                acc += x;
            }
            total_tx.send(acc).unwrap();
        });

        tx.send(3).unwrap();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        assert_eq!(total_rx.recv().unwrap(), 6);
    }

    #[test]
    fn test_recv_iter_break() {
        let (tx, rx) = sync_channel::<i32>(0);
        let (count_tx, count_rx) = sync_channel(0);

        let _t = thread::spawn(move|| {
            let mut count = 0;
            for x in rx.iter() {
                if count >= 3 {
                    break;
                } else {
                    count += x;
                }
            }
            count_tx.send(count).unwrap();
        });

        tx.send(2).unwrap();
        tx.send(2).unwrap();
        tx.send(2).unwrap();
        let _ = tx.try_send(2);
        drop(tx);
        assert_eq!(count_rx.recv().unwrap(), 4);
    }

    #[test]
    fn try_recv_states() {
        let (tx1, rx1) = sync_channel::<i32>(1);
        let (tx2, rx2) = sync_channel::<()>(1);
        let (tx3, rx3) = sync_channel::<()>(1);
        let _t = thread::spawn(move|| {
            rx2.recv().unwrap();
            tx1.send(1).unwrap();
            tx3.send(()).unwrap();
            rx2.recv().unwrap();
            drop(tx1);
            tx3.send(()).unwrap();
        });

        assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
        tx2.send(()).unwrap();
        rx3.recv().unwrap();
        assert_eq!(rx1.try_recv(), Ok(1));
        assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
        tx2.send(()).unwrap();
        rx3.recv().unwrap();
        assert_eq!(rx1.try_recv(), Err(TryRecvError::Disconnected));
    }

    // This bug used to end up in a livelock inside of the Receiver destructor
    // because the internal state of the Shared packet was corrupted
    #[test]
    fn destroy_upgraded_shared_port_when_sender_still_active() {
        let (tx, rx) = sync_channel::<()>(0);
        let (tx2, rx2) = sync_channel::<()>(0);
        let _t = thread::spawn(move|| {
            rx.recv().unwrap(); // wait on a oneshot
            drop(rx);  // destroy a shared
            tx2.send(()).unwrap();
        });
        // make sure the other thread has gone to sleep
        for _ in 0..5000 { thread::yield_now(); }

        // upgrade to a shared chan and send a message
        let t = tx.clone();
        drop(tx);
        t.send(()).unwrap();

        // wait for the child thread to exit before we exit
        rx2.recv().unwrap();
    }

    #[test]
    fn send1() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move|| { rx.recv().unwrap(); });
        assert_eq!(tx.send(1), Ok(()));
    }

    #[test]
    fn send2() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move|| { drop(rx); });
        assert!(tx.send(1).is_err());
    }

    #[test]
    fn send3() {
        let (tx, rx) = sync_channel::<i32>(1);
        assert_eq!(tx.send(1), Ok(()));
        let _t =thread::spawn(move|| { drop(rx); });
        assert!(tx.send(1).is_err());
    }

    #[test]
    fn send4() {
        let (tx, rx) = sync_channel::<i32>(0);
        let tx2 = tx.clone();
        let (done, donerx) = channel();
        let done2 = done.clone();
        let _t = thread::spawn(move|| {
            assert!(tx.send(1).is_err());
            done.send(()).unwrap();
        });
        let _t = thread::spawn(move|| {
            assert!(tx2.send(2).is_err());
            done2.send(()).unwrap();
        });
        drop(rx);
        donerx.recv().unwrap();
        donerx.recv().unwrap();
    }

    #[test]
    fn try_send1() {
        let (tx, _rx) = sync_channel::<i32>(0);
        assert_eq!(tx.try_send(1), Err(TrySendError::Full(1)));
    }

    #[test]
    fn try_send2() {
        let (tx, _rx) = sync_channel::<i32>(1);
        assert_eq!(tx.try_send(1), Ok(()));
        assert_eq!(tx.try_send(1), Err(TrySendError::Full(1)));
    }

    #[test]
    fn try_send3() {
        let (tx, rx) = sync_channel::<i32>(1);
        assert_eq!(tx.try_send(1), Ok(()));
        drop(rx);
        assert_eq!(tx.try_send(1), Err(TrySendError::Disconnected(1)));
    }

    #[test]
    fn issue_15761() {
        fn repro() {
            let (tx1, rx1) = sync_channel::<()>(3);
            let (tx2, rx2) = sync_channel::<()>(3);

            let _t = thread::spawn(move|| {
                rx1.recv().unwrap();
                tx2.try_send(()).unwrap();
            });

            tx1.try_send(()).unwrap();
            rx2.recv().unwrap();
        }

        for _ in 0..100 {
            repro()
        }
    }
}
