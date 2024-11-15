//! Multi-producer, single-consumer FIFO queue communication primitives.
//!
//! This module provides message-based communication over channels, concretely
//! defined among three types:
//!
//! * [`Sender`]
//! * [`SyncSender`]
//! * [`Receiver`]
//!
//! A [`Sender`] or [`SyncSender`] is used to send data to a [`Receiver`]. Both
//! senders are clone-able (multi-producer) such that many threads can send
//! simultaneously to one receiver (single-consumer).
//!
//! These channels come in two flavors:
//!
//! 1. An asynchronous, infinitely buffered channel. The [`channel`] function
//!    will return a `(Sender, Receiver)` tuple where all sends will be
//!    **asynchronous** (they never block). The channel conceptually has an
//!    infinite buffer.
//!
//! 2. A synchronous, bounded channel. The [`sync_channel`] function will
//!    return a `(SyncSender, Receiver)` tuple where the storage for pending
//!    messages is a pre-allocated buffer of a fixed size. All sends will be
//!    **synchronous** by blocking until there is buffer space available. Note
//!    that a bound of 0 is allowed, causing the channel to become a "rendezvous"
//!    channel where each sender atomically hands off a message to a receiver.
//!
//! [`send`]: Sender::send
//!
//! ## Disconnection
//!
//! The send and receive operations on channels will all return a [`Result`]
//! indicating whether the operation succeeded or not. An unsuccessful operation
//! is normally indicative of the other half of a channel having "hung up" by
//! being dropped in its corresponding thread.
//!
//! Once half of a channel has been deallocated, most operations can no longer
//! continue to make progress, so [`Err`] will be returned. Many applications
//! will continue to [`unwrap`] the results returned from this module,
//! instigating a propagation of failure among threads if one unexpectedly dies.
//!
//! [`unwrap`]: Result::unwrap
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
//! thread::spawn(move || {
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
//!     thread::spawn(move || {
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
//! thread::spawn(move || {
//!     // This will wait for the parent thread to start receiving
//!     tx.send(53).unwrap();
//! });
//! rx.recv().unwrap();
//! ```
//!
//! Unbounded receive loop:
//!
//! ```
//! use std::sync::mpsc::sync_channel;
//! use std::thread;
//!
//! let (tx, rx) = sync_channel(3);
//!
//! for _ in 0..3 {
//!     // It would be the same without thread and clone here
//!     // since there will still be one `tx` left.
//!     let tx = tx.clone();
//!     // cloned tx dropped within thread
//!     thread::spawn(move || tx.send("ok").unwrap());
//! }
//!
//! // Drop the last sender to stop `rx` waiting for message.
//! // The program will not complete if we comment this out.
//! // **All** `tx` needs to be dropped for `rx` to have `Err`.
//! drop(tx);
//!
//! // Unbounded receiver waiting for all senders to complete.
//! while let Ok(msg) = rx.recv() {
//!     println!("{msg}");
//! }
//!
//! println!("completed");
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(all(test, not(any(target_os = "emscripten", target_os = "wasi"))))]
mod tests;

#[cfg(all(test, not(any(target_os = "emscripten", target_os = "wasi"))))]
mod sync_tests;

// MPSC channels are built as a wrapper around MPMC channels, which
// were ported from the `crossbeam-channel` crate. MPMC channels are
// not exposed publicly, but if you are curious about the implementation,
// that's where everything is.

use crate::sync::mpmc;
use crate::time::{Duration, Instant};
use crate::{error, fmt};

/// The receiving half of Rust's [`channel`] (or [`sync_channel`]) type.
/// This half can only be owned by one thread.
///
/// Messages sent to the channel can be retrieved using [`recv`].
///
/// [`recv`]: Receiver::recv
///
/// # Examples
///
/// ```rust
/// use std::sync::mpsc::channel;
/// use std::thread;
/// use std::time::Duration;
///
/// let (send, recv) = channel();
///
/// thread::spawn(move || {
///     send.send("Hello world!").unwrap();
///     thread::sleep(Duration::from_secs(2)); // block for two seconds
///     send.send("Delayed for 2 seconds").unwrap();
/// });
///
/// println!("{}", recv.recv().unwrap()); // Received immediately
/// println!("Waiting...");
/// println!("{}", recv.recv().unwrap()); // Received after 2 seconds
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Receiver")]
pub struct Receiver<T> {
    inner: mpmc::Receiver<T>,
}

// The receiver port can be sent from place to place, so long as it
// is not used to receive non-sendable things.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for Receiver<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> !Sync for Receiver<T> {}

/// An iterator over messages on a [`Receiver`], created by [`iter`].
///
/// This iterator will block whenever [`next`] is called,
/// waiting for a new message, and [`None`] will be returned
/// when the corresponding channel has hung up.
///
/// [`iter`]: Receiver::iter
/// [`next`]: Iterator::next
///
/// # Examples
///
/// ```rust
/// use std::sync::mpsc::channel;
/// use std::thread;
///
/// let (send, recv) = channel();
///
/// thread::spawn(move || {
///     send.send(1u8).unwrap();
///     send.send(2u8).unwrap();
///     send.send(3u8).unwrap();
/// });
///
/// for x in recv.iter() {
///     println!("Got: {x}");
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct Iter<'a, T: 'a> {
    rx: &'a Receiver<T>,
}

/// An iterator that attempts to yield all pending values for a [`Receiver`],
/// created by [`try_iter`].
///
/// [`None`] will be returned when there are no pending values remaining or
/// if the corresponding channel has hung up.
///
/// This iterator will never block the caller in order to wait for data to
/// become available. Instead, it will return [`None`].
///
/// [`try_iter`]: Receiver::try_iter
///
/// # Examples
///
/// ```rust
/// use std::sync::mpsc::channel;
/// use std::thread;
/// use std::time::Duration;
///
/// let (sender, receiver) = channel();
///
/// // Nothing is in the buffer yet
/// assert!(receiver.try_iter().next().is_none());
/// println!("Nothing in the buffer...");
///
/// thread::spawn(move || {
///     sender.send(1).unwrap();
///     sender.send(2).unwrap();
///     sender.send(3).unwrap();
/// });
///
/// println!("Going to sleep...");
/// thread::sleep(Duration::from_secs(2)); // block for two seconds
///
/// for x in receiver.try_iter() {
///     println!("Got: {x}");
/// }
/// ```
#[stable(feature = "receiver_try_iter", since = "1.15.0")]
#[derive(Debug)]
pub struct TryIter<'a, T: 'a> {
    rx: &'a Receiver<T>,
}

/// An owning iterator over messages on a [`Receiver`],
/// created by [`into_iter`].
///
/// This iterator will block whenever [`next`]
/// is called, waiting for a new message, and [`None`] will be
/// returned if the corresponding channel has hung up.
///
/// [`into_iter`]: Receiver::into_iter
/// [`next`]: Iterator::next
///
/// # Examples
///
/// ```rust
/// use std::sync::mpsc::channel;
/// use std::thread;
///
/// let (send, recv) = channel();
///
/// thread::spawn(move || {
///     send.send(1u8).unwrap();
///     send.send(2u8).unwrap();
///     send.send(3u8).unwrap();
/// });
///
/// for x in recv.into_iter() {
///     println!("Got: {x}");
/// }
/// ```
#[stable(feature = "receiver_into_iter", since = "1.1.0")]
#[derive(Debug)]
pub struct IntoIter<T> {
    rx: Receiver<T>,
}

/// The sending-half of Rust's asynchronous [`channel`] type.
///
/// Messages can be sent through this channel with [`send`].
///
/// Note: all senders (the original and its clones) need to be dropped for the receiver
/// to stop blocking to receive messages with [`Receiver::recv`].
///
/// [`send`]: Sender::send
///
/// # Examples
///
/// ```rust
/// use std::sync::mpsc::channel;
/// use std::thread;
///
/// let (sender, receiver) = channel();
/// let sender2 = sender.clone();
///
/// // First thread owns sender
/// thread::spawn(move || {
///     sender.send(1).unwrap();
/// });
///
/// // Second thread owns sender2
/// thread::spawn(move || {
///     sender2.send(2).unwrap();
/// });
///
/// let msg = receiver.recv().unwrap();
/// let msg2 = receiver.recv().unwrap();
///
/// assert_eq!(3, msg + msg2);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Sender<T> {
    inner: mpmc::Sender<T>,
}

// The send port can be sent from place to place, so long as it
// is not used to send non-sendable things.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for Sender<T> {}

#[stable(feature = "mpsc_sender_sync", since = "1.72.0")]
unsafe impl<T: Send> Sync for Sender<T> {}

/// The sending-half of Rust's synchronous [`sync_channel`] type.
///
/// Messages can be sent through this channel with [`send`] or [`try_send`].
///
/// [`send`] will block if there is no space in the internal buffer.
///
/// [`send`]: SyncSender::send
/// [`try_send`]: SyncSender::try_send
///
/// # Examples
///
/// ```rust
/// use std::sync::mpsc::sync_channel;
/// use std::thread;
///
/// // Create a sync_channel with buffer size 2
/// let (sync_sender, receiver) = sync_channel(2);
/// let sync_sender2 = sync_sender.clone();
///
/// // First thread owns sync_sender
/// thread::spawn(move || {
///     sync_sender.send(1).unwrap();
///     sync_sender.send(2).unwrap();
/// });
///
/// // Second thread owns sync_sender2
/// thread::spawn(move || {
///     sync_sender2.send(3).unwrap();
///     // thread will now block since the buffer is full
///     println!("Thread unblocked!");
/// });
///
/// let mut msg;
///
/// msg = receiver.recv().unwrap();
/// println!("message {msg} received");
///
/// // "Thread unblocked!" will be printed now
///
/// msg = receiver.recv().unwrap();
/// println!("message {msg} received");
///
/// msg = receiver.recv().unwrap();
///
/// println!("message {msg} received");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SyncSender<T> {
    inner: mpmc::Sender<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for SyncSender<T> {}

/// An error returned from the [`Sender::send`] or [`SyncSender::send`]
/// function on **channel**s.
///
/// A **send** operation can only fail if the receiving end of a channel is
/// disconnected, implying that the data could never be received. The error
/// contains the data being sent as a payload so it can be recovered.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct SendError<T>(#[stable(feature = "rust1", since = "1.0.0")] pub T);

/// An error returned from the [`recv`] function on a [`Receiver`].
///
/// The [`recv`] operation can only fail if the sending half of a
/// [`channel`] (or [`sync_channel`]) is disconnected, implying that no further
/// messages will ever be received.
///
/// [`recv`]: Receiver::recv
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RecvError;

/// This enumeration is the list of the possible reasons that [`try_recv`] could
/// not return data when called. This can occur with both a [`channel`] and
/// a [`sync_channel`].
///
/// [`try_recv`]: Receiver::try_recv
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum TryRecvError {
    /// This **channel** is currently empty, but the **Sender**(s) have not yet
    /// disconnected, so data may yet become available.
    #[stable(feature = "rust1", since = "1.0.0")]
    Empty,

    /// The **channel**'s sending half has become disconnected, and there will
    /// never be any more data received on it.
    #[stable(feature = "rust1", since = "1.0.0")]
    Disconnected,
}

/// This enumeration is the list of possible errors that made [`recv_timeout`]
/// unable to return data when called. This can occur with both a [`channel`] and
/// a [`sync_channel`].
///
/// [`recv_timeout`]: Receiver::recv_timeout
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "mpsc_recv_timeout", since = "1.12.0")]
pub enum RecvTimeoutError {
    /// This **channel** is currently empty, but the **Sender**(s) have not yet
    /// disconnected, so data may yet become available.
    #[stable(feature = "mpsc_recv_timeout", since = "1.12.0")]
    Timeout,
    /// The **channel**'s sending half has become disconnected, and there will
    /// never be any more data received on it.
    #[stable(feature = "mpsc_recv_timeout", since = "1.12.0")]
    Disconnected,
}

/// This enumeration is the list of the possible error outcomes for the
/// [`try_send`] method.
///
/// [`try_send`]: SyncSender::try_send
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TrySendError<T> {
    /// The data could not be sent on the [`sync_channel`] because it would require that
    /// the callee block to send the data.
    ///
    /// If this is a buffered channel, then the buffer is full at this time. If
    /// this is not a buffered channel, then there is no [`Receiver`] available to
    /// acquire the data.
    #[stable(feature = "rust1", since = "1.0.0")]
    Full(#[stable(feature = "rust1", since = "1.0.0")] T),

    /// This [`sync_channel`]'s receiving half has disconnected, so the data could not be
    /// sent. The data is returned back to the callee in this case.
    #[stable(feature = "rust1", since = "1.0.0")]
    Disconnected(#[stable(feature = "rust1", since = "1.0.0")] T),
}

/// Creates a new asynchronous channel, returning the sender/receiver halves.
///
/// All data sent on the [`Sender`] will become available on the [`Receiver`] in
/// the same order as it was sent, and no [`send`] will block the calling thread
/// (this channel has an "infinite buffer", unlike [`sync_channel`], which will
/// block after its buffer limit is reached). [`recv`] will block until a message
/// is available while there is at least one [`Sender`] alive (including clones).
///
/// The [`Sender`] can be cloned to [`send`] to the same channel multiple times, but
/// only one [`Receiver`] is supported.
///
/// If the [`Receiver`] is disconnected while trying to [`send`] with the
/// [`Sender`], the [`send`] method will return a [`SendError`]. Similarly, if the
/// [`Sender`] is disconnected while trying to [`recv`], the [`recv`] method will
/// return a [`RecvError`].
///
/// [`send`]: Sender::send
/// [`recv`]: Receiver::recv
///
/// # Examples
///
/// ```
/// use std::sync::mpsc::channel;
/// use std::thread;
///
/// let (sender, receiver) = channel();
///
/// // Spawn off an expensive computation
/// thread::spawn(move || {
/// #   fn expensive_computation() {}
///     sender.send(expensive_computation()).unwrap();
/// });
///
/// // Do some useful work for awhile
///
/// // Let's see what that answer was
/// println!("{:?}", receiver.recv().unwrap());
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = mpmc::channel();
    (Sender { inner: tx }, Receiver { inner: rx })
}

/// Creates a new synchronous, bounded channel.
///
/// All data sent on the [`SyncSender`] will become available on the [`Receiver`]
/// in the same order as it was sent. Like asynchronous [`channel`]s, the
/// [`Receiver`] will block until a message becomes available. `sync_channel`
/// differs greatly in the semantics of the sender, however.
///
/// This channel has an internal buffer on which messages will be queued.
/// `bound` specifies the buffer size. When the internal buffer becomes full,
/// future sends will *block* waiting for the buffer to open up. Note that a
/// buffer size of 0 is valid, in which case this becomes "rendezvous channel"
/// where each [`send`] will not return until a [`recv`] is paired with it.
///
/// The [`SyncSender`] can be cloned to [`send`] to the same channel multiple
/// times, but only one [`Receiver`] is supported.
///
/// Like asynchronous channels, if the [`Receiver`] is disconnected while trying
/// to [`send`] with the [`SyncSender`], the [`send`] method will return a
/// [`SendError`]. Similarly, If the [`SyncSender`] is disconnected while trying
/// to [`recv`], the [`recv`] method will return a [`RecvError`].
///
/// [`send`]: SyncSender::send
/// [`recv`]: Receiver::recv
///
/// # Examples
///
/// ```
/// use std::sync::mpsc::sync_channel;
/// use std::thread;
///
/// let (sender, receiver) = sync_channel(1);
///
/// // this returns immediately
/// sender.send(1).unwrap();
///
/// thread::spawn(move || {
///     // this will block until the previous message has been received
///     sender.send(2).unwrap();
/// });
///
/// assert_eq!(receiver.recv().unwrap(), 1);
/// assert_eq!(receiver.recv().unwrap(), 2);
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn sync_channel<T>(bound: usize) -> (SyncSender<T>, Receiver<T>) {
    let (tx, rx) = mpmc::sync_channel(bound);
    (SyncSender { inner: tx }, Receiver { inner: rx })
}

////////////////////////////////////////////////////////////////////////////////
// Sender
////////////////////////////////////////////////////////////////////////////////

impl<T> Sender<T> {
    /// Attempts to send a value on this channel, returning it back if it could
    /// not be sent.
    ///
    /// A successful send occurs when it is determined that the other end of
    /// the channel has not hung up already. An unsuccessful send would be one
    /// where the corresponding receiver has already been deallocated. Note
    /// that a return value of [`Err`] means that the data will never be
    /// received, but a return value of [`Ok`] does *not* mean that the data
    /// will be received. It is possible for the corresponding receiver to
    /// hang up immediately after this function returns [`Ok`].
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
    /// assert_eq!(tx.send(1).unwrap_err().0, 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send(&self, t: T) -> Result<(), SendError<T>> {
        self.inner.send(t)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Sender<T> {
    /// Clone a sender to send to other threads.
    ///
    /// Note, be aware of the lifetime of the sender because all senders
    /// (including the original) need to be dropped in order for
    /// [`Receiver::recv`] to stop blocking.
    fn clone(&self) -> Sender<T> {
        Sender { inner: self.inner.clone() }
    }
}

#[stable(feature = "mpsc_debug", since = "1.8.0")]
impl<T> fmt::Debug for Sender<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sender").finish_non_exhaustive()
    }
}

////////////////////////////////////////////////////////////////////////////////
// SyncSender
////////////////////////////////////////////////////////////////////////////////

impl<T> SyncSender<T> {
    /// Sends a value on this synchronous channel.
    ///
    /// This function will *block* until space in the internal buffer becomes
    /// available or a receiver is available to hand off the message to.
    ///
    /// Note that a successful send does *not* guarantee that the receiver will
    /// ever see the data if there is a buffer on this channel. Items may be
    /// enqueued in the internal buffer for the receiver to receive at a later
    /// time. If the buffer size is 0, however, the channel becomes a rendezvous
    /// channel and it guarantees that the receiver has indeed received
    /// the data if this function returns success.
    ///
    /// This function will never panic, but it may return [`Err`] if the
    /// [`Receiver`] has disconnected and is no longer able to receive
    /// information.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::mpsc::sync_channel;
    /// use std::thread;
    ///
    /// // Create a rendezvous sync_channel with buffer size 0
    /// let (sync_sender, receiver) = sync_channel(0);
    ///
    /// thread::spawn(move || {
    ///    println!("sending message...");
    ///    sync_sender.send(1).unwrap();
    ///    // Thread is now blocked until the message is received
    ///
    ///    println!("...message received!");
    /// });
    ///
    /// let msg = receiver.recv().unwrap();
    /// assert_eq!(1, msg);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send(&self, t: T) -> Result<(), SendError<T>> {
        self.inner.send(t)
    }

    /// Attempts to send a value on this channel without blocking.
    ///
    /// This method differs from [`send`] by returning immediately if the
    /// channel's buffer is full or no receiver is waiting to acquire some
    /// data. Compared with [`send`], this function has two failure cases
    /// instead of one (one for disconnection, one for a full buffer).
    ///
    /// See [`send`] for notes about guarantees of whether the
    /// receiver has received the data or not if this function is successful.
    ///
    /// [`send`]: Self::send
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::mpsc::sync_channel;
    /// use std::thread;
    ///
    /// // Create a sync_channel with buffer size 1
    /// let (sync_sender, receiver) = sync_channel(1);
    /// let sync_sender2 = sync_sender.clone();
    ///
    /// // First thread owns sync_sender
    /// thread::spawn(move || {
    ///     sync_sender.send(1).unwrap();
    ///     sync_sender.send(2).unwrap();
    ///     // Thread blocked
    /// });
    ///
    /// // Second thread owns sync_sender2
    /// thread::spawn(move || {
    ///     // This will return an error and send
    ///     // no message if the buffer is full
    ///     let _ = sync_sender2.try_send(3);
    /// });
    ///
    /// let mut msg;
    /// msg = receiver.recv().unwrap();
    /// println!("message {msg} received");
    ///
    /// msg = receiver.recv().unwrap();
    /// println!("message {msg} received");
    ///
    /// // Third message may have never been sent
    /// match receiver.try_recv() {
    ///     Ok(msg) => println!("message {msg} received"),
    ///     Err(_) => println!("the third message was never sent"),
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_send(&self, t: T) -> Result<(), TrySendError<T>> {
        self.inner.try_send(t)
    }

    // Attempts to send for a value on this receiver, returning an error if the
    // corresponding channel has hung up, or if it waits more than `timeout`.
    //
    // This method is currently private and only used for tests.
    #[allow(unused)]
    fn send_timeout(&self, t: T, timeout: Duration) -> Result<(), mpmc::SendTimeoutError<T>> {
        self.inner.send_timeout(t, timeout)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for SyncSender<T> {
    fn clone(&self) -> SyncSender<T> {
        SyncSender { inner: self.inner.clone() }
    }
}

#[stable(feature = "mpsc_debug", since = "1.8.0")]
impl<T> fmt::Debug for SyncSender<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyncSender").finish_non_exhaustive()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Receiver
////////////////////////////////////////////////////////////////////////////////

impl<T> Receiver<T> {
    /// Attempts to return a pending value on this receiver without blocking.
    ///
    /// This method will never block the caller in order to wait for data to
    /// become available. Instead, this will always return immediately with a
    /// possible option of pending data on the channel.
    ///
    /// This is useful for a flavor of "optimistic check" before deciding to
    /// block on a receiver.
    ///
    /// Compared with [`recv`], this function has two failure cases instead of one
    /// (one for disconnection, one for an empty buffer).
    ///
    /// [`recv`]: Self::recv
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::mpsc::{Receiver, channel};
    ///
    /// let (_, receiver): (_, Receiver<i32>) = channel();
    ///
    /// assert!(receiver.try_recv().is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.inner.try_recv()
    }

    /// Attempts to wait for a value on this receiver, returning an error if the
    /// corresponding channel has hung up.
    ///
    /// This function will always block the current thread if there is no data
    /// available and it's possible for more data to be sent (at least one sender
    /// still exists). Once a message is sent to the corresponding [`Sender`]
    /// (or [`SyncSender`]), this receiver will wake up and return that
    /// message.
    ///
    /// If the corresponding [`Sender`] has disconnected, or it disconnects while
    /// this call is blocking, this call will wake up and return [`Err`] to
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
        self.inner.recv()
    }

    /// Attempts to wait for a value on this receiver, returning an error if the
    /// corresponding channel has hung up, or if it waits more than `timeout`.
    ///
    /// This function will always block the current thread if there is no data
    /// available and it's possible for more data to be sent (at least one sender
    /// still exists). Once a message is sent to the corresponding [`Sender`]
    /// (or [`SyncSender`]), this receiver will wake up and return that
    /// message.
    ///
    /// If the corresponding [`Sender`] has disconnected, or it disconnects while
    /// this call is blocking, this call will wake up and return [`Err`] to
    /// indicate that no more messages can ever be received on this channel.
    /// However, since channels are buffered, messages sent before the disconnect
    /// will still be properly received.
    ///
    /// # Examples
    ///
    /// Successfully receiving value before encountering timeout:
    ///
    /// ```no_run
    /// use std::thread;
    /// use std::time::Duration;
    /// use std::sync::mpsc;
    ///
    /// let (send, recv) = mpsc::channel();
    ///
    /// thread::spawn(move || {
    ///     send.send('a').unwrap();
    /// });
    ///
    /// assert_eq!(
    ///     recv.recv_timeout(Duration::from_millis(400)),
    ///     Ok('a')
    /// );
    /// ```
    ///
    /// Receiving an error upon reaching timeout:
    ///
    /// ```no_run
    /// use std::thread;
    /// use std::time::Duration;
    /// use std::sync::mpsc;
    ///
    /// let (send, recv) = mpsc::channel();
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_millis(800));
    ///     send.send('a').unwrap();
    /// });
    ///
    /// assert_eq!(
    ///     recv.recv_timeout(Duration::from_millis(400)),
    ///     Err(mpsc::RecvTimeoutError::Timeout)
    /// );
    /// ```
    #[stable(feature = "mpsc_recv_timeout", since = "1.12.0")]
    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvTimeoutError> {
        self.inner.recv_timeout(timeout)
    }

    /// Attempts to wait for a value on this receiver, returning an error if the
    /// corresponding channel has hung up, or if `deadline` is reached.
    ///
    /// This function will always block the current thread if there is no data
    /// available and it's possible for more data to be sent. Once a message is
    /// sent to the corresponding [`Sender`] (or [`SyncSender`]), then this
    /// receiver will wake up and return that message.
    ///
    /// If the corresponding [`Sender`] has disconnected, or it disconnects while
    /// this call is blocking, this call will wake up and return [`Err`] to
    /// indicate that no more messages can ever be received on this channel.
    /// However, since channels are buffered, messages sent before the disconnect
    /// will still be properly received.
    ///
    /// # Examples
    ///
    /// Successfully receiving value before reaching deadline:
    ///
    /// ```no_run
    /// #![feature(deadline_api)]
    /// use std::thread;
    /// use std::time::{Duration, Instant};
    /// use std::sync::mpsc;
    ///
    /// let (send, recv) = mpsc::channel();
    ///
    /// thread::spawn(move || {
    ///     send.send('a').unwrap();
    /// });
    ///
    /// assert_eq!(
    ///     recv.recv_deadline(Instant::now() + Duration::from_millis(400)),
    ///     Ok('a')
    /// );
    /// ```
    ///
    /// Receiving an error upon reaching deadline:
    ///
    /// ```no_run
    /// #![feature(deadline_api)]
    /// use std::thread;
    /// use std::time::{Duration, Instant};
    /// use std::sync::mpsc;
    ///
    /// let (send, recv) = mpsc::channel();
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_millis(800));
    ///     send.send('a').unwrap();
    /// });
    ///
    /// assert_eq!(
    ///     recv.recv_deadline(Instant::now() + Duration::from_millis(400)),
    ///     Err(mpsc::RecvTimeoutError::Timeout)
    /// );
    /// ```
    #[unstable(feature = "deadline_api", issue = "46316")]
    pub fn recv_deadline(&self, deadline: Instant) -> Result<T, RecvTimeoutError> {
        self.inner.recv_deadline(deadline)
    }

    /// Returns an iterator that will block waiting for messages, but never
    /// [`panic!`]. It will return [`None`] when the channel has hung up.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::mpsc::channel;
    /// use std::thread;
    ///
    /// let (send, recv) = channel();
    ///
    /// thread::spawn(move || {
    ///     send.send(1).unwrap();
    ///     send.send(2).unwrap();
    ///     send.send(3).unwrap();
    /// });
    ///
    /// let mut iter = recv.iter();
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { rx: self }
    }

    /// Returns an iterator that will attempt to yield all pending values.
    /// It will return `None` if there are no more pending values or if the
    /// channel has hung up. The iterator will never [`panic!`] or block the
    /// user by waiting for values.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::mpsc::channel;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let (sender, receiver) = channel();
    ///
    /// // nothing is in the buffer yet
    /// assert!(receiver.try_iter().next().is_none());
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_secs(1));
    ///     sender.send(1).unwrap();
    ///     sender.send(2).unwrap();
    ///     sender.send(3).unwrap();
    /// });
    ///
    /// // nothing is in the buffer yet
    /// assert!(receiver.try_iter().next().is_none());
    ///
    /// // block for two seconds
    /// thread::sleep(Duration::from_secs(2));
    ///
    /// let mut iter = receiver.try_iter();
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[stable(feature = "receiver_try_iter", since = "1.15.0")]
    pub fn try_iter(&self) -> TryIter<'_, T> {
        TryIter { rx: self }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.rx.recv().ok()
    }
}

#[stable(feature = "receiver_try_iter", since = "1.15.0")]
impl<'a, T> Iterator for TryIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.rx.try_recv().ok()
    }
}

#[stable(feature = "receiver_into_iter", since = "1.1.0")]
impl<'a, T> IntoIterator for &'a Receiver<T> {
    type Item = T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "receiver_into_iter", since = "1.1.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.rx.recv().ok()
    }
}

#[stable(feature = "receiver_into_iter", since = "1.1.0")]
impl<T> IntoIterator for Receiver<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter { rx: self }
    }
}

#[stable(feature = "mpsc_debug", since = "1.8.0")]
impl<T> fmt::Debug for Receiver<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receiver").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Debug for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SendError").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "sending on a closed channel".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> error::Error for SendError<T> {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "sending on a closed channel"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Debug for TrySendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TrySendError::Full(..) => "Full(..)".fmt(f),
            TrySendError::Disconnected(..) => "Disconnected(..)".fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Display for TrySendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TrySendError::Full(..) => "sending on a full channel".fmt(f),
            TrySendError::Disconnected(..) => "sending on a closed channel".fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> error::Error for TrySendError<T> {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match *self {
            TrySendError::Full(..) => "sending on a full channel",
            TrySendError::Disconnected(..) => "sending on a closed channel",
        }
    }
}

#[stable(feature = "mpsc_error_conversions", since = "1.24.0")]
impl<T> From<SendError<T>> for TrySendError<T> {
    /// Converts a `SendError<T>` into a `TrySendError<T>`.
    ///
    /// This conversion always returns a `TrySendError::Disconnected` containing the data in the `SendError<T>`.
    ///
    /// No data is allocated on the heap.
    fn from(err: SendError<T>) -> TrySendError<T> {
        match err {
            SendError(t) => TrySendError::Disconnected(t),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for RecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "receiving on a closed channel".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for RecvError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "receiving on a closed channel"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for TryRecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TryRecvError::Empty => "receiving on an empty channel".fmt(f),
            TryRecvError::Disconnected => "receiving on a closed channel".fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for TryRecvError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match *self {
            TryRecvError::Empty => "receiving on an empty channel",
            TryRecvError::Disconnected => "receiving on a closed channel",
        }
    }
}

#[stable(feature = "mpsc_error_conversions", since = "1.24.0")]
impl From<RecvError> for TryRecvError {
    /// Converts a `RecvError` into a `TryRecvError`.
    ///
    /// This conversion always returns `TryRecvError::Disconnected`.
    ///
    /// No data is allocated on the heap.
    fn from(err: RecvError) -> TryRecvError {
        match err {
            RecvError => TryRecvError::Disconnected,
        }
    }
}

#[stable(feature = "mpsc_recv_timeout_error", since = "1.15.0")]
impl fmt::Display for RecvTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            RecvTimeoutError::Timeout => "timed out waiting on channel".fmt(f),
            RecvTimeoutError::Disconnected => "channel is empty and sending half is closed".fmt(f),
        }
    }
}

#[stable(feature = "mpsc_recv_timeout_error", since = "1.15.0")]
impl error::Error for RecvTimeoutError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match *self {
            RecvTimeoutError::Timeout => "timed out waiting on channel",
            RecvTimeoutError::Disconnected => "channel is empty and sending half is closed",
        }
    }
}

#[stable(feature = "mpsc_error_conversions", since = "1.24.0")]
impl From<RecvError> for RecvTimeoutError {
    /// Converts a `RecvError` into a `RecvTimeoutError`.
    ///
    /// This conversion always returns `RecvTimeoutError::Disconnected`.
    ///
    /// No data is allocated on the heap.
    fn from(err: RecvError) -> RecvTimeoutError {
        match err {
            RecvError => RecvTimeoutError::Disconnected,
        }
    }
}
