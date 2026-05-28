//! A single-producer, single-consumer (oneshot) channel.
//!
//! This is an experimental module, so the API will likely change.

use crate::sync::mpmc;
use crate::sync::mpsc::{RecvError, SendError};
use crate::time::{Duration, Instant};
use crate::{error, fmt};

/// Creates a new oneshot channel, returning the sender/receiver halves.
///
/// # Examples
///
/// ```
/// #![feature(oneshot_channel)]
/// use std::sync::oneshot;
/// use std::thread;
///
/// let (sender, receiver) = oneshot::channel();
///
/// // Spawn off an expensive computation.
/// thread::spawn(move || {
/// #   fn expensive_computation() -> i32 { 42 }
///     sender.send(expensive_computation()).unwrap();
///     // `sender` is consumed by `send`, so we cannot use it anymore.
/// });
///
/// # fn do_other_work() -> i32 { 42 }
/// do_other_work();
///
/// // Let's see what that answer was...
/// println!("{:?}", receiver.recv().unwrap());
/// // `receiver` is consumed by `recv`, so we cannot use it anymore.
/// ```
#[must_use]
#[unstable(feature = "oneshot_channel", issue = "143674")]
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    // Using a `sync_channel` with capacity 1 means that the internal implementation will use the
    // `Array`-flavored channel implementation.
    let (sender, receiver) = mpmc::sync_channel(1);
    (Sender { inner: sender }, Receiver { inner: receiver })
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sender
////////////////////////////////////////////////////////////////////////////////////////////////////

/// The sending half of a oneshot channel.
///
/// # Examples
///
/// ```
/// #![feature(oneshot_channel)]
/// use std::sync::oneshot;
/// use std::thread;
///
/// let (sender, receiver) = oneshot::channel();
///
/// thread::spawn(move || {
///     sender.send("Hello from thread!").unwrap();
/// });
///
/// assert_eq!(receiver.recv().unwrap(), "Hello from thread!");
/// ```
///
/// `Sender` cannot be sent between threads if it is sending non-`Send` types.
///
/// ```compile_fail
/// #![feature(oneshot_channel)]
/// use std::sync::oneshot;
/// use std::thread;
/// use std::ptr;
///
/// let (sender, receiver) = oneshot::channel();
///
/// struct NotSend(*mut ());
/// thread::spawn(move || {
///     sender.send(NotSend(ptr::null_mut()));
/// });
///
/// let reply = receiver.try_recv().unwrap();
/// ```
#[unstable(feature = "oneshot_channel", issue = "143674")]
pub struct Sender<T> {
    /// The `oneshot` channel is simply a wrapper around a `mpmc` channel.
    inner: mpmc::Sender<T>,
}

// SAFETY: Since the only methods in which synchronization must occur take full ownership of the
// [`Sender`], it is perfectly safe to share a `&Sender` between threads (as it is effectively
// useless without ownership).
#[unstable(feature = "oneshot_channel", issue = "143674")]
unsafe impl<T> Sync for Sender<T> {}

impl<T> Sender<T> {
    /// Attempts to send a value through this channel. This can only fail if the corresponding
    /// [`Receiver<T>`] has been dropped.
    ///
    /// This method is non-blocking (wait-free).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(oneshot_channel)]
    /// use std::sync::oneshot;
    /// use std::thread;
    ///
    /// let (tx, rx) = oneshot::channel();
    ///
    /// thread::spawn(move || {
    ///     // Perform some computation.
    ///     let result = 2 + 2;
    ///     tx.send(result).unwrap();
    /// });
    ///
    /// assert_eq!(rx.recv().unwrap(), 4);
    /// ```
    #[unstable(feature = "oneshot_channel", issue = "143674")]
    pub fn send(self, t: T) -> Result<(), SendError<T>> {
        self.inner.send(t)
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> fmt::Debug for Sender<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sender").finish_non_exhaustive()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Receiver
////////////////////////////////////////////////////////////////////////////////////////////////////

/// The receiving half of a oneshot channel.
///
/// # Examples
///
/// ```
/// #![feature(oneshot_channel)]
/// use std::sync::oneshot;
/// use std::thread;
/// use std::time::Duration;
///
/// let (sender, receiver) = oneshot::channel();
///
/// thread::spawn(move || {
///     thread::sleep(Duration::from_millis(100));
///     sender.send("Hello after delay!").unwrap();
/// });
///
/// println!("Waiting for message...");
/// println!("{}", receiver.recv().unwrap());
/// ```
///
/// `Receiver` cannot be sent between threads if it is receiving non-`Send` types.
///
/// ```compile_fail
/// # #![feature(oneshot_channel)]
/// # use std::sync::oneshot;
/// # use std::thread;
/// # use std::ptr;
/// #
/// let (sender, receiver) = oneshot::channel();
///
/// struct NotSend(*mut ());
/// sender.send(NotSend(ptr::null_mut()));
///
/// thread::spawn(move || {
///     let reply = receiver.try_recv().unwrap();
/// });
/// ```
#[unstable(feature = "oneshot_channel", issue = "143674")]
pub struct Receiver<T> {
    /// The `oneshot` channel is simply a wrapper around a `mpmc` channel.
    inner: mpmc::Receiver<T>,
}

// SAFETY: Since the only methods in which synchronization must occur take full ownership of the
// [`Receiver`], it is perfectly safe to share a `&Receiver` between threads (as it is unable to
// receive any values without ownership).
#[unstable(feature = "oneshot_channel", issue = "143674")]
unsafe impl<T> Sync for Receiver<T> {}

impl<T> Receiver<T> {
    /// Receives the value from the sending end, blocking the calling thread until it gets it.
    ///
    /// Can only fail if the corresponding [`Sender<T>`] has been dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(oneshot_channel)]
    /// use std::sync::oneshot;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let (tx, rx) = oneshot::channel();
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_millis(500));
    ///     tx.send("Done!").unwrap();
    /// });
    ///
    /// // This will block until the message arrives.
    /// println!("{}", rx.recv().unwrap());
    /// ```
    #[unstable(feature = "oneshot_channel", issue = "143674")]
    pub fn recv(self) -> Result<T, RecvError> {
        self.inner.recv()
    }

    // Fallible methods.

    /// Attempts to return a pending value on this receiver without blocking.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(oneshot_channel)]
    /// use std::sync::oneshot;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let (sender, mut receiver) = oneshot::channel();
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_millis(100));
    ///     sender.send(42).unwrap();
    /// });
    ///
    /// // Keep trying until we get the message, doing other work in the process.
    /// loop {
    ///     match receiver.try_recv() {
    ///         Ok(value) => {
    ///             assert_eq!(value, 42);
    ///             break;
    ///         }
    ///         Err(oneshot::TryRecvError::Empty(rx)) => {
    ///             // Retake ownership of the receiver.
    ///             receiver = rx;
    /// #           fn do_other_work() { thread::sleep(Duration::from_millis(25)); }
    ///             do_other_work();
    ///         }
    ///         Err(oneshot::TryRecvError::Disconnected) => panic!("Sender disconnected"),
    ///     }
    /// }
    /// ```
    #[unstable(feature = "oneshot_channel", issue = "143674")]
    pub fn try_recv(self) -> Result<T, TryRecvError<T>> {
        self.inner.try_recv().map_err(|err| match err {
            mpmc::TryRecvError::Empty => TryRecvError::Empty(self),
            mpmc::TryRecvError::Disconnected => TryRecvError::Disconnected,
        })
    }

    /// Attempts to wait for a value on this receiver, returning an error if the corresponding
    /// [`Sender`] half of this channel has been dropped, or if it waits more than `timeout`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(oneshot_channel)]
    /// use std::sync::oneshot;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let (sender, receiver) = oneshot::channel();
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_millis(500));
    ///     sender.send("Success!").unwrap();
    /// });
    ///
    /// // Wait up to 1 second for the message
    /// match receiver.recv_timeout(Duration::from_secs(1)) {
    ///     Ok(msg) => println!("Received: {}", msg),
    ///     Err(oneshot::RecvTimeoutError::Timeout(_)) => println!("Timed out!"),
    ///     Err(oneshot::RecvTimeoutError::Disconnected) => println!("Sender dropped!"),
    /// }
    /// ```
    #[unstable(feature = "oneshot_channel", issue = "143674")]
    pub fn recv_timeout(self, timeout: Duration) -> Result<T, RecvTimeoutError<T>> {
        self.inner.recv_timeout(timeout).map_err(|err| match err {
            mpmc::RecvTimeoutError::Timeout => RecvTimeoutError::Timeout(self),
            mpmc::RecvTimeoutError::Disconnected => RecvTimeoutError::Disconnected,
        })
    }

    /// Attempts to wait for a value on this receiver, returning an error if the corresponding
    /// [`Sender`] half of this channel has been dropped, or if `deadline` is reached.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(oneshot_channel)]
    /// use std::sync::oneshot;
    /// use std::thread;
    /// use std::time::{Duration, Instant};
    ///
    /// let (sender, receiver) = oneshot::channel();
    ///
    /// thread::spawn(move || {
    ///     thread::sleep(Duration::from_millis(100));
    ///     sender.send("Just in time!").unwrap();
    /// });
    ///
    /// let deadline = Instant::now() + Duration::from_millis(500);
    /// match receiver.recv_deadline(deadline) {
    ///     Ok(msg) => println!("Received: {}", msg),
    ///     Err(oneshot::RecvTimeoutError::Timeout(_)) => println!("Missed deadline!"),
    ///     Err(oneshot::RecvTimeoutError::Disconnected) => println!("Sender dropped!"),
    /// }
    /// ```
    #[unstable(feature = "oneshot_channel", issue = "143674")]
    pub fn recv_deadline(self, deadline: Instant) -> Result<T, RecvTimeoutError<T>> {
        self.inner.recv_deadline(deadline).map_err(|err| match err {
            mpmc::RecvTimeoutError::Timeout => RecvTimeoutError::Timeout(self),
            mpmc::RecvTimeoutError::Disconnected => RecvTimeoutError::Disconnected,
        })
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> fmt::Debug for Receiver<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receiver").finish_non_exhaustive()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Receiver Errors
////////////////////////////////////////////////////////////////////////////////////////////////////

/// An error returned from the [`try_recv`](Receiver::try_recv) method.
///
/// See the documentation for [`try_recv`] for more information on how to use this error.
///
/// [`try_recv`]: Receiver::try_recv
#[unstable(feature = "oneshot_channel", issue = "143674")]
pub enum TryRecvError<T> {
    /// The [`Sender`] has not sent a message yet, but it might in the future (as it has not yet
    /// disconnected). This variant contains the [`Receiver`] that [`try_recv`](Receiver::try_recv)
    /// took ownership over.
    Empty(Receiver<T>),
    /// The corresponding [`Sender`] half of this channel has become disconnected, and there will
    /// never be any more data sent over the channel.
    Disconnected,
}

/// An error returned from the [`recv_timeout`](Receiver::recv_timeout) or
/// [`recv_deadline`](Receiver::recv_deadline) methods.
///
/// # Examples
///
/// Usage of this error is similar to [`TryRecvError`].
///
/// ```
/// #![feature(oneshot_channel)]
/// use std::sync::oneshot::{self, RecvTimeoutError};
/// use std::thread;
/// use std::time::Duration;
///
/// let (sender, receiver) = oneshot::channel();
///
/// let send_failure = thread::spawn(move || {
///     // Simulate a long computation that takes longer than our timeout.
///     thread::sleep(Duration::from_millis(250));
///
///     // This will likely fail to send because we drop the receiver in the main thread.
///     sender.send("Goodbye!".to_string()).unwrap();
/// });
///
/// // Try to receive the message with a short timeout.
/// match receiver.recv_timeout(Duration::from_millis(10)) {
///     Ok(msg) => println!("Received: {}", msg),
///     Err(RecvTimeoutError::Timeout(rx)) => {
///         println!("Timed out waiting for message!");
///
///         // Note that you can reuse the receiver without dropping it.
///         drop(rx);
///     },
///     Err(RecvTimeoutError::Disconnected) => println!("Sender dropped!"),
/// }
///
/// send_failure.join().unwrap_err();
/// ```
#[unstable(feature = "oneshot_channel", issue = "143674")]
pub enum RecvTimeoutError<T> {
    /// The [`Sender`] has not sent a message yet, but it might in the future (as it has not yet
    /// disconnected). This variant contains the [`Receiver`] that either
    /// [`recv_timeout`](Receiver::recv_timeout) or [`recv_deadline`](Receiver::recv_deadline) took
    /// ownership over.
    Timeout(Receiver<T>),
    /// The corresponding [`Sender`] half of this channel has become disconnected, and there will
    /// never be any more data sent over the channel.
    Disconnected,
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> fmt::Debug for TryRecvError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TryRecvError").finish_non_exhaustive()
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> fmt::Display for TryRecvError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TryRecvError::Empty(..) => "receiving on an empty oneshot channel".fmt(f),
            TryRecvError::Disconnected => "receiving on a closed oneshot channel".fmt(f),
        }
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> error::Error for TryRecvError<T> {}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> From<RecvError> for TryRecvError<T> {
    /// Converts a `RecvError` into a `TryRecvError`.
    ///
    /// This conversion always returns `TryRecvError::Disconnected`.
    ///
    /// No data is allocated on the heap.
    fn from(err: RecvError) -> TryRecvError<T> {
        match err {
            RecvError => TryRecvError::Disconnected,
        }
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> fmt::Debug for RecvTimeoutError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RecvTimeoutError").finish_non_exhaustive()
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> fmt::Display for RecvTimeoutError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            RecvTimeoutError::Timeout(..) => "timed out waiting on oneshot channel".fmt(f),
            RecvTimeoutError::Disconnected => "receiving on a closed oneshot channel".fmt(f),
        }
    }
}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> error::Error for RecvTimeoutError<T> {}

#[unstable(feature = "oneshot_channel", issue = "143674")]
impl<T> From<RecvError> for RecvTimeoutError<T> {
    /// Converts a `RecvError` into a `RecvTimeoutError`.
    ///
    /// This conversion always returns `RecvTimeoutError::Disconnected`.
    ///
    /// No data is allocated on the heap.
    fn from(err: RecvError) -> RecvTimeoutError<T> {
        match err {
            RecvError => RecvTimeoutError::Disconnected,
        }
    }
}
