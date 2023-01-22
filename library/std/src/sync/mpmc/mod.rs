//! Multi-producer multi-consumer channels.

// This module is not currently exposed publicly, but is used
// as the implementation for the channels in `sync::mpsc`. The
// implementation comes from the crossbeam-channel crate:
//
// Copyright (c) 2019 The Crossbeam Project Developers
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

mod array;
mod context;
mod counter;
mod error;
mod list;
mod select;
mod utils;
mod waker;
mod zero;

use crate::fmt;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::time::{Duration, Instant};
pub use error::*;

/// Creates a channel of unbounded capacity.
///
/// This channel has a growable buffer that can hold any number of messages at a time.
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let (s, r) = counter::new(ChannelFlavor::List(list::Channel::new()));
    let s = Sender { flavor: s };
    let r = Receiver { flavor: r };
    (s, r)
}

/// Creates a channel of bounded capacity.
///
/// This channel has a buffer that can hold at most `cap` messages at a time.
///
/// A special case is zero-capacity channel, which cannot hold any messages. Instead, send and
/// receive operations must appear at the same time in order to pair up and pass the message over.
pub fn sync_channel<T>(cap: usize) -> (Sender<T>, Receiver<T>) {
    if cap == 0 {
        let (s, r) = counter::new(ChannelFlavor::Zero(zero::Channel::new()));
        let s = Sender { flavor: s };
        let r = Receiver { flavor: r };
        (s, r)
    } else {
        let (s, r) = counter::new(ChannelFlavor::Array(array::Channel::with_capacity(cap)));
        let s = Sender { flavor: s };
        let r = Receiver { flavor: r };
        (s, r)
    }
}

enum ChannelFlavor<T> {
    /// Bounded channel based on a preallocated array.
    Array(array::Channel<T>),

    /// Unbounded channel implemented as a linked list.
    List(list::Channel<T>),

    /// Zero-capacity channel.
    Zero(zero::Channel<T>),
}

/// The sending side of a channel.
pub struct Sender<T> {
    flavor: counter::Sender<ChannelFlavor<T>>,
}

unsafe impl<T: Send> Send for Sender<T> {}
unsafe impl<T: Send> Sync for Sender<T> {}

impl<T> UnwindSafe for Sender<T> {}
impl<T> RefUnwindSafe for Sender<T> {}

impl<T> Sender<T> {
    /// Attempts to send a message into the channel without blocking.
    ///
    /// This method will either send a message into the channel immediately or return an error if
    /// the channel is full or disconnected. The returned error contains the original message.
    ///
    /// If called on a zero-capacity channel, this method will send the message only if there
    /// happens to be a receive operation on the other side of the channel at the same time.
    pub fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.try_send(msg),
            ChannelFlavor::List(chan) => chan.try_send(msg),
            ChannelFlavor::Zero(chan) => chan.try_send(msg),
        }
    }

    /// Blocks the current thread until a message is sent or the channel is disconnected.
    ///
    /// If the channel is full and not disconnected, this call will block until the send operation
    /// can proceed. If the channel becomes disconnected, this call will wake up and return an
    /// error. The returned error contains the original message.
    ///
    /// If called on a zero-capacity channel, this method will wait for a receive operation to
    /// appear on the other side of the channel.
    pub fn send(&self, msg: T) -> Result<(), SendError<T>> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.send(msg, None),
            ChannelFlavor::List(chan) => chan.send(msg, None),
            ChannelFlavor::Zero(chan) => chan.send(msg, None),
        }
        .map_err(|err| match err {
            SendTimeoutError::Disconnected(msg) => SendError(msg),
            SendTimeoutError::Timeout(_) => unreachable!(),
        })
    }
}

// The methods below are not used by `sync::mpsc`, but
// are useful and we'll likely want to expose them
// eventually
#[allow(unused)]
impl<T> Sender<T> {
    /// Waits for a message to be sent into the channel, but only for a limited time.
    ///
    /// If the channel is full and not disconnected, this call will block until the send operation
    /// can proceed or the operation times out. If the channel becomes disconnected, this call will
    /// wake up and return an error. The returned error contains the original message.
    ///
    /// If called on a zero-capacity channel, this method will wait for a receive operation to
    /// appear on the other side of the channel.
    pub fn send_timeout(&self, msg: T, timeout: Duration) -> Result<(), SendTimeoutError<T>> {
        match Instant::now().checked_add(timeout) {
            Some(deadline) => self.send_deadline(msg, deadline),
            // So far in the future that it's practically the same as waiting indefinitely.
            None => self.send(msg).map_err(SendTimeoutError::from),
        }
    }

    /// Waits for a message to be sent into the channel, but only until a given deadline.
    ///
    /// If the channel is full and not disconnected, this call will block until the send operation
    /// can proceed or the operation times out. If the channel becomes disconnected, this call will
    /// wake up and return an error. The returned error contains the original message.
    ///
    /// If called on a zero-capacity channel, this method will wait for a receive operation to
    /// appear on the other side of the channel.
    pub fn send_deadline(&self, msg: T, deadline: Instant) -> Result<(), SendTimeoutError<T>> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.send(msg, Some(deadline)),
            ChannelFlavor::List(chan) => chan.send(msg, Some(deadline)),
            ChannelFlavor::Zero(chan) => chan.send(msg, Some(deadline)),
        }
    }

    /// Returns `true` if the channel is empty.
    ///
    /// Note: Zero-capacity channels are always empty.
    pub fn is_empty(&self) -> bool {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.is_empty(),
            ChannelFlavor::List(chan) => chan.is_empty(),
            ChannelFlavor::Zero(chan) => chan.is_empty(),
        }
    }

    /// Returns `true` if the channel is full.
    ///
    /// Note: Zero-capacity channels are always full.
    pub fn is_full(&self) -> bool {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.is_full(),
            ChannelFlavor::List(chan) => chan.is_full(),
            ChannelFlavor::Zero(chan) => chan.is_full(),
        }
    }

    /// Returns the number of messages in the channel.
    pub fn len(&self) -> usize {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.len(),
            ChannelFlavor::List(chan) => chan.len(),
            ChannelFlavor::Zero(chan) => chan.len(),
        }
    }

    /// If the channel is bounded, returns its capacity.
    pub fn capacity(&self) -> Option<usize> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.capacity(),
            ChannelFlavor::List(chan) => chan.capacity(),
            ChannelFlavor::Zero(chan) => chan.capacity(),
        }
    }

    /// Returns `true` if senders belong to the same channel.
    pub fn same_channel(&self, other: &Sender<T>) -> bool {
        self.flavor == other.flavor
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        unsafe {
            self.flavor.release(|chan| match chan {
                ChannelFlavor::Array(c) => c.disconnect(),
                ChannelFlavor::List(c) => c.disconnect_senders(),
                ChannelFlavor::Zero(c) => c.disconnect(),
            });
        }
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Sender { flavor: self.flavor.acquire() }
    }
}

impl<T> fmt::Debug for Sender<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Sender { .. }")
    }
}

/// The receiving side of a channel.
pub struct Receiver<T> {
    flavor: counter::Receiver<ChannelFlavor<T>>,
}

unsafe impl<T: Send> Send for Receiver<T> {}
unsafe impl<T: Send> Sync for Receiver<T> {}

impl<T> UnwindSafe for Receiver<T> {}
impl<T> RefUnwindSafe for Receiver<T> {}

impl<T> Receiver<T> {
    /// Attempts to receive a message from the channel without blocking.
    ///
    /// This method will either receive a message from the channel immediately or return an error
    /// if the channel is empty.
    ///
    /// If called on a zero-capacity channel, this method will receive a message only if there
    /// happens to be a send operation on the other side of the channel at the same time.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.try_recv(),
            ChannelFlavor::List(chan) => chan.try_recv(),
            ChannelFlavor::Zero(chan) => chan.try_recv(),
        }
    }

    /// Blocks the current thread until a message is received or the channel is empty and
    /// disconnected.
    ///
    /// If the channel is empty and not disconnected, this call will block until the receive
    /// operation can proceed. If the channel is empty and becomes disconnected, this call will
    /// wake up and return an error.
    ///
    /// If called on a zero-capacity channel, this method will wait for a send operation to appear
    /// on the other side of the channel.
    pub fn recv(&self) -> Result<T, RecvError> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.recv(None),
            ChannelFlavor::List(chan) => chan.recv(None),
            ChannelFlavor::Zero(chan) => chan.recv(None),
        }
        .map_err(|_| RecvError)
    }

    /// Waits for a message to be received from the channel, but only for a limited time.
    ///
    /// If the channel is empty and not disconnected, this call will block until the receive
    /// operation can proceed or the operation times out. If the channel is empty and becomes
    /// disconnected, this call will wake up and return an error.
    ///
    /// If called on a zero-capacity channel, this method will wait for a send operation to appear
    /// on the other side of the channel.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvTimeoutError> {
        match Instant::now().checked_add(timeout) {
            Some(deadline) => self.recv_deadline(deadline),
            // So far in the future that it's practically the same as waiting indefinitely.
            None => self.recv().map_err(RecvTimeoutError::from),
        }
    }

    /// Waits for a message to be received from the channel, but only for a limited time.
    ///
    /// If the channel is empty and not disconnected, this call will block until the receive
    /// operation can proceed or the operation times out. If the channel is empty and becomes
    /// disconnected, this call will wake up and return an error.
    ///
    /// If called on a zero-capacity channel, this method will wait for a send operation to appear
    /// on the other side of the channel.
    pub fn recv_deadline(&self, deadline: Instant) -> Result<T, RecvTimeoutError> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.recv(Some(deadline)),
            ChannelFlavor::List(chan) => chan.recv(Some(deadline)),
            ChannelFlavor::Zero(chan) => chan.recv(Some(deadline)),
        }
    }
}

// The methods below are not used by `sync::mpsc`, but
// are useful and we'll likely want to expose them
// eventually
#[allow(unused)]
impl<T> Receiver<T> {
    /// Returns `true` if the channel is empty.
    ///
    /// Note: Zero-capacity channels are always empty.
    pub fn is_empty(&self) -> bool {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.is_empty(),
            ChannelFlavor::List(chan) => chan.is_empty(),
            ChannelFlavor::Zero(chan) => chan.is_empty(),
        }
    }

    /// Returns `true` if the channel is full.
    ///
    /// Note: Zero-capacity channels are always full.
    pub fn is_full(&self) -> bool {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.is_full(),
            ChannelFlavor::List(chan) => chan.is_full(),
            ChannelFlavor::Zero(chan) => chan.is_full(),
        }
    }

    /// Returns the number of messages in the channel.
    pub fn len(&self) -> usize {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.len(),
            ChannelFlavor::List(chan) => chan.len(),
            ChannelFlavor::Zero(chan) => chan.len(),
        }
    }

    /// If the channel is bounded, returns its capacity.
    pub fn capacity(&self) -> Option<usize> {
        match &*self.flavor {
            ChannelFlavor::Array(chan) => chan.capacity(),
            ChannelFlavor::List(chan) => chan.capacity(),
            ChannelFlavor::Zero(chan) => chan.capacity(),
        }
    }

    /// Returns `true` if receivers belong to the same channel.
    pub fn same_channel(&self, other: &Receiver<T>) -> bool {
        self.flavor == other.flavor
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        unsafe {
            self.flavor.release(|chan| match chan {
                ChannelFlavor::Array(c) => c.disconnect(),
                ChannelFlavor::List(c) => c.disconnect_receivers(),
                ChannelFlavor::Zero(c) => c.disconnect(),
            });
        }
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        Receiver { flavor: self.flavor.acquire() }
    }
}

impl<T> fmt::Debug for Receiver<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Receiver { .. }")
    }
}
