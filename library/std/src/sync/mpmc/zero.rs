//! Zero-capacity channel.
//!
//! This kind of channel is also known as *rendezvous* channel.

use super::context::Context;
use super::error::*;
use super::select::{Operation, Selected, Token};
use super::utils::Backoff;
use super::waker::Waker;
use crate::cell::UnsafeCell;
use crate::marker::PhantomData;
use crate::sync::Mutex;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::time::Instant;
use crate::{fmt, ptr};

/// A pointer to a packet.
pub(crate) struct ZeroToken(*mut ());

impl Default for ZeroToken {
    fn default() -> Self {
        Self(ptr::null_mut())
    }
}

impl fmt::Debug for ZeroToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&(self.0 as usize), f)
    }
}

/// A slot for passing one message from a sender to a receiver.
struct Packet<T> {
    /// Equals `true` if the packet is allocated on the stack.
    on_stack: bool,

    /// Equals `true` once the packet is ready for reading or writing.
    ready: AtomicBool,

    /// The message.
    msg: UnsafeCell<Option<T>>,
}

impl<T> Packet<T> {
    /// Creates an empty packet on the stack.
    fn empty_on_stack() -> Packet<T> {
        Packet { on_stack: true, ready: AtomicBool::new(false), msg: UnsafeCell::new(None) }
    }

    /// Creates a packet on the stack, containing a message.
    fn message_on_stack(msg: T) -> Packet<T> {
        Packet { on_stack: true, ready: AtomicBool::new(false), msg: UnsafeCell::new(Some(msg)) }
    }

    /// Waits until the packet becomes ready for reading or writing.
    fn wait_ready(&self) {
        let backoff = Backoff::new();
        while !self.ready.load(Ordering::Acquire) {
            backoff.spin_heavy();
        }
    }
}

/// Inner representation of a zero-capacity channel.
struct Inner {
    /// Senders waiting to pair up with a receive operation.
    senders: Waker,

    /// Receivers waiting to pair up with a send operation.
    receivers: Waker,

    /// Equals `true` when the channel is disconnected.
    is_disconnected: bool,
}

/// Zero-capacity channel.
pub(crate) struct Channel<T> {
    /// Inner representation of the channel.
    inner: Mutex<Inner>,

    /// Indicates that dropping a `Channel<T>` may drop values of type `T`.
    _marker: PhantomData<T>,
}

impl<T> Channel<T> {
    /// Constructs a new zero-capacity channel.
    pub(crate) fn new() -> Self {
        Channel {
            inner: Mutex::new(Inner {
                senders: Waker::new(),
                receivers: Waker::new(),
                is_disconnected: false,
            }),
            _marker: PhantomData,
        }
    }

    /// Writes a message into the packet.
    pub(crate) unsafe fn write(&self, token: &mut Token, msg: T) -> Result<(), T> {
        // If there is no packet, the channel is disconnected.
        if token.zero.0.is_null() {
            return Err(msg);
        }

        unsafe {
            let packet = &*(token.zero.0 as *const Packet<T>);
            packet.msg.get().write(Some(msg));
            packet.ready.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Reads a message from the packet.
    pub(crate) unsafe fn read(&self, token: &mut Token) -> Result<T, ()> {
        // If there is no packet, the channel is disconnected.
        if token.zero.0.is_null() {
            return Err(());
        }

        let packet = unsafe { &*(token.zero.0 as *const Packet<T>) };

        if packet.on_stack {
            // The message has been in the packet from the beginning, so there is no need to wait
            // for it. However, after reading the message, we need to set `ready` to `true` in
            // order to signal that the packet can be destroyed.
            let msg = unsafe { packet.msg.get().replace(None) }.unwrap();
            packet.ready.store(true, Ordering::Release);
            Ok(msg)
        } else {
            // Wait until the message becomes available, then read it and destroy the
            // heap-allocated packet.
            packet.wait_ready();
            unsafe {
                let msg = packet.msg.get().replace(None).unwrap();
                drop(Box::from_raw(token.zero.0 as *mut Packet<T>));
                Ok(msg)
            }
        }
    }

    /// Attempts to send a message into the channel.
    pub(crate) fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        let token = &mut Token::default();
        let mut inner = self.inner.lock().unwrap();

        // If there's a waiting receiver, pair up with it.
        if let Some(operation) = inner.receivers.try_select() {
            token.zero.0 = operation.packet;
            drop(inner);
            unsafe {
                self.write(token, msg).ok().unwrap();
            }
            Ok(())
        } else if inner.is_disconnected {
            Err(TrySendError::Disconnected(msg))
        } else {
            Err(TrySendError::Full(msg))
        }
    }

    /// Sends a message into the channel.
    pub(crate) fn send(
        &self,
        msg: T,
        deadline: Option<Instant>,
    ) -> Result<(), SendTimeoutError<T>> {
        let token = &mut Token::default();
        let mut inner = self.inner.lock().unwrap();

        // If there's a waiting receiver, pair up with it.
        if let Some(operation) = inner.receivers.try_select() {
            token.zero.0 = operation.packet;
            drop(inner);
            unsafe {
                self.write(token, msg).ok().unwrap();
            }
            return Ok(());
        }

        if inner.is_disconnected {
            return Err(SendTimeoutError::Disconnected(msg));
        }

        Context::with(|cx| {
            // Prepare for blocking until a receiver wakes us up.
            let oper = Operation::hook(token);
            let mut packet = Packet::<T>::message_on_stack(msg);
            inner.senders.register_with_packet(oper, (&raw mut packet) as *mut (), cx);
            inner.receivers.notify();
            drop(inner);

            // Block the current thread.
            // SAFETY: the context belongs to the current thread.
            let sel = unsafe { cx.wait_until(deadline) };

            match sel {
                Selected::Waiting => unreachable!(),
                Selected::Aborted => {
                    self.inner.lock().unwrap().senders.unregister(oper).unwrap();
                    let msg = unsafe { packet.msg.get().replace(None).unwrap() };
                    Err(SendTimeoutError::Timeout(msg))
                }
                Selected::Disconnected => {
                    self.inner.lock().unwrap().senders.unregister(oper).unwrap();
                    let msg = unsafe { packet.msg.get().replace(None).unwrap() };
                    Err(SendTimeoutError::Disconnected(msg))
                }
                Selected::Operation(_) => {
                    // Wait until the message is read, then drop the packet.
                    packet.wait_ready();
                    Ok(())
                }
            }
        })
    }

    /// Attempts to receive a message without blocking.
    pub(crate) fn try_recv(&self) -> Result<T, TryRecvError> {
        let token = &mut Token::default();
        let mut inner = self.inner.lock().unwrap();

        // If there's a waiting sender, pair up with it.
        if let Some(operation) = inner.senders.try_select() {
            token.zero.0 = operation.packet;
            drop(inner);
            unsafe { self.read(token).map_err(|_| TryRecvError::Disconnected) }
        } else if inner.is_disconnected {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    /// Receives a message from the channel.
    pub(crate) fn recv(&self, deadline: Option<Instant>) -> Result<T, RecvTimeoutError> {
        let token = &mut Token::default();
        let mut inner = self.inner.lock().unwrap();

        // If there's a waiting sender, pair up with it.
        if let Some(operation) = inner.senders.try_select() {
            token.zero.0 = operation.packet;
            drop(inner);
            unsafe {
                return self.read(token).map_err(|_| RecvTimeoutError::Disconnected);
            }
        }

        if inner.is_disconnected {
            return Err(RecvTimeoutError::Disconnected);
        }

        Context::with(|cx| {
            // Prepare for blocking until a sender wakes us up.
            let oper = Operation::hook(token);
            let mut packet = Packet::<T>::empty_on_stack();
            inner.receivers.register_with_packet(oper, (&raw mut packet) as *mut (), cx);
            inner.senders.notify();
            drop(inner);

            // Block the current thread.
            // SAFETY: the context belongs to the current thread.
            let sel = unsafe { cx.wait_until(deadline) };

            match sel {
                Selected::Waiting => unreachable!(),
                Selected::Aborted => {
                    self.inner.lock().unwrap().receivers.unregister(oper).unwrap();
                    Err(RecvTimeoutError::Timeout)
                }
                Selected::Disconnected => {
                    self.inner.lock().unwrap().receivers.unregister(oper).unwrap();
                    Err(RecvTimeoutError::Disconnected)
                }
                Selected::Operation(_) => {
                    // Wait until the message is provided, then read it.
                    packet.wait_ready();
                    unsafe { Ok(packet.msg.get().replace(None).unwrap()) }
                }
            }
        })
    }

    /// Disconnects the channel and wakes up all blocked senders and receivers.
    ///
    /// Returns `true` if this call disconnected the channel.
    pub(crate) fn disconnect(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();

        if !inner.is_disconnected {
            inner.is_disconnected = true;
            inner.senders.disconnect();
            inner.receivers.disconnect();
            true
        } else {
            false
        }
    }

    /// Returns the current number of messages inside the channel.
    pub(crate) fn len(&self) -> usize {
        0
    }

    /// Returns the capacity of the channel.
    #[allow(clippy::unnecessary_wraps)] // This is intentional.
    pub(crate) fn capacity(&self) -> Option<usize> {
        Some(0)
    }

    /// Returns `true` if the channel is empty.
    pub(crate) fn is_empty(&self) -> bool {
        true
    }

    /// Returns `true` if the channel is full.
    pub(crate) fn is_full(&self) -> bool {
        true
    }
}
