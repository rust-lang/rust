//! Bounded channel based on a preallocated array.
//!
//! This flavor has a fixed, positive capacity.
//!
//! The implementation is based on Dmitry Vyukov's bounded MPMC queue.
//!
//! Source:
//!   - <http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue>
//!   - <https://docs.google.com/document/d/1yIAYmbvL3JxOKOjuCyon7JhW4cSv1wy5hC0ApeGMV9s/pub>

use super::context::Context;
use super::error::*;
use super::select::{Operation, Selected, Token};
use super::utils::{Backoff, CachePadded};
use super::waker::SyncWaker;
use crate::cell::UnsafeCell;
use crate::mem::MaybeUninit;
use crate::ptr;
use crate::sync::atomic::{self, AtomicUsize, Ordering};
use crate::time::Instant;

/// A slot in a channel.
struct Slot<T> {
    /// The current stamp.
    stamp: AtomicUsize,

    /// The message in this slot. Either read out in `read` or dropped through
    /// `discard_all_messages`.
    msg: UnsafeCell<MaybeUninit<T>>,
}

/// The token type for the array flavor.
#[derive(Debug)]
pub(crate) struct ArrayToken {
    /// Slot to read from or write to.
    slot: *const u8,

    /// Stamp to store into the slot after reading or writing.
    stamp: usize,
}

impl Default for ArrayToken {
    #[inline]
    fn default() -> Self {
        ArrayToken { slot: ptr::null(), stamp: 0 }
    }
}

/// Bounded channel based on a preallocated array.
pub(crate) struct Channel<T> {
    /// The head of the channel.
    ///
    /// This value is a "stamp" consisting of an index into the buffer, a mark bit, and a lap, but
    /// packed into a single `usize`. The lower bits represent the index, while the upper bits
    /// represent the lap. The mark bit in the head is always zero.
    ///
    /// Messages are popped from the head of the channel.
    head: CachePadded<AtomicUsize>,

    /// The tail of the channel.
    ///
    /// This value is a "stamp" consisting of an index into the buffer, a mark bit, and a lap, but
    /// packed into a single `usize`. The lower bits represent the index, while the upper bits
    /// represent the lap. The mark bit indicates that the channel is disconnected.
    ///
    /// Messages are pushed into the tail of the channel.
    tail: CachePadded<AtomicUsize>,

    /// The buffer holding slots.
    buffer: Box<[Slot<T>]>,

    /// The channel capacity.
    cap: usize,

    /// A stamp with the value of `{ lap: 1, mark: 0, index: 0 }`.
    one_lap: usize,

    /// If this bit is set in the tail, that means the channel is disconnected.
    mark_bit: usize,

    /// Senders waiting while the channel is full.
    senders: SyncWaker,

    /// Receivers waiting while the channel is empty and not disconnected.
    receivers: SyncWaker,
}

impl<T> Channel<T> {
    /// Creates a bounded channel of capacity `cap`.
    pub(crate) fn with_capacity(cap: usize) -> Self {
        assert!(cap > 0, "capacity must be positive");

        // Compute constants `mark_bit` and `one_lap`.
        let mark_bit = (cap + 1).next_power_of_two();
        let one_lap = mark_bit * 2;

        // Head is initialized to `{ lap: 0, mark: 0, index: 0 }`.
        let head = 0;
        // Tail is initialized to `{ lap: 0, mark: 0, index: 0 }`.
        let tail = 0;

        // Allocate a buffer of `cap` slots initialized
        // with stamps.
        let buffer: Box<[Slot<T>]> = (0..cap)
            .map(|i| {
                // Set the stamp to `{ lap: 0, mark: 0, index: i }`.
                Slot { stamp: AtomicUsize::new(i), msg: UnsafeCell::new(MaybeUninit::uninit()) }
            })
            .collect();

        Channel {
            buffer,
            cap,
            one_lap,
            mark_bit,
            head: CachePadded::new(AtomicUsize::new(head)),
            tail: CachePadded::new(AtomicUsize::new(tail)),
            senders: SyncWaker::new(),
            receivers: SyncWaker::new(),
        }
    }

    /// Attempts to reserve a slot for sending a message.
    fn start_send(&self, token: &mut Token) -> bool {
        let backoff = Backoff::new();
        let mut tail = self.tail.load(Ordering::Relaxed);

        loop {
            // Check if the channel is disconnected.
            if tail & self.mark_bit != 0 {
                token.array.slot = ptr::null();
                token.array.stamp = 0;
                return true;
            }

            // Deconstruct the tail.
            let index = tail & (self.mark_bit - 1);
            let lap = tail & !(self.one_lap - 1);

            // Inspect the corresponding slot.
            debug_assert!(index < self.buffer.len());
            let slot = unsafe { self.buffer.get_unchecked(index) };
            let stamp = slot.stamp.load(Ordering::Acquire);

            // If the tail and the stamp match, we may attempt to push.
            if tail == stamp {
                let new_tail = if index + 1 < self.cap {
                    // Same lap, incremented index.
                    // Set to `{ lap: lap, mark: 0, index: index + 1 }`.
                    tail + 1
                } else {
                    // One lap forward, index wraps around to zero.
                    // Set to `{ lap: lap.wrapping_add(1), mark: 0, index: 0 }`.
                    lap.wrapping_add(self.one_lap)
                };

                // Try moving the tail.
                match self.tail.compare_exchange_weak(
                    tail,
                    new_tail,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Prepare the token for the follow-up call to `write`.
                        token.array.slot = slot as *const Slot<T> as *const u8;
                        token.array.stamp = tail + 1;
                        return true;
                    }
                    Err(_) => {
                        backoff.spin_light();
                        tail = self.tail.load(Ordering::Relaxed);
                    }
                }
            } else if stamp.wrapping_add(self.one_lap) == tail + 1 {
                atomic::fence(Ordering::SeqCst);
                let head = self.head.load(Ordering::Relaxed);

                // If the head lags one lap behind the tail as well...
                if head.wrapping_add(self.one_lap) == tail {
                    // ...then the channel is full.
                    return false;
                }

                backoff.spin_light();
                tail = self.tail.load(Ordering::Relaxed);
            } else {
                // Snooze because we need to wait for the stamp to get updated.
                backoff.spin_heavy();
                tail = self.tail.load(Ordering::Relaxed);
            }
        }
    }

    /// Writes a message into the channel.
    pub(crate) unsafe fn write(&self, token: &mut Token, msg: T) -> Result<(), T> {
        // If there is no slot, the channel is disconnected.
        if token.array.slot.is_null() {
            return Err(msg);
        }

        // Write the message into the slot and update the stamp.
        unsafe {
            let slot: &Slot<T> = &*(token.array.slot as *const Slot<T>);
            slot.msg.get().write(MaybeUninit::new(msg));
            slot.stamp.store(token.array.stamp, Ordering::Release);
        }

        // Wake a sleeping receiver.
        self.receivers.notify();
        Ok(())
    }

    /// Attempts to reserve a slot for receiving a message.
    fn start_recv(&self, token: &mut Token) -> bool {
        let backoff = Backoff::new();
        let mut head = self.head.load(Ordering::Relaxed);

        loop {
            // Deconstruct the head.
            let index = head & (self.mark_bit - 1);
            let lap = head & !(self.one_lap - 1);

            // Inspect the corresponding slot.
            debug_assert!(index < self.buffer.len());
            let slot = unsafe { self.buffer.get_unchecked(index) };
            let stamp = slot.stamp.load(Ordering::Acquire);

            // If the stamp is ahead of the head by 1, we may attempt to pop.
            if head + 1 == stamp {
                let new = if index + 1 < self.cap {
                    // Same lap, incremented index.
                    // Set to `{ lap: lap, mark: 0, index: index + 1 }`.
                    head + 1
                } else {
                    // One lap forward, index wraps around to zero.
                    // Set to `{ lap: lap.wrapping_add(1), mark: 0, index: 0 }`.
                    lap.wrapping_add(self.one_lap)
                };

                // Try moving the head.
                match self.head.compare_exchange_weak(
                    head,
                    new,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Prepare the token for the follow-up call to `read`.
                        token.array.slot = slot as *const Slot<T> as *const u8;
                        token.array.stamp = head.wrapping_add(self.one_lap);
                        return true;
                    }
                    Err(_) => {
                        backoff.spin_light();
                        head = self.head.load(Ordering::Relaxed);
                    }
                }
            } else if stamp == head {
                atomic::fence(Ordering::SeqCst);
                let tail = self.tail.load(Ordering::Relaxed);

                // If the tail equals the head, that means the channel is empty.
                if (tail & !self.mark_bit) == head {
                    // If the channel is disconnected...
                    if tail & self.mark_bit != 0 {
                        // ...then receive an error.
                        token.array.slot = ptr::null();
                        token.array.stamp = 0;
                        return true;
                    } else {
                        // Otherwise, the receive operation is not ready.
                        return false;
                    }
                }

                backoff.spin_light();
                head = self.head.load(Ordering::Relaxed);
            } else {
                // Snooze because we need to wait for the stamp to get updated.
                backoff.spin_heavy();
                head = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// Reads a message from the channel.
    pub(crate) unsafe fn read(&self, token: &mut Token) -> Result<T, ()> {
        if token.array.slot.is_null() {
            // The channel is disconnected.
            return Err(());
        }

        // Read the message from the slot and update the stamp.
        let msg = unsafe {
            let slot: &Slot<T> = &*(token.array.slot as *const Slot<T>);

            let msg = slot.msg.get().read().assume_init();
            slot.stamp.store(token.array.stamp, Ordering::Release);
            msg
        };

        // Wake a sleeping sender.
        self.senders.notify();
        Ok(msg)
    }

    /// Attempts to send a message into the channel.
    pub(crate) fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        let token = &mut Token::default();
        if self.start_send(token) {
            unsafe { self.write(token, msg).map_err(TrySendError::Disconnected) }
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
        loop {
            // Try sending a message.
            if self.start_send(token) {
                let res = unsafe { self.write(token, msg) };
                return res.map_err(SendTimeoutError::Disconnected);
            }

            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Err(SendTimeoutError::Timeout(msg));
                }
            }

            Context::with(|cx| {
                // Prepare for blocking until a receiver wakes us up.
                let oper = Operation::hook(token);
                self.senders.register(oper, cx);

                // Has the channel become ready just now?
                if !self.is_full() || self.is_disconnected() {
                    let _ = cx.try_select(Selected::Aborted);
                }

                // Block the current thread.
                // SAFETY: the context belongs to the current thread.
                let sel = unsafe { cx.wait_until(deadline) };

                match sel {
                    Selected::Waiting => unreachable!(),
                    Selected::Aborted | Selected::Disconnected => {
                        self.senders.unregister(oper).unwrap();
                    }
                    Selected::Operation(_) => {}
                }
            });
        }
    }

    /// Attempts to receive a message without blocking.
    pub(crate) fn try_recv(&self) -> Result<T, TryRecvError> {
        let token = &mut Token::default();

        if self.start_recv(token) {
            unsafe { self.read(token).map_err(|_| TryRecvError::Disconnected) }
        } else {
            Err(TryRecvError::Empty)
        }
    }

    /// Receives a message from the channel.
    pub(crate) fn recv(&self, deadline: Option<Instant>) -> Result<T, RecvTimeoutError> {
        let token = &mut Token::default();
        loop {
            // Try receiving a message.
            if self.start_recv(token) {
                let res = unsafe { self.read(token) };
                return res.map_err(|_| RecvTimeoutError::Disconnected);
            }

            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Err(RecvTimeoutError::Timeout);
                }
            }

            Context::with(|cx| {
                // Prepare for blocking until a sender wakes us up.
                let oper = Operation::hook(token);
                self.receivers.register(oper, cx);

                // Has the channel become ready just now?
                if !self.is_empty() || self.is_disconnected() {
                    let _ = cx.try_select(Selected::Aborted);
                }

                // Block the current thread.
                // SAFETY: the context belongs to the current thread.
                let sel = unsafe { cx.wait_until(deadline) };

                match sel {
                    Selected::Waiting => unreachable!(),
                    Selected::Aborted | Selected::Disconnected => {
                        self.receivers.unregister(oper).unwrap();
                        // If the channel was disconnected, we still have to check for remaining
                        // messages.
                    }
                    Selected::Operation(_) => {}
                }
            });
        }
    }

    /// Returns the current number of messages inside the channel.
    pub(crate) fn len(&self) -> usize {
        loop {
            // Load the tail, then load the head.
            let tail = self.tail.load(Ordering::SeqCst);
            let head = self.head.load(Ordering::SeqCst);

            // If the tail didn't change, we've got consistent values to work with.
            if self.tail.load(Ordering::SeqCst) == tail {
                let hix = head & (self.mark_bit - 1);
                let tix = tail & (self.mark_bit - 1);

                return if hix < tix {
                    tix - hix
                } else if hix > tix {
                    self.cap - hix + tix
                } else if (tail & !self.mark_bit) == head {
                    0
                } else {
                    self.cap
                };
            }
        }
    }

    /// Returns the capacity of the channel.
    #[allow(clippy::unnecessary_wraps)] // This is intentional.
    pub(crate) fn capacity(&self) -> Option<usize> {
        Some(self.cap)
    }

    /// Disconnects senders and wakes up all blocked receivers.
    ///
    /// Returns `true` if this call disconnected the channel.
    pub(crate) fn disconnect_senders(&self) -> bool {
        let tail = self.tail.fetch_or(self.mark_bit, Ordering::SeqCst);

        if tail & self.mark_bit == 0 {
            self.receivers.disconnect();
            true
        } else {
            false
        }
    }

    /// Disconnects receivers and wakes up all blocked senders.
    ///
    /// Returns `true` if this call disconnected the channel.
    ///
    /// # Safety
    /// May only be called once upon dropping the last receiver. The
    /// destruction of all other receivers must have been observed with acquire
    /// ordering or stronger.
    pub(crate) unsafe fn disconnect_receivers(&self) -> bool {
        let tail = self.tail.fetch_or(self.mark_bit, Ordering::SeqCst);
        let disconnected = if tail & self.mark_bit == 0 {
            self.senders.disconnect();
            true
        } else {
            false
        };

        unsafe { self.discard_all_messages(tail) };
        disconnected
    }

    /// Discards all messages.
    ///
    /// `tail` should be the current (and therefore last) value of `tail`.
    ///
    /// # Panicking
    /// If a destructor panics, the remaining messages are leaked, matching the
    /// behavior of the unbounded channel.
    ///
    /// # Safety
    /// This method must only be called when dropping the last receiver. The
    /// destruction of all other receivers must have been observed with acquire
    /// ordering or stronger.
    unsafe fn discard_all_messages(&self, tail: usize) {
        debug_assert!(self.is_disconnected());

        // Only receivers modify `head`, so since we are the last one,
        // this value will not change and will not be observed (since
        // no new messages can be sent after disconnection).
        let mut head = self.head.load(Ordering::Relaxed);
        let tail = tail & !self.mark_bit;

        let backoff = Backoff::new();
        loop {
            // Deconstruct the head.
            let index = head & (self.mark_bit - 1);
            let lap = head & !(self.one_lap - 1);

            // Inspect the corresponding slot.
            debug_assert!(index < self.buffer.len());
            let slot = unsafe { self.buffer.get_unchecked(index) };
            let stamp = slot.stamp.load(Ordering::Acquire);

            // If the stamp is ahead of the head by 1, we may drop the message.
            if head + 1 == stamp {
                head = if index + 1 < self.cap {
                    // Same lap, incremented index.
                    // Set to `{ lap: lap, mark: 0, index: index + 1 }`.
                    head + 1
                } else {
                    // One lap forward, index wraps around to zero.
                    // Set to `{ lap: lap.wrapping_add(1), mark: 0, index: 0 }`.
                    lap.wrapping_add(self.one_lap)
                };

                unsafe {
                    (*slot.msg.get()).assume_init_drop();
                }
            // If the tail equals the head, that means the channel is empty.
            } else if tail == head {
                return;
            // Otherwise, a sender is about to write into the slot, so we need
            // to wait for it to update the stamp.
            } else {
                backoff.spin_heavy();
            }
        }
    }

    /// Returns `true` if the channel is disconnected.
    pub(crate) fn is_disconnected(&self) -> bool {
        self.tail.load(Ordering::SeqCst) & self.mark_bit != 0
    }

    /// Returns `true` if the channel is empty.
    pub(crate) fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::SeqCst);
        let tail = self.tail.load(Ordering::SeqCst);

        // Is the tail equal to the head?
        //
        // Note: If the head changes just before we load the tail, that means there was a moment
        // when the channel was not empty, so it is safe to just return `false`.
        (tail & !self.mark_bit) == head
    }

    /// Returns `true` if the channel is full.
    pub(crate) fn is_full(&self) -> bool {
        let tail = self.tail.load(Ordering::SeqCst);
        let head = self.head.load(Ordering::SeqCst);

        // Is the head lagging one lap behind tail?
        //
        // Note: If the tail changes just before we load the head, that means there was a moment
        // when the channel was not full, so it is safe to just return `false`.
        head.wrapping_add(self.one_lap) == tail & !self.mark_bit
    }
}
