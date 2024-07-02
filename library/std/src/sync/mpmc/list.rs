//! Unbounded channel implemented as a linked list.

use super::context::Context;
use super::error::*;
use super::select::{Operation, Selected, Token};
use super::utils::{Backoff, CachePadded};
use super::waker::SyncWaker;

use crate::cell::UnsafeCell;
use crate::marker::PhantomData;
use crate::mem::MaybeUninit;
use crate::ptr::{self, NonNull};
use crate::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};
use crate::time::Instant;

// Bits indicating the state of a slot:
// * If a message has been written into the slot, `WRITE` is set.
// * If a message has been read from the slot, `READ` is set.
// * If the block is being destroyed, `DESTROY` is set.
const WRITE: usize = 1;
const READ: usize = 2;
const DESTROY: usize = 4;

// Each block covers one "lap" of indices.
const LAP: usize = 32;
// The maximum number of messages a block can hold.
const BLOCK_CAP: usize = LAP - 1;
// How many lower bits are reserved for metadata.
const SHIFT: usize = 1;
// Has two different purposes:
// * If set in head, indicates that the block is not the last one.
// * If set in tail, indicates that the channel is disconnected.
const MARK_BIT: usize = 1;

/// A slot in a block.
struct Slot<T> {
    /// The message.
    msg: UnsafeCell<MaybeUninit<T>>,

    /// The state of the slot.
    state: AtomicUsize,
}

impl<T> Slot<T> {
    const UNINIT: Self =
        Self { msg: UnsafeCell::new(MaybeUninit::uninit()), state: AtomicUsize::new(0) };

    /// Blocks until a message is written into the slot.
    fn wait_write(&self, receivers: &SyncWaker, token: &mut Token) {
        watch_until(receivers, token, || {
            (self.state.load(Ordering::SeqCst) & WRITE != 0).then_some(())
        })
    }
}

/// A block in a linked list.
///
/// Each block in the list can hold up to `BLOCK_CAP` messages.
struct Block<T> {
    /// The next block in the linked list.
    next: AtomicPtr<Block<T>>,

    /// Slots for messages.
    slots: [Slot<T>; BLOCK_CAP],
}

impl<T> Block<T> {
    /// Creates an empty block.
    fn new() -> Block<T> {
        Block { next: AtomicPtr::new(ptr::null_mut()), slots: [Slot::UNINIT; BLOCK_CAP] }
    }

    /// Blocks until the next pointer is set.
    fn wait_next(&self, receivers: &SyncWaker, token: &mut Token) -> *mut Self {
        watch_until(receivers, token, || NonNull::new(self.next.load(Ordering::SeqCst))).as_ptr()
    }

    /// Sets the `DESTROY` bit in slots starting from `start` and destroys the block.
    unsafe fn destroy(this: *mut Block<T>, start: usize) {
        // It is not necessary to set the `DESTROY` bit in the last slot because that slot has
        // begun destruction of the block.
        for i in start..BLOCK_CAP - 1 {
            let slot = unsafe { (*this).slots.get_unchecked(i) };

            // Mark the `DESTROY` bit if a thread is still using the slot.
            if slot.state.load(Ordering::Acquire) & READ == 0
                && slot.state.fetch_or(DESTROY, Ordering::AcqRel) & READ == 0
            {
                // If a thread is still using the slot, it will continue destruction of the block.
                return;
            }
        }

        // No thread is using the block, now it is safe to destroy it.
        drop(unsafe { Box::from_raw(this) });
    }
}

/// A position in a channel.
#[derive(Debug)]
struct Position<T> {
    /// The index in the channel.
    index: AtomicUsize,

    /// The block in the linked list.
    block: AtomicPtr<Block<T>>,
}

/// The token type for the list flavor.
#[derive(Debug)]
pub(crate) struct ListToken {
    /// The block of slots.
    block: *const u8,

    /// The offset into the block.
    offset: usize,
}

impl Default for ListToken {
    #[inline]
    fn default() -> Self {
        ListToken { block: ptr::null(), offset: 0 }
    }
}

/// Unbounded channel implemented as a linked list.
///
/// Each message sent into the channel is assigned a sequence number, i.e. an index. Indices are
/// represented as numbers of type `usize` and wrap on overflow.
///
/// Consecutive messages are grouped into blocks in order to put less pressure on the allocator and
/// improve cache efficiency.
pub(crate) struct Channel<T> {
    /// The head of the channel.
    head: CachePadded<Position<T>>,

    /// The tail of the channel.
    tail: CachePadded<Position<T>>,

    /// Receivers waiting while the channel is empty and not disconnected.
    receivers: SyncWaker,

    /// Indicates that dropping a `Channel<T>` may drop messages of type `T`.
    _marker: PhantomData<T>,
}

/// The status of the channel after calling `start_recv`.
#[derive(PartialEq, Eq)]
enum Status {
    /// The channel has a message ready to read.
    Ready,
    /// There is currently a send in progress holding up the queue.
    /// All operations must block to preserve linearizability.
    InProgress,
    /// The channel is empty.
    Empty,
}

impl<T> Channel<T> {
    /// Creates a new unbounded channel.
    pub(crate) fn new() -> Self {
        Channel {
            head: CachePadded::new(Position {
                block: AtomicPtr::new(ptr::null_mut()),
                index: AtomicUsize::new(0),
            }),
            tail: CachePadded::new(Position {
                block: AtomicPtr::new(ptr::null_mut()),
                index: AtomicUsize::new(0),
            }),
            receivers: SyncWaker::new(),
            _marker: PhantomData,
        }
    }

    /// Attempts to reserve a slot for sending a message.
    fn start_send(&self, token: &mut Token) -> Status {
        let backoff = Backoff::new();
        let mut tail = self.tail.index.load(Ordering::Acquire);
        let mut block = self.tail.block.load(Ordering::Acquire);
        let mut next_block = None;

        loop {
            // Check if the channel is disconnected.
            if tail & MARK_BIT != 0 {
                token.list.block = ptr::null();
                return Status::Ready;
            }

            // Calculate the offset of the index into the block.
            let offset = (tail >> SHIFT) % LAP;

            // If we reached the end of the block, wait until the next one is installed.
            // If we've been waiting for too long, fallback to blocking.
            if offset == BLOCK_CAP {
                if !backoff.try_spin_light() {
                    return Status::InProgress;
                }

                tail = self.tail.index.load(Ordering::Acquire);
                block = self.tail.block.load(Ordering::Acquire);
                continue;
            }

            // If we're going to have to install the next block, allocate it in advance in order to
            // make the wait for other threads as short as possible.
            if offset + 1 == BLOCK_CAP && next_block.is_none() {
                next_block = Some(Box::new(Block::<T>::new()));
            }

            // If this is the first message to be sent into the channel, we need to allocate the
            // first block and install it.
            if block.is_null() {
                let new = Box::into_raw(Box::new(Block::<T>::new()));

                if self
                    .tail
                    .block
                    .compare_exchange(block, new, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    self.head.block.store(new, Ordering::Release);
                    block = new;
                } else {
                    next_block = unsafe { Some(Box::from_raw(new)) };
                    tail = self.tail.index.load(Ordering::Acquire);
                    block = self.tail.block.load(Ordering::Acquire);
                    continue;
                }
            }

            let new_tail = tail + (1 << SHIFT);

            // Try advancing the tail forward.
            match self.tail.index.compare_exchange_weak(
                tail,
                new_tail,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => unsafe {
                    // If we've reached the end of the block, install the next one.
                    if offset + 1 == BLOCK_CAP {
                        let next_block = Box::into_raw(next_block.unwrap());
                        self.tail.block.store(next_block, Ordering::Release);
                        self.tail.index.fetch_add(1 << SHIFT, Ordering::Release);
                        (*block).next.store(next_block, Ordering::SeqCst);
                    }

                    token.list.block = block as *const u8;
                    token.list.offset = offset;
                    return Status::Ready;
                },
                Err(_) => {
                    backoff.spin_light();
                    tail = self.tail.index.load(Ordering::Acquire);
                    block = self.tail.block.load(Ordering::Acquire);
                }
            }
        }
    }

    /// Writes a message into the channel.
    pub(crate) unsafe fn write(&self, token: &mut Token, msg: T) -> Result<(), T> {
        // If there is no slot, the channel is disconnected.
        if token.list.block.is_null() {
            return Err(msg);
        }

        // Write the message into the slot.
        let block = token.list.block.cast::<Block<T>>();
        let offset = token.list.offset;
        let slot = unsafe { (*block).slots.get_unchecked(offset) };
        unsafe { slot.msg.get().write(MaybeUninit::new(msg)) }
        slot.state.fetch_or(WRITE, Ordering::SeqCst);

        // Wake a sleeping receiver.
        self.receivers.notify();
        Ok(())
    }

    /// Attempts to reserve a slot for receiving a message.
    fn start_recv(&self, token: &mut Token) -> Status {
        let backoff = Backoff::new();
        let mut head = self.head.index.load(Ordering::Acquire);
        let mut block = self.head.block.load(Ordering::Acquire);

        loop {
            // Calculate the offset of the index into the block.
            let offset = (head >> SHIFT) % LAP;

            // We reached the end of the block but the block is not installed yet, meaning
            // the last send on the previous block is still in progress. The send is likely to
            // complete soon so we spin here before falling back to blocking.
            if offset == BLOCK_CAP {
                if !backoff.try_spin_light() {
                    return Status::InProgress;
                }

                head = self.head.index.load(Ordering::Acquire);
                block = self.head.block.load(Ordering::Acquire);
                continue;
            }

            let mut new_head = head + (1 << SHIFT);

            if new_head & MARK_BIT == 0 {
                atomic::fence(Ordering::SeqCst);
                let tail = self.tail.index.load(Ordering::Relaxed);

                // If the tail equals the head, that means the channel is empty.
                if head >> SHIFT == tail >> SHIFT {
                    // If the channel is disconnected...
                    if tail & MARK_BIT != 0 {
                        // ...then receive an error.
                        token.list.block = ptr::null();
                        return Status::Ready;
                    } else {
                        // Otherwise, the receive operation is not ready.
                        return Status::Empty;
                    }
                }

                // If head and tail are not in the same block, set `MARK_BIT` in head.
                if (head >> SHIFT) / LAP != (tail >> SHIFT) / LAP {
                    new_head |= MARK_BIT;
                }
            }

            // The block can be null here only if the first message sent into the channel is
            // in progress. The send is likely to complete soon so we spin here before falling
            // back to blocking.
            if block.is_null() {
                if !backoff.try_spin_light() {
                    return Status::InProgress;
                }

                head = self.head.index.load(Ordering::Acquire);
                block = self.head.block.load(Ordering::Acquire);
                continue;
            }

            // Try moving the head index forward.
            match self.head.index.compare_exchange_weak(
                head,
                new_head,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => unsafe {
                    // If we've reached the end of the block, move to the next one.
                    if offset + 1 == BLOCK_CAP {
                        let next = (*block).wait_next(&self.receivers, token);
                        let mut next_index = (new_head & !MARK_BIT).wrapping_add(1 << SHIFT);
                        if !(*next).next.load(Ordering::Relaxed).is_null() {
                            next_index |= MARK_BIT;
                        }

                        self.head.block.store(next, Ordering::Release);
                        self.head.index.store(next_index, Ordering::Release);
                    }

                    token.list.block = block as *const u8;
                    token.list.offset = offset;
                    return Status::Ready;
                },
                Err(_) => {
                    backoff.spin_light();
                    head = self.head.index.load(Ordering::Acquire);
                    block = self.head.block.load(Ordering::Acquire);
                }
            }
        }
    }

    /// Reads a message from the channel.
    pub(crate) unsafe fn read(&self, token: &mut Token) -> Result<T, ()> {
        if token.list.block.is_null() {
            // The channel is disconnected.
            return Err(());
        }

        // Read the message.
        let block = token.list.block as *mut Block<T>;
        let offset = token.list.offset;
        let slot = unsafe { (*block).slots.get_unchecked(offset) };
        slot.wait_write(&self.receivers, token);
        let msg = unsafe { slot.msg.get().read().assume_init() };

        // Destroy the block if we've reached the end, or if another thread wanted to destroy but
        // couldn't because we were busy reading from the slot.
        unsafe {
            if offset + 1 == BLOCK_CAP {
                Block::destroy(block, 0);
            } else if slot.state.fetch_or(READ, Ordering::AcqRel) & DESTROY != 0 {
                Block::destroy(block, offset + 1);
            }
        }

        Ok(msg)
    }

    /// Attempts to send a message into the channel.
    pub(crate) fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        self.send(msg, None).map_err(|err| match err {
            SendTimeoutError::Disconnected(msg) => TrySendError::Disconnected(msg),
            SendTimeoutError::Timeout(_) => unreachable!(),
        })
    }

    /// Sends a message into the channel.
    pub(crate) fn send(
        &self,
        msg: T,
        _deadline: Option<Instant>,
    ) -> Result<(), SendTimeoutError<T>> {
        let token = &mut Token::default();

        // It's possible that we can't proceed because of the sender that
        // is supposed to install the next block lagging, so we might have to
        // block for a message to be sent.
        let mut state = self.receivers.start();
        let mut started = false;
        loop {
            // Try sending a message several times.
            let backoff = Backoff::new();
            loop {
                if started || self.start_send(token) == Status::Ready {
                    return unsafe {
                        self.write(token, msg).map_err(SendTimeoutError::Disconnected)
                    };
                }

                // Spin for a bit before blocking.
                if !backoff.try_spin_light() {
                    break;
                }
            }

            // Prepare for blocking until a sender wakes us up.
            Context::with(|cx| {
                // Register to be notified after any message is sent.
                let oper = Operation::hook(token);
                self.receivers.watch(oper, cx, &state);

                // Has the channel become ready just now?
                if self.start_send(token) == Status::Ready {
                    let _ = cx.try_select(Selected::Aborted);
                    started = true;
                }

                // Block the current thread.
                let sel = cx.wait_until(None);

                match sel {
                    Selected::Waiting => unreachable!(),
                    Selected::Aborted | Selected::Disconnected => {
                        self.receivers.unwatch(oper);
                    }
                    Selected::Operation(_) => {}
                }

                state.unpark();
            });
        }
    }

    /// Attempts to receive a message without blocking.
    pub(crate) fn try_recv(&self) -> Result<T, TryRecvError> {
        match self.recv_blocking(None, false) {
            Ok(Some(value)) => Ok(value),
            Ok(None) => Err(TryRecvError::Empty),
            Err(RecvTimeoutError::Disconnected) => Err(TryRecvError::Disconnected),
            Err(RecvTimeoutError::Timeout) => {
                unreachable!("called recv_blocking with deadline: None")
            }
        }
    }

    /// Receives a message from the channel.
    pub(crate) fn recv(&self, deadline: Option<Instant>) -> Result<T, RecvTimeoutError> {
        self.recv_blocking(deadline, true)
            .map(|value| value.expect("called recv_blocking with block: true"))
    }

    /// Receives a message from the channel.
    ///
    /// Blocks until a message is received if `should_block` is `true`. Otherwise, returns `Ok(None)` if
    /// the channel is full.
    ///
    /// Note this may still block when `should_block` is `false` if the channel is in an inconsistent state.
    pub(crate) fn recv_blocking(
        &self,
        deadline: Option<Instant>,
        should_block: bool,
    ) -> Result<Option<T>, RecvTimeoutError> {
        let token = &mut Token::default();

        let mut state = self.receivers.start();
        loop {
            // Try receiving a message several times.
            let backoff = Backoff::new();
            loop {
                match self.start_recv(token) {
                    Status::Ready => {
                        let res = unsafe { self.read(token) };
                        return res.map(Some).map_err(|_| RecvTimeoutError::Disconnected);
                    }
                    // If the channel is empty, return or block immediately.
                    Status::Empty if !should_block => return Ok(None),
                    Status::Empty => break,
                    // Otherwise spin for a bit before blocking.
                    Status::InProgress => {}
                }

                if !backoff.try_spin_light() {
                    break;
                }
            }

            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Err(RecvTimeoutError::Timeout);
                }
            }

            // Prepare for blocking until a sender wakes us up.
            Context::with(|cx| {
                let oper = Operation::hook(token);
                self.receivers.register(oper, cx, &state);

                // Has the channel become ready just now?
                if !self.is_empty() || self.is_disconnected() {
                    let _ = cx.try_select(Selected::Aborted);
                }

                // Block the current thread.
                let sel = cx.wait_until(deadline);

                match sel {
                    Selected::Waiting => unreachable!(),
                    Selected::Aborted | Selected::Disconnected => {
                        self.receivers.unregister(oper).unwrap();
                        // If the channel was disconnected, we still have to check for remaining
                        // messages.
                    }
                    Selected::Operation(_) => {}
                }

                state.unpark();
            });
        }
    }

    /// Returns the current number of messages inside the channel.
    pub(crate) fn len(&self) -> usize {
        loop {
            // Load the tail index, then load the head index.
            let mut tail = self.tail.index.load(Ordering::SeqCst);
            let mut head = self.head.index.load(Ordering::SeqCst);

            // If the tail index didn't change, we've got consistent indices to work with.
            if self.tail.index.load(Ordering::SeqCst) == tail {
                // Erase the lower bits.
                tail &= !((1 << SHIFT) - 1);
                head &= !((1 << SHIFT) - 1);

                // Fix up indices if they fall onto block ends.
                if (tail >> SHIFT) & (LAP - 1) == LAP - 1 {
                    tail = tail.wrapping_add(1 << SHIFT);
                }
                if (head >> SHIFT) & (LAP - 1) == LAP - 1 {
                    head = head.wrapping_add(1 << SHIFT);
                }

                // Rotate indices so that head falls into the first block.
                let lap = (head >> SHIFT) / LAP;
                tail = tail.wrapping_sub((lap * LAP) << SHIFT);
                head = head.wrapping_sub((lap * LAP) << SHIFT);

                // Remove the lower bits.
                tail >>= SHIFT;
                head >>= SHIFT;

                // Return the difference minus the number of blocks between tail and head.
                return tail - head - tail / LAP;
            }
        }
    }

    /// Returns the capacity of the channel.
    pub(crate) fn capacity(&self) -> Option<usize> {
        None
    }

    /// Disconnects senders and wakes up all blocked receivers.
    ///
    /// Returns `true` if this call disconnected the channel.
    pub(crate) fn disconnect_senders(&self) -> bool {
        let tail = self.tail.index.fetch_or(MARK_BIT, Ordering::SeqCst);

        if tail & MARK_BIT == 0 {
            self.receivers.disconnect();
            true
        } else {
            false
        }
    }

    /// Disconnects receivers.
    ///
    /// Returns `true` if this call disconnected the channel.
    pub(crate) fn disconnect_receivers(&self) -> bool {
        let tail = self.tail.index.fetch_or(MARK_BIT, Ordering::SeqCst);

        if tail & MARK_BIT == 0 {
            // If receivers are dropped first, discard all messages to free
            // memory eagerly.
            self.discard_all_messages();
            true
        } else {
            false
        }
    }

    /// Discards all messages.
    ///
    /// This method should only be called when all receivers are dropped.
    fn discard_all_messages(&self) {
        let token = &mut Token::default();
        let backoff = Backoff::new();
        let mut tail = self.tail.index.load(Ordering::Acquire);
        loop {
            let offset = (tail >> SHIFT) % LAP;
            if offset != BLOCK_CAP {
                break;
            }

            // New updates to tail will be rejected by MARK_BIT and aborted unless it's
            // at boundary. We need to wait for the updates take affect otherwise there
            // can be memory leaks.
            backoff.spin_heavy();
            tail = self.tail.index.load(Ordering::Acquire);
        }

        let mut head = self.head.index.load(Ordering::Acquire);
        // The channel may be uninitialized, so we have to swap to avoid overwriting any sender's attempts
        // to initialize the first block before noticing that the receivers disconnected. Late allocations
        // will be deallocated by the sender in Drop.
        let mut block = self.head.block.swap(ptr::null_mut(), Ordering::AcqRel);

        // If we're going to be dropping messages we need to synchronize with initialization
        if head >> SHIFT != tail >> SHIFT {
            // The block can be null here only if a sender is in the process of initializing the
            // channel while another sender managed to send a message by inserting it into the
            // semi-initialized channel and advanced the tail.
            // In that case, just wait until it gets initialized.
            while block.is_null() {
                backoff.spin_heavy();
                block = self.head.block.load(Ordering::Acquire);
            }
        }

        unsafe {
            // Drop all messages between head and tail and deallocate the heap-allocated blocks.
            while head >> SHIFT != tail >> SHIFT {
                let offset = (head >> SHIFT) % LAP;

                if offset < BLOCK_CAP {
                    // Drop the message in the slot.
                    let slot = (*block).slots.get_unchecked(offset);
                    slot.wait_write(&self.receivers, token);
                    (*slot.msg.get()).assume_init_drop();
                } else {
                    (*block).wait_next(&self.receivers, token);
                    // Deallocate the block and move to the next one.
                    let next = (*block).next.load(Ordering::Acquire);
                    drop(Box::from_raw(block));
                    block = next;
                }

                head = head.wrapping_add(1 << SHIFT);
            }

            // Deallocate the last remaining block.
            if !block.is_null() {
                drop(Box::from_raw(block));
            }
        }

        head &= !MARK_BIT;
        self.head.index.store(head, Ordering::Release);
    }

    /// Returns `true` if the channel is disconnected.
    pub(crate) fn is_disconnected(&self) -> bool {
        self.tail.index.load(Ordering::SeqCst) & MARK_BIT != 0
    }

    /// Returns `true` if the channel is empty.
    pub(crate) fn is_empty(&self) -> bool {
        let head = self.head.index.load(Ordering::SeqCst);
        let tail = self.tail.index.load(Ordering::SeqCst);
        head >> SHIFT == tail >> SHIFT
    }

    /// Returns `true` if the channel is full.
    pub(crate) fn is_full(&self) -> bool {
        false
    }
}

/// Blocks until a read operation succeeds, returning the value once it does.
///
/// Note that the read and corresponding store must be use `SeqCst` ordering to
/// synchronize with notifications.
fn watch_until<T>(receivers: &SyncWaker, token: &mut Token, try_read: impl Fn() -> Option<T>) -> T {
    let mut state = receivers.start();
    let mut value = None;

    loop {
        // Try reading several times.
        let backoff = Backoff::new();
        loop {
            if value.is_none() {
                value = try_read();
            }

            if let Some(value) = value {
                return value;
            }

            if !backoff.try_spin_light() {
                break;
            }
        }

        // Prepare for blocking until a sender wakes us up.
        Context::with(|cx| {
            // Register to be notified after any message is sent.
            let oper = Operation::hook(token);
            receivers.watch(oper, cx, &state);

            // Was the message just sent?
            if let Some(read) = try_read() {
                value = Some(read);
                let _ = cx.try_select(Selected::Aborted);
            }

            // Block the current thread.
            let sel = cx.wait_until(None);

            match sel {
                Selected::Waiting => unreachable!(),
                Selected::Aborted | Selected::Disconnected => {
                    receivers.unwatch(oper);
                }
                Selected::Operation(_) => {}
            }

            state.unpark();
        });
    }
}

impl<T> Drop for Channel<T> {
    fn drop(&mut self) {
        let mut head = *self.head.index.get_mut();
        let mut tail = *self.tail.index.get_mut();
        let mut block = *self.head.block.get_mut();

        // Erase the lower bits.
        head &= !((1 << SHIFT) - 1);
        tail &= !((1 << SHIFT) - 1);

        unsafe {
            // Drop all messages between head and tail and deallocate the heap-allocated blocks.
            while head != tail {
                let offset = (head >> SHIFT) % LAP;

                if offset < BLOCK_CAP {
                    // Drop the message in the slot.
                    let slot = (*block).slots.get_unchecked(offset);
                    (*slot.msg.get()).assume_init_drop();
                } else {
                    // Deallocate the block and move to the next one.
                    let next = *(*block).next.get_mut();
                    drop(Box::from_raw(block));
                    block = next;
                }

                head = head.wrapping_add(1 << SHIFT);
            }

            // Deallocate the last remaining block.
            if !block.is_null() {
                drop(Box::from_raw(block));
            }
        }
    }
}
