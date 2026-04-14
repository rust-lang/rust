//! Inbox: minimal kernel-backed queue primitive for typed message delivery.
//!
//! # Design
//!
//! An [`Inbox`] is a first-class kernel object that holds a bounded FIFO queue
//! of [`MessageEnvelope`] values.  It is the canonical receive-side primitive
//! for kernel-mediated message delivery in ThingOS.
//!
//! ## Ownership and lifetime
//!
//! Inboxes are reference-counted via [`alloc::sync::Arc`].  A global registry
//! assigns each inbox a unique [`InboxId`]; callers obtain an [`Arc<Inbox>`]
//! handle through [`get_inbox`].  When the last external reference is dropped
//! **and** the registry entry has been removed by [`close_inbox`], the inbox
//! is destroyed and all queued messages are dropped.
//!
//! ## Queue semantics
//!
//! - **FIFO ordering**: messages are dequeued in the order they were enqueued.
//! - **Bounded capacity**: enqueue into a full inbox returns
//!   [`SendError::Full`] instead of silently dropping.
//! - **Atomic whole-message enqueue**: each [`MessageEnvelope`] is written as
//!   one indivisible unit; no partial writes.
//! - **Closed state**: once [`Inbox::close`] is called, all subsequent sends
//!   return [`SendError::Closed`] and all waiting receivers are woken.
//!
//! ## Receive semantics
//!
//! - [`Inbox::try_recv`]: non-blocking; returns `Ok(None)` on empty.
//! - Blocking receive requires the caller to park the current task on
//!   [`Inbox::waiters`] and call [`Inbox::try_recv`] after waking.
//!   This is intentionally left to the scheduling layer to avoid baking
//!   scheduler policy into the queue primitive.
//!
//! ## Wait / readiness integration
//!
//! [`Inbox`] exposes a public [`WaitQueue`] field (`waiters`) so that
//! higher-level code can register tasks for wakeup on message arrival without
//! needing scheduler policy inside the primitive itself.  [`Inbox::send`]
//! calls [`WaitQueue::wake_one`] after a successful enqueue.
//!
//! ## What this is *not*
//!
//! - Not a router — the Inbox does not decide who should receive a message.
//! - Not a signal replacement — signal semantics live in `kernel::signal`.
//! - Not a pipe — this is message-oriented, not byte-stream-oriented.
//! - Not a priority queue — FIFO order only.
//!
//! # Follow-on tasks enabled by this primitive
//!
//! - Typed direct send to an Inbox
//! - JobExit delivery via Inbox
//! - Signal-routing handoff into Inbox or recipient hook
//! - Group broadcast fanout into multiple Inboxes
//! - Unified `DeliveryTarget` abstraction above Inbox

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::message::Message;
use crate::sched::WaitQueue;

// ---------------------------------------------------------------------------
// InboxId
// ---------------------------------------------------------------------------

/// Opaque unique identifier for an [`Inbox`] in the global registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InboxId(pub u64);

// ---------------------------------------------------------------------------
// MessageEnvelope
// ---------------------------------------------------------------------------

/// The canonical typed unit stored in an [`Inbox`] queue.
///
/// Every message delivered through the inbox system is wrapped in an envelope
/// that carries the typed payload together with minimal delivery metadata.
/// The `message` field holds the canonical [`Message`] (kind + opaque bytes);
/// `sender` optionally identifies the sending thread.
///
/// This type is deliberately kept small.  Routing metadata (target group,
/// delivery strategy, sequence number) belongs to layers above the inbox
/// primitive and should not leak into the envelope at this level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageEnvelope {
    /// The typed message to deliver.
    pub message: Message,
    /// TID of the sending task at enqueue time, if known.
    pub sender: Option<u64>,
}

impl MessageEnvelope {
    /// Construct an envelope with a known sender TID.
    pub fn with_sender(message: Message, sender_tid: u64) -> Self {
        Self {
            message,
            sender: Some(sender_tid),
        }
    }

    /// Construct an anonymous envelope (no sender attribution).
    pub fn anonymous(message: Message) -> Self {
        Self {
            message,
            sender: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SendError / RecvError
// ---------------------------------------------------------------------------

/// Error returned when enqueue into an [`Inbox`] fails.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SendError {
    /// The inbox has reached its bounded capacity.
    Full {
        /// Maximum number of messages the inbox can hold.
        capacity: usize,
    },
    /// The inbox has been closed; no further messages can be delivered.
    Closed,
}

/// Error returned when a dequeue operation on an [`Inbox`] fails.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecvError {
    /// The inbox has been closed and all previously queued messages have
    /// already been consumed.
    Closed,
}

// ---------------------------------------------------------------------------
// Inbox internals
// ---------------------------------------------------------------------------

struct InboxInner {
    queue: VecDeque<MessageEnvelope>,
    capacity: usize,
    closed: bool,
}

impl InboxInner {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            queue: VecDeque::with_capacity(capacity.min(64)),
            capacity,
            closed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Inbox
// ---------------------------------------------------------------------------

/// Minimal kernel-backed queue for typed [`MessageEnvelope`] delivery.
///
/// See the [module-level documentation](self) for design rationale and
/// usage guidelines.
pub struct Inbox {
    inner: Mutex<InboxInner>,
    /// Wait queue for tasks blocked waiting for a message.
    ///
    /// Exposed publicly so that the scheduling layer can register and dequeue
    /// waiters without the inbox primitive needing to know about scheduler
    /// policy.  [`Inbox::send`] calls [`WaitQueue::wake_one`] on success.
    pub waiters: WaitQueue,
}

impl Inbox {
    /// Create a new open inbox with the given bounded capacity.
    ///
    /// `capacity` is the maximum number of [`MessageEnvelope`] values the
    /// inbox can hold at one time.  A value of 0 is silently raised to 1.
    ///
    /// # Default capacity
    ///
    /// Callers that do not have a specific capacity requirement should use
    /// [`DEFAULT_INBOX_CAPACITY`].
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(InboxInner::new(capacity)),
            waiters: WaitQueue::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Core queue operations
    // -----------------------------------------------------------------------

    /// Enqueue one [`MessageEnvelope`] into the inbox.
    ///
    /// On success the function returns `Ok(())` and wakes one waiter (if any)
    /// via the internal [`WaitQueue`].
    ///
    /// # Errors
    ///
    /// - [`SendError::Full`]   — inbox is at capacity.
    /// - [`SendError::Closed`] — inbox has been closed.
    pub fn send(&self, envelope: MessageEnvelope) -> Result<(), SendError> {
        {
            let mut inner = self.inner.lock();
            if inner.closed {
                return Err(SendError::Closed);
            }
            if inner.queue.len() >= inner.capacity {
                return Err(SendError::Full {
                    capacity: inner.capacity,
                });
            }
            inner.queue.push_back(envelope);
            crate::kdebug!(
                "inbox::send: enqueued message, depth={}",
                inner.queue.len()
            );
        }
        // Wake one waiter outside the lock to avoid priority inversion.
        self.waiters.wake_one();
        Ok(())
    }

    /// Attempt to dequeue the next [`MessageEnvelope`] without blocking.
    ///
    /// Returns:
    /// - `Ok(Some(envelope))` — a message was available and has been removed.
    /// - `Ok(None)`           — the inbox is empty (but still open).
    /// - `Err(RecvError::Closed)` — the inbox is closed **and** empty.
    ///
    /// A closed inbox that still holds queued messages returns `Ok(Some(…))`
    /// until all messages are consumed; only then does it return
    /// `Err(RecvError::Closed)`.
    pub fn try_recv(&self) -> Result<Option<MessageEnvelope>, RecvError> {
        let mut inner = self.inner.lock();
        if let Some(envelope) = inner.queue.pop_front() {
            crate::kdebug!(
                "inbox::try_recv: dequeued message, depth={}",
                inner.queue.len()
            );
            return Ok(Some(envelope));
        }
        if inner.closed {
            return Err(RecvError::Closed);
        }
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Close the inbox.
    ///
    /// After this call:
    /// - All subsequent [`send`](Inbox::send) calls return
    ///   [`SendError::Closed`].
    /// - [`try_recv`](Inbox::try_recv) continues to drain any messages still
    ///   queued at close time; once the queue is empty it returns
    ///   [`Err(RecvError::Closed)`].
    /// - All tasks currently registered on the internal [`WaitQueue`] are
    ///   woken so they can observe the closed state.
    ///
    /// Closing an already-closed inbox is a no-op.
    pub fn close(&self) {
        {
            let mut inner = self.inner.lock();
            if inner.closed {
                return;
            }
            inner.closed = true;
            crate::kdebug!(
                "inbox::close: inbox closed, {} messages dropped on close",
                inner.queue.len()
            );
        }
        // Wake all waiters so they can observe the closed state.
        self.waiters.wake_all();
    }

    // -----------------------------------------------------------------------
    // Diagnostics / readiness queries
    // -----------------------------------------------------------------------

    /// Current number of messages queued.
    pub fn len(&self) -> usize {
        self.inner.lock().queue.len()
    }

    /// Returns `true` if the queue contains no messages.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().queue.is_empty()
    }

    /// Returns `true` if the queue has reached its bounded capacity.
    pub fn is_full(&self) -> bool {
        let inner = self.inner.lock();
        inner.queue.len() >= inner.capacity
    }

    /// Returns `true` if the inbox has been closed.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().closed
    }

    /// The maximum number of messages the inbox can hold.
    pub fn capacity(&self) -> usize {
        self.inner.lock().capacity
    }
}

// ---------------------------------------------------------------------------
// Default capacity
// ---------------------------------------------------------------------------

/// Default bounded capacity for a new [`Inbox`].
///
/// Chosen to match the existing prototype inbox capacity used by process
/// message delivery, providing a predictable default without over-allocating.
pub const DEFAULT_INBOX_CAPACITY: usize = 64;

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------

/// Global registry of live [`Inbox`] objects.
///
/// Entries are `Option<Arc<Inbox>>` so that slots can be recycled when an
/// inbox is removed.  The [`InboxId`] is a stable index into this table for
/// the lifetime of the inbox.
static INBOX_REGISTRY: Mutex<Vec<Option<Arc<Inbox>>>> = Mutex::new(Vec::new());

/// Create a new [`Inbox`] with the given capacity and register it globally.
///
/// Returns the stable [`InboxId`] that can later be passed to [`get_inbox`]
/// or [`close_inbox`].
pub fn create_inbox(capacity: usize) -> InboxId {
    let inbox = Arc::new(Inbox::new(capacity));
    let mut registry = INBOX_REGISTRY.lock();

    // Reuse a free slot if one is available.
    for (i, slot) in registry.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(inbox);
            return InboxId(i as u64);
        }
    }

    // No free slot; append.
    let id = registry.len() as u64;
    registry.push(Some(inbox));
    InboxId(id)
}

/// Look up a live [`Inbox`] by its [`InboxId`].
///
/// Returns `None` if the id is out of range or the inbox has already been
/// removed from the registry via [`close_inbox`].
pub fn get_inbox(id: InboxId) -> Option<Arc<Inbox>> {
    let registry = INBOX_REGISTRY.lock();
    registry
        .get(id.0 as usize)
        .and_then(|slot| slot.clone())
}

/// Close and remove an [`Inbox`] from the global registry.
///
/// The inbox is first closed (all waiters woken, further sends rejected), then
/// the registry entry is cleared.  If the caller holds additional
/// [`Arc<Inbox>`] references the underlying object lives until those are
/// dropped.
///
/// Removing an already-absent entry is a no-op.
pub fn close_inbox(id: InboxId) {
    let inbox = {
        let mut registry = INBOX_REGISTRY.lock();
        if let Some(slot) = registry.get_mut(id.0 as usize) {
            slot.take()
        } else {
            None
        }
    };
    if let Some(inbox) = inbox {
        inbox.close();
    }
}

/// Number of live inboxes currently in the registry (diagnostics).
pub fn inbox_count() -> usize {
    INBOX_REGISTRY.lock().iter().filter(|s| s.is_some()).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::KindId;
    use crate::sched::blocking::WAKE_TASK_HOOK;
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_kind(byte: u8) -> KindId {
        KindId([byte; 16])
    }

    fn make_envelope(byte: u8) -> MessageEnvelope {
        MessageEnvelope::anonymous(Message::new(make_kind(byte), alloc::vec![byte]))
    }

    fn make_envelope_with_sender(byte: u8, tid: u64) -> MessageEnvelope {
        MessageEnvelope::with_sender(Message::new(make_kind(byte), alloc::vec![byte]), tid)
    }

    // -----------------------------------------------------------------------
    // Wake-recording infrastructure (mirrors wait_queue.rs tests)
    // -----------------------------------------------------------------------

    static WOKEN_IDS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    static WOKEN_LEN: AtomicUsize = AtomicUsize::new(0);

    fn reset_wakes() {
        WOKEN_LEN.store(0, Ordering::SeqCst);
        for slot in &WOKEN_IDS {
            slot.store(0, Ordering::SeqCst);
        }
    }

    fn record_wake(id: u64) {
        let idx = WOKEN_LEN.fetch_add(1, Ordering::SeqCst);
        if idx < WOKEN_IDS.len() {
            WOKEN_IDS[idx].store(id, Ordering::SeqCst);
        }
    }

    fn wake_log() -> alloc::vec::Vec<u64> {
        let len = WOKEN_LEN.load(Ordering::SeqCst).min(WOKEN_IDS.len());
        (0..len)
            .map(|idx| WOKEN_IDS[idx].load(Ordering::SeqCst))
            .collect()
    }

    // -----------------------------------------------------------------------
    // MessageEnvelope construction
    // -----------------------------------------------------------------------

    #[test]
    fn envelope_anonymous_has_no_sender() {
        let msg = Message::new(make_kind(0x01), alloc::vec![1, 2, 3]);
        let env = MessageEnvelope::anonymous(msg.clone());
        assert_eq!(env.sender, None);
        assert_eq!(env.message, msg);
    }

    #[test]
    fn envelope_with_sender_records_tid() {
        let msg = Message::new(make_kind(0x02), alloc::vec![]);
        let env = MessageEnvelope::with_sender(msg.clone(), 42);
        assert_eq!(env.sender, Some(42));
        assert_eq!(env.message, msg);
    }

    // -----------------------------------------------------------------------
    // Basic enqueue / dequeue
    // -----------------------------------------------------------------------

    #[test]
    fn basic_send_and_try_recv_round_trips_message() {
        let inbox = Inbox::new(8);
        let env = make_envelope(0xAA);
        assert!(inbox.send(env.clone()).is_ok());
        assert_eq!(inbox.try_recv().unwrap(), Some(env));
    }

    #[test]
    fn try_recv_on_empty_open_inbox_returns_none() {
        let inbox = Inbox::new(8);
        assert_eq!(inbox.try_recv().unwrap(), None);
    }

    #[test]
    fn multiple_sends_before_receive_all_succeed() {
        let inbox = Inbox::new(8);
        for i in 0u8..5 {
            assert!(inbox.send(make_envelope(i)).is_ok(), "send {} failed", i);
        }
        assert_eq!(inbox.len(), 5);
    }

    // -----------------------------------------------------------------------
    // FIFO ordering
    // -----------------------------------------------------------------------

    #[test]
    fn dequeue_preserves_fifo_order() {
        let inbox = Inbox::new(8);
        for i in 0u8..4 {
            inbox.send(make_envelope(i)).unwrap();
        }
        for i in 0u8..4 {
            let got = inbox.try_recv().unwrap().unwrap();
            assert_eq!(
                got.message.payload[0], i,
                "FIFO order violated at index {}",
                i
            );
        }
    }

    // -----------------------------------------------------------------------
    // Capacity / full inbox
    // -----------------------------------------------------------------------

    #[test]
    fn send_to_full_inbox_returns_full_error() {
        let inbox = Inbox::new(3);
        for i in 0u8..3 {
            inbox.send(make_envelope(i)).unwrap();
        }
        assert!(inbox.is_full());
        let result = inbox.send(make_envelope(99));
        assert_eq!(result, Err(SendError::Full { capacity: 3 }));
    }

    #[test]
    fn inbox_capacity_is_reported_correctly() {
        let inbox = Inbox::new(16);
        assert_eq!(inbox.capacity(), 16);
    }

    #[test]
    fn is_empty_and_is_full_reflect_queue_state() {
        let inbox = Inbox::new(2);
        assert!(inbox.is_empty());
        assert!(!inbox.is_full());

        inbox.send(make_envelope(1)).unwrap();
        assert!(!inbox.is_empty());
        assert!(!inbox.is_full());

        inbox.send(make_envelope(2)).unwrap();
        assert!(!inbox.is_empty());
        assert!(inbox.is_full());
    }

    // -----------------------------------------------------------------------
    // Closed inbox behavior
    // -----------------------------------------------------------------------

    #[test]
    fn send_to_closed_inbox_returns_closed_error() {
        let inbox = Inbox::new(8);
        inbox.close();
        assert_eq!(inbox.send(make_envelope(1)), Err(SendError::Closed));
    }

    #[test]
    fn try_recv_on_empty_closed_inbox_returns_closed_error() {
        let inbox = Inbox::new(8);
        inbox.close();
        assert_eq!(inbox.try_recv(), Err(RecvError::Closed));
    }

    #[test]
    fn try_recv_drains_messages_enqueued_before_close() {
        let inbox = Inbox::new(8);
        inbox.send(make_envelope(10)).unwrap();
        inbox.send(make_envelope(20)).unwrap();
        inbox.close();

        // Messages enqueued before close are still readable.
        assert_eq!(inbox.try_recv().unwrap().unwrap().message.payload[0], 10);
        assert_eq!(inbox.try_recv().unwrap().unwrap().message.payload[0], 20);
        // Once drained, closed state becomes visible.
        assert_eq!(inbox.try_recv(), Err(RecvError::Closed));
    }

    #[test]
    fn close_is_idempotent() {
        let inbox = Inbox::new(8);
        inbox.close();
        inbox.close(); // must not panic or double-wake
        assert!(inbox.is_closed());
    }

    #[test]
    fn is_closed_reflects_close_call() {
        let inbox = Inbox::new(8);
        assert!(!inbox.is_closed());
        inbox.close();
        assert!(inbox.is_closed());
    }

    // -----------------------------------------------------------------------
    // Atomic whole-message delivery
    // -----------------------------------------------------------------------

    #[test]
    fn each_dequeued_message_is_a_complete_envelope() {
        let inbox = Inbox::new(4);
        let env_a = make_envelope_with_sender(0xAA, 1001);
        let env_b = make_envelope_with_sender(0xBB, 1002);

        inbox.send(env_a.clone()).unwrap();
        inbox.send(env_b.clone()).unwrap();

        let got_a = inbox.try_recv().unwrap().unwrap();
        let got_b = inbox.try_recv().unwrap().unwrap();

        // Full envelope (message + sender) preserved intact
        assert_eq!(got_a, env_a);
        assert_eq!(got_b, env_b);
    }

    // -----------------------------------------------------------------------
    // Readiness / wake behavior
    // -----------------------------------------------------------------------

    #[test]
    fn send_wakes_one_registered_waiter() {
        reset_wakes();
        WAKE_TASK_HOOK.store(record_wake as *mut (), Ordering::SeqCst);

        let inbox = Inbox::new(8);
        inbox.waiters.push_back(77);
        inbox.send(make_envelope(1)).unwrap();

        assert_eq!(wake_log(), alloc::vec![77u64]);
    }

    #[test]
    fn close_wakes_all_registered_waiters() {
        reset_wakes();
        WAKE_TASK_HOOK.store(record_wake as *mut (), Ordering::SeqCst);

        let inbox = Inbox::new(8);
        inbox.waiters.push_back(10);
        inbox.waiters.push_back(11);
        inbox.waiters.push_back(12);
        inbox.close();

        let woken = wake_log();
        assert_eq!(woken.len(), 3);
        assert!(woken.contains(&10));
        assert!(woken.contains(&11));
        assert!(woken.contains(&12));
    }

    #[test]
    fn send_to_full_inbox_does_not_wake_waiter() {
        reset_wakes();
        WAKE_TASK_HOOK.store(record_wake as *mut (), Ordering::SeqCst);

        let inbox = Inbox::new(1);
        inbox.send(make_envelope(0)).unwrap(); // fill it
        inbox.waiters.push_back(99);

        let result = inbox.send(make_envelope(1)); // should fail
        assert!(result.is_err());

        // Waiter must not have been woken on a failed send
        assert!(wake_log().is_empty(), "waiter woken on failed send");
    }

    // -----------------------------------------------------------------------
    // Registry
    // -----------------------------------------------------------------------

    #[test]
    fn create_and_get_inbox_roundtrip() {
        let id = create_inbox(DEFAULT_INBOX_CAPACITY);
        let inbox = get_inbox(id);
        assert!(inbox.is_some());
        assert_eq!(inbox.unwrap().capacity(), DEFAULT_INBOX_CAPACITY);
        close_inbox(id);
    }

    #[test]
    fn get_inbox_returns_none_after_close_inbox() {
        let id = create_inbox(4);
        close_inbox(id);
        assert!(get_inbox(id).is_none());
    }

    #[test]
    fn close_inbox_marks_inbox_closed() {
        let id = create_inbox(4);
        let arc = get_inbox(id).expect("inbox should exist");
        close_inbox(id);
        // The Arc we retained should observe closed state.
        assert!(arc.is_closed());
    }

    #[test]
    fn get_inbox_returns_none_for_out_of_range_id() {
        let bogus = InboxId(u64::MAX);
        assert!(get_inbox(bogus).is_none());
    }
}
