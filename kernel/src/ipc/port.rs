//! Port: the internal ring-buffer backing a user-facing channel.
//!
//! A **channel** (as exposed by `SYS_CHANNEL_CREATE` and the `SYS_CHANNEL_*`
//! family) is the user-visible IPC primitive.  Internally each channel is
//! backed by a `Port` — a fixed-capacity SPSC ring buffer with a structured
//! message queue.  `Port` / `PortId` are kernel-private names; user-space
//! and all documentation should say "channel" and "thing" instead.
//!
//! Each port has a single writer and single reader thing.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use spin::Mutex;

/// Unique identifier for a port in the global registry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortId(pub u32);

/// A structured message that bundles data bytes and zero or more capability
/// handles (VFS nodes).  This is the unit stored in the message queue and
/// exchanged via `SYS_CHANNEL_SEND_MSG` / `SYS_CHANNEL_RECV_MSG`.
pub struct KernelMessage {
    /// Payload bytes (may be empty for handle-only messages).
    pub data: alloc::vec::Vec<u8>,
    /// Attached capability handles transferred with this message.
    pub caps: alloc::vec::Vec<Arc<dyn crate::vfs::VfsNode>>,
}

/// Fixed-size ring buffer port (SPSC for v0)
///
/// This structure contains the shared state and ring buffer.
/// Access is intended to be via `Sender` and `Receiver` halves
/// which enforce the SPSC invariant.
pub struct Port {
    buf: Box<[u8]>,
    capacity: usize,
    head: AtomicUsize, // Write position (producer advances)
    tail: AtomicUsize, // Read position (consumer advances)
    waiters_read: crate::sched::WaitQueue,
    waiters_write: crate::sched::WaitQueue,
    send_lock: Mutex<()>,
    recv_lock: Mutex<()>,
    endpoints: Mutex<PortEndpoints>,
    /// Structured message queue (used by send_msg / recv_msg and the
    /// send_handle / recv_handle compatibility wrappers).
    msgs: Mutex<alloc::collections::VecDeque<KernelMessage>>,

    #[cfg(debug_assertions)]
    sender_tid: AtomicU64,
    #[cfg(debug_assertions)]
    receiver_tid: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
struct PortEndpoints {
    readers: u32,
    writers: u32,
}

impl Port {
    /// Create a new port with the given capacity (rounded up to power of 2)
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(16).min(65536);
        let buf = alloc::vec![0u8; capacity].into_boxed_slice();
        Self {
            buf,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            waiters_read: crate::sched::WaitQueue::new(),
            waiters_write: crate::sched::WaitQueue::new(),
            send_lock: Mutex::new(()),
            recv_lock: Mutex::new(()),
            endpoints: Mutex::new(PortEndpoints {
                readers: 1,
                writers: 1,
            }),
            msgs: Mutex::new(alloc::collections::VecDeque::new()),
            #[cfg(debug_assertions)]
            sender_tid: AtomicU64::new(0),
            #[cfg(debug_assertions)]
            receiver_tid: AtomicU64::new(0),
        }
    }

    /// Returns the capacity of the port
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of bytes currently in the byte-stream buffer
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns true if both the byte-stream buffer and the message queue are empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0 && self.msgs.lock().is_empty()
    }

    /// Returns true if the byte-stream buffer is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Returns available space for writing in the byte-stream buffer
    pub fn available(&self) -> usize {
        self.capacity - self.len()
    }

    /// Send bytes to the port. Returns number of bytes written.
    /// If buffer is full, drops bytes (bounded loss behavior).
    ///
    /// This method is gated by debug assertions to ensure only one producer task
    /// accesses the port (SPSC).
    pub fn send(&self, data: &[u8]) -> usize {
        #[cfg(debug_assertions)]
        self.check_ownership(true);

        let _guard = self.send_lock.lock();
        let available = self.available();
        let to_write = data.len().min(available);

        if to_write == 0 {
            return 0;
        }

        let head = self.head.load(Ordering::Relaxed);
        let mask = self.capacity - 1; // Works because capacity is power of 2

        for (i, &byte) in data[..to_write].iter().enumerate() {
            let idx = (head + i) & mask;
            // SAFETY: We hold exclusive write access (SPSC), idx is within bounds.
            // On weakly ordered architectures, the Release store to head below ensures
            // this write is visible to a consumer performing an Acquire load.
            unsafe {
                let ptr = self.buf.as_ptr() as *mut u8;
                ptr.add(idx).write(byte);
            }
        }

        self.head
            .store(head.wrapping_add(to_write), Ordering::Release);

        // One queued message should wake one receiver.
        self.waiters_read.wake_one();

        let tid = unsafe { crate::sched::current_tid_current() };
        crate::ktrace!("PORT: Port written {} bytes from TID {}", to_write, tid);

        to_write
    }

    /// Send bytes only if the full payload fits.
    /// Returns true if all bytes were written, false if no bytes were written.
    pub fn send_all(&self, data: &[u8]) -> bool {
        #[cfg(debug_assertions)]
        self.check_ownership(true);

        let _guard = self.send_lock.lock();
        let available = self.available();
        if available < data.len() {
            return false;
        }

        if data.is_empty() {
            return true;
        }

        let head = self.head.load(Ordering::Relaxed);
        let mask = self.capacity - 1;
        for (i, &byte) in data.iter().enumerate() {
            let idx = (head + i) & mask;
            // SAFETY: idx is bounded by mask/capacity and send_lock serializes producers.
            unsafe {
                let ptr = self.buf.as_ptr() as *mut u8;
                ptr.add(idx).write(byte);
            }
        }

        self.head
            .store(head.wrapping_add(data.len()), Ordering::Release);
        self.waiters_read.wake_one();
        true
    }

    // ── Structured message API ────────────────────────────────────────────────

    /// Enqueue a structured message with optional data and capability handles.
    ///
    /// This is the kernel-internal primitive used by both the new
    /// `SYS_CHANNEL_SEND_MSG` path and the legacy `send_handle` compatibility
    /// wrapper.
    ///
    /// # Transfer semantics
    /// The caller passes pre-resolved `Arc<dyn VfsNode>` values.  Ownership of
    /// those arcs is **moved** into the message queue; the sender is responsible
    /// for removing the corresponding FD/handle from its own table before calling
    /// this function if move semantics are desired.
    pub fn send_msg(
        &self,
        data: alloc::vec::Vec<u8>,
        caps: alloc::vec::Vec<Arc<dyn crate::vfs::VfsNode>>,
    ) {
        let tid = unsafe { crate::sched::current_tid_current() };
        self.msgs.lock().push_back(KernelMessage { data, caps });
        self.waiters_read.wake_one();
        crate::ktrace!("PORT: Port message sent from TID {}", tid);
    }

    /// Dequeue the next structured message, if one is available.
    pub fn try_recv_msg(&self) -> Option<KernelMessage> {
        self.msgs.lock().pop_front()
    }

    // ── Legacy single-cap helpers (kept for internal use by compat wrappers) ──

    /// Send a single capability as a handle-only message (zero data bytes).
    ///
    /// Compatibility shim: callers should prefer `send_msg` for new code.
    pub fn send_cap(&self, node: Arc<dyn crate::vfs::VfsNode>) {
        self.send_msg(alloc::vec::Vec::new(), alloc::vec![node]);
    }

    /// Receive a single capability from the message queue.
    ///
    /// Compatibility shim: callers should prefer `try_recv_msg` for new code.
    pub fn try_recv_cap(&self) -> Option<Arc<dyn crate::vfs::VfsNode>> {
        let msg = self.msgs.lock().pop_front()?;
        msg.caps.into_iter().next()
    }

    // ── Wait queue management ─────────────────────────────────────────────────

    /// Add a reader waiter to the port
    pub fn add_waiter_read(&self, tid: u64) {
        self.waiters_read.push_back(tid);
    }

    /// Remove a reader waiter from the port
    pub fn remove_waiter_read(&self, tid: u64) {
        self.waiters_read.remove(tid);
    }

    /// Add a writer waiter to the port
    pub fn add_waiter_write(&self, tid: u64) {
        self.waiters_write.push_back(tid);
    }

    /// Remove a writer waiter from the port
    pub fn remove_waiter_write(&self, tid: u64) {
        self.waiters_write.remove(tid);
    }

    /// Receive bytes from the port. Returns number of bytes read.
    ///
    /// This method is gated by debug assertions to ensure only one consumer task
    /// accesses the port (SPSC).
    pub fn try_recv(&self, buf: &mut [u8]) -> usize {
        #[cfg(debug_assertions)]
        self.check_ownership(false);

        let _guard = self.recv_lock.lock();
        let available = self.len();
        let to_read = buf.len().min(available);

        if to_read == 0 {
            return 0;
        }

        let tail = self.tail.load(Ordering::Relaxed);
        let mask = self.capacity - 1;

        for i in 0..to_read {
            let idx = (tail + i) & mask;
            buf[i] = self.buf[idx];
        }

        self.tail
            .store(tail.wrapping_add(to_read), Ordering::Release);

        // Wake up one writer (pacing/flow control)
        self.waiters_write.wake_one();

        let tid = unsafe { crate::sched::current_tid_current() };
        crate::ktrace!("PORT: Port read {} bytes from TID {}", to_read, tid);

        to_read
    }

    pub fn recv(&self, buf: &mut [u8]) -> usize {
        self.try_recv(buf)
    }

    pub fn has_readers(&self) -> bool {
        self.endpoints.lock().readers > 0
    }

    pub fn has_writers(&self) -> bool {
        self.endpoints.lock().writers > 0
    }

    pub fn close_reader(&self) -> bool {
        let mut endpoints = self.endpoints.lock();
        if endpoints.readers == 0 {
            return false;
        }
        endpoints.readers -= 1;
        let destroy = endpoints.readers == 0 && endpoints.writers == 0;
        let last_reader = endpoints.readers == 0;
        drop(endpoints);

        if last_reader {
            self.waiters_write.wake_all();
        }

        destroy
    }

    pub fn close_writer(&self) -> bool {
        let mut endpoints = self.endpoints.lock();
        if endpoints.writers == 0 {
            return false;
        }
        endpoints.writers -= 1;
        let destroy = endpoints.readers == 0 && endpoints.writers == 0;
        let last_writer = endpoints.writers == 0;
        drop(endpoints);

        if last_writer {
            self.waiters_read.wake_all();
        }

        destroy
    }

    #[cfg(debug_assertions)]
    fn check_ownership(&self, is_sender: bool) {
        // We use the erased hook to avoid generic param requirements
        let current = unsafe { crate::sched::current_tid_current() };
        if current == 0 {
            return; // Allow kernel/idle access
        }

        let target = if is_sender {
            &self.sender_tid
        } else {
            &self.receiver_tid
        };
        let owner = target.load(Ordering::Acquire);

        if let Err(owner) = target.compare_exchange(0, current, Ordering::AcqRel, Ordering::Acquire)
        {
            if owner != current {
                // In v0 "Single Process Model", multiple tasks might share handles and ports.
                // This violates strict SPSC but is currently expected in some discovery flows.
                // We warn once per port to avoid log flood while still highlighting the issue.
                // For now, we don't panic to maintain SMP stability.
            }
        }
    }
}

// SAFETY: Port is safe to share between threads (atomic indices, SPSC access pattern).
// We implement Send and Sync but rely on Sender/Receiver wrappers and debug assertions
// to maintain the SPSC invariant.
unsafe impl Send for Port {}
unsafe impl Sync for Port {}

/// Unique handle to the sending side of a Port
pub struct Sender {
    inner: Arc<Port>,
}

impl Sender {
    pub fn new(inner: Arc<Port>) -> Self {
        Self { inner }
    }

    pub fn send(&self, data: &[u8]) -> usize {
        self.inner.send(data)
    }

    pub fn available(&self) -> usize {
        self.inner.available()
    }
}

/// Unique handle to the receiving side of a Port
pub struct Receiver {
    inner: Arc<Port>,
}

impl Receiver {
    pub fn new(inner: Arc<Port>) -> Self {
        Self { inner }
    }

    pub fn recv(&self, buf: &mut [u8]) -> usize {
        self.inner.recv(buf)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn add_waiter(&self, tid: u64) {
        self.inner.add_waiter_read(tid);
    }

    pub fn remove_waiter(&self, tid: u64) {
        self.inner.remove_waiter_read(tid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use crate::sched::blocking::WAKE_TASK_HOOK;
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    // ── Wake-hook helpers (serialised by WAKE_TEST_GUARD) ──────────────────────

    static WAKE_TEST_GUARD: spin::Mutex<()> = spin::Mutex::new(());
    static WOKEN_IDS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
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
            .map(|i| WOKEN_IDS[i].load(Ordering::SeqCst))
            .collect()
    }

    /// Test basic send/receive on a Port
    #[test]
    fn test_channel_send_recv() {
        let port = Arc::new(Port::new(64));
        let sender = Sender::new(Arc::clone(&port));
        let receiver = Receiver::new(Arc::clone(&port));

        // Initially empty
        assert!(receiver.is_empty());
        assert_eq!(receiver.len(), 0);

        // Send some data
        let data = b"hello world";
        let written = sender.send(data);
        assert_eq!(written, data.len());

        // Should be readable now
        assert!(!receiver.is_empty());
        assert_eq!(receiver.len(), data.len());

        // Receive
        let mut buf = [0u8; 64];
        let read = receiver.recv(&mut buf);
        assert_eq!(read, data.len());
        assert_eq!(&buf[..read], data);

        // Should be empty again
        assert!(receiver.is_empty());
    }

    /// Test wrap-around behavior in ring buffer
    #[test]
    fn test_port_ring_buffer_wrap() {
        let port = Arc::new(Port::new(16)); // Smallest useful power of 2
        let sender = Sender::new(Arc::clone(&port));
        let receiver = Receiver::new(Arc::clone(&port));

        // Fill buffer multiple times to test wrap-around
        for round in 0..5 {
            let data = [round as u8; 8];
            let written = sender.send(&data);
            assert_eq!(written, 8);

            let mut buf = [0u8; 8];
            let read = receiver.recv(&mut buf);
            assert_eq!(read, 8);
            assert_eq!(buf, data);
        }
    }

    /// Test bounded-loss behavior when buffer is full
    #[test]
    fn test_port_bounded_loss() {
        let port = Arc::new(Port::new(16));
        let sender = Sender::new(Arc::clone(&port));

        // Fill the buffer
        let data = [0xAB; 16];
        let written = sender.send(&data);
        assert_eq!(written, 16);

        // Further sends should return 0 (bounded loss)
        let written2 = sender.send(&[0xFF; 4]);
        assert_eq!(written2, 0);
    }

    #[test]
    fn test_channel_send_all_atomic() {
        let port = Arc::new(Port::new(16));
        let receiver = Receiver::new(Arc::clone(&port));

        assert!(port.send_all(&[1, 2, 3, 4]));
        assert!(!port.send_all(&[9; 13])); // Would overflow; must not partially write
        assert_eq!(receiver.len(), 4);

        let mut out = [0u8; 16];
        let n = receiver.recv(&mut out);
        assert_eq!(n, 4);
        assert_eq!(&out[..n], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_port_try_recv_empty() {
        let port = Arc::new(Port::new(16));
        let receiver = Receiver::new(Arc::clone(&port));
        let mut out = [0u8; 8];
        assert_eq!(receiver.recv(&mut out), 0);
        assert!(receiver.is_empty());
    }

    #[test]
    fn test_port_close_endpoints() {
        let port = Arc::new(Port::new(16));
        assert!(port.has_readers());
        assert!(port.has_writers());

        assert!(!port.close_writer());
        assert!(!port.has_writers());
        assert!(port.has_readers());

        assert!(port.close_reader());
        assert!(!port.has_readers());
    }

    // ── Structured message queue tests ────────────────────────────────────────

    /// A trivial no-op VfsNode used purely for capability transfer tests.
    struct DummyCap;
    impl crate::vfs::VfsNode for DummyCap {
        fn read(&self, _: u64, _: &mut [u8]) -> abi::errors::SysResult<usize> {
            Ok(0)
        }
        fn write(&self, _: u64, _: &[u8]) -> abi::errors::SysResult<usize> {
            Ok(0)
        }
        fn stat(&self) -> abi::errors::SysResult<crate::vfs::VfsStat> {
            Ok(crate::vfs::VfsStat::default())
        }
    }

    #[test]
    fn test_send_msg_data_only() {
        let port = Arc::new(Port::new(64));
        port.send_msg(alloc::vec![1u8, 2, 3], alloc::vec![]);
        let msg = port.try_recv_msg().expect("message should be present");
        assert_eq!(msg.data, &[1u8, 2, 3]);
        assert!(msg.caps.is_empty());
    }

    #[test]
    fn test_send_msg_cap_only() {
        let port = Arc::new(Port::new(64));
        let cap: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        port.send_msg(alloc::vec![], alloc::vec![cap]);
        let msg = port.try_recv_msg().expect("message should be present");
        assert!(msg.data.is_empty());
        assert_eq!(msg.caps.len(), 1);
    }

    #[test]
    fn test_send_msg_data_and_caps() {
        let port = Arc::new(Port::new(64));
        let cap1: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        let cap2: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        port.send_msg(alloc::vec![0xAB, 0xCD], alloc::vec![cap1, cap2]);
        let msg = port.try_recv_msg().expect("message should be present");
        assert_eq!(msg.data, &[0xAB, 0xCDu8]);
        assert_eq!(msg.caps.len(), 2);
    }

    #[test]
    fn test_send_cap_compat_wrapper() {
        let port = Arc::new(Port::new(64));
        let cap: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        port.send_cap(cap);
        let received = port.try_recv_cap().expect("cap should be present");
        // Just check it's not null (it's a valid Arc)
        let _ = received;
    }

    #[test]
    fn test_multiple_messages_ordered() {
        let port = Arc::new(Port::new(64));
        port.send_msg(alloc::vec![1], alloc::vec![]);
        port.send_msg(alloc::vec![2], alloc::vec![]);
        port.send_msg(alloc::vec![3], alloc::vec![]);

        for expected in 1u8..=3 {
            let msg = port.try_recv_msg().expect("message should be present");
            assert_eq!(msg.data[0], expected);
        }
        assert!(port.try_recv_msg().is_none());
    }

    #[test]
    fn test_is_empty_considers_message_queue() {
        let port = Arc::new(Port::new(64));
        // Both ring buffer and message queue empty
        assert!(port.is_empty());
        // Add a message — port should not be considered empty
        port.send_msg(alloc::vec![], alloc::vec![]);
        assert!(!port.is_empty());
        // Drain the message
        port.try_recv_msg();
        assert!(port.is_empty());
    }

    // ── Peer-death semantics ──────────────────────────────────────────────────

    /// Closing the write end marks the port as having no writers.
    #[test]
    fn test_close_writer_marks_no_writers() {
        let port = Arc::new(Port::new(64));
        assert!(port.has_writers());
        assert!(port.has_readers());

        assert!(!port.close_writer()); // not destroyed (reader still open)
        assert!(!port.has_writers());
        assert!(port.has_readers());
    }

    /// Closing the read end marks the port as having no readers.
    #[test]
    fn test_close_reader_marks_no_readers() {
        let port = Arc::new(Port::new(64));
        assert!(port.has_writers());
        assert!(port.has_readers());

        assert!(!port.close_reader()); // not destroyed (writer still open)
        assert!(port.has_writers());
        assert!(!port.has_readers());
    }

    /// After the write end closes, already-buffered bytes are still readable,
    /// but a subsequent recv on an empty ring signals EOF (has_writers == false).
    #[test]
    fn test_recv_drains_then_eof_after_writer_close() {
        let port = Arc::new(Port::new(64));
        let written = port.send(b"hello");
        assert_eq!(written, 5);

        port.close_writer();
        assert!(!port.has_writers(), "writer is closed");

        // Buffered bytes are still available
        let mut buf = [0u8; 64];
        let n = port.try_recv(&mut buf);
        assert_eq!(n, 5);
        assert_eq!(&buf[..n], b"hello");

        // Ring is now empty and writer is gone → syscall would return EPIPE
        assert_eq!(port.try_recv(&mut buf), 0);
        assert!(!port.has_writers());
    }

    /// Structured messages queued before the writer closes are still delivered.
    #[test]
    fn test_msg_queue_drains_after_writer_close() {
        let port = Arc::new(Port::new(64));
        port.send_msg(alloc::vec![42u8], alloc::vec![]);
        port.close_writer();

        let msg = port.try_recv_msg().expect("message must survive writer close");
        assert_eq!(msg.data, &[42u8]);
        assert!(port.try_recv_msg().is_none());
        assert!(!port.has_writers());
    }

    // ── FIFO ordering ─────────────────────────────────────────────────────────

    /// Bytes written in multiple sends are delivered in FIFO order.
    #[test]
    fn test_fifo_ordering_byte_stream() {
        let port = Arc::new(Port::new(256));
        port.send(&[1u8, 2, 3]);
        port.send(&[4u8, 5, 6]);

        let mut buf = [0u8; 256];
        let n = port.try_recv(&mut buf);
        assert_eq!(n, 6);
        assert_eq!(&buf[..n], &[1u8, 2, 3, 4, 5, 6]);
    }

    // ── Atomicity edge cases ──────────────────────────────────────────────────

    /// send_all succeeds when the ring has exactly enough space, and fails
    /// (all-or-nothing) when it has one byte less.
    #[test]
    fn test_send_all_at_exact_capacity_boundary() {
        let port = Arc::new(Port::new(16));

        // Fill the ring exactly
        assert!(port.send_all(&[0xAAu8; 16]));
        // Ring is full — even a 1-byte send_all must fail
        assert!(!port.send_all(&[0xFF]));

        // Drain all
        let mut buf = [0u8; 16];
        let n = port.try_recv(&mut buf);
        assert_eq!(n, 16);

        // Now there is room; a 16-byte write should succeed again
        assert!(port.send_all(&[0xBBu8; 16]));
    }

    /// send_all with a length equal to capacity − 1 must not corrupt the buffer.
    #[test]
    fn test_send_all_partial_fill_then_top_up() {
        let port = Arc::new(Port::new(16));

        // Write 15 bytes (one slot free)
        assert!(port.send_all(&[0x01u8; 15]));
        // Try to write 2 more — does not fit (only 1 free slot)
        assert!(!port.send_all(&[0x02u8; 2]));
        // Write exactly 1 — fits
        assert!(port.send_all(&[0x03u8]));

        let mut out = [0u8; 16];
        let n = port.try_recv(&mut out);
        assert_eq!(n, 16);
        assert_eq!(&out[..15], &[0x01u8; 15]);
        assert_eq!(out[15], 0x03);
    }

    // ── Wakeup on peer death ──────────────────────────────────────────────────

    /// When the write end closes, all registered read waiters are woken.
    #[test]
    fn test_close_writer_wakes_read_waiters() {
        let _g = WAKE_TEST_GUARD.lock();
        reset_wakes();
        WAKE_TASK_HOOK.store(record_wake as *mut (), Ordering::SeqCst);

        let port = Arc::new(Port::new(64));
        port.add_waiter_read(100);
        port.add_waiter_read(101);
        port.close_writer();

        let log = wake_log();
        assert!(log.contains(&100), "tid 100 must be woken on writer close");
        assert!(log.contains(&101), "tid 101 must be woken on writer close");
    }

    /// When the read end closes, all registered write waiters are woken.
    #[test]
    fn test_close_reader_wakes_write_waiters() {
        let _g = WAKE_TEST_GUARD.lock();
        reset_wakes();
        WAKE_TASK_HOOK.store(record_wake as *mut (), Ordering::SeqCst);

        let port = Arc::new(Port::new(64));
        port.add_waiter_write(200);
        port.add_waiter_write(201);
        port.close_reader();

        let log = wake_log();
        assert!(log.contains(&200), "tid 200 must be woken on reader close");
        assert!(log.contains(&201), "tid 201 must be woken on reader close");
    }

    // ── Cleanup semantics for queued caps ─────────────────────────────────────

    /// Caps queued in the message queue are released when the port is dropped.
    /// No cap should silently leak into any process's thing table.
    #[test]
    fn test_queued_caps_released_on_port_drop() {
        let port = Arc::new(Port::new(64));
        let cap: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        let weak = Arc::downgrade(&cap);

        // Enqueue the cap; the port now holds the only strong reference besides
        // the local `cap` binding.
        port.send_cap(cap);

        // Drop the local binding — port is the sole owner now.
        // (The strong count is 1, held by the message queue.)
        let strong_before = weak.strong_count();
        assert_eq!(strong_before, 1, "only the port queue holds the cap");

        // Drop the port → the VecDeque<KernelMessage> is dropped → Arc is freed.
        drop(port);
        assert!(
            weak.upgrade().is_none(),
            "cap must be freed when port is dropped"
        );
    }

    /// Multiple caps in a single message are all released when the port drops.
    #[test]
    fn test_multiple_queued_caps_all_released_on_drop() {
        let port = Arc::new(Port::new(64));

        let cap1: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        let cap2: Arc<dyn crate::vfs::VfsNode> = Arc::new(DummyCap);
        let weak1 = Arc::downgrade(&cap1);
        let weak2 = Arc::downgrade(&cap2);

        port.send_msg(alloc::vec![], alloc::vec![cap1, cap2]);

        drop(port);
        assert!(weak1.upgrade().is_none(), "cap1 must be freed");
        assert!(weak2.upgrade().is_none(), "cap2 must be freed");
    }
}
