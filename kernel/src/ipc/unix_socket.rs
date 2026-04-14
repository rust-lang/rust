//! Unix domain stream sockets — first-class VFS-backed IPC.
//!
//! # Design
//!
//! A Unix domain socket is a VFS file descriptor that participates in the
//! standard `poll()` readiness infrastructure.  Sockets are bound to
//! filesystem paths under `/run` (or elsewhere) via `bind()`, which:
//!
//!  1. Records the socket in the global [`SOCKET_REGISTRY`] keyed by path.
//!  2. Inserts a lightweight marker node into the VFS at that path.
//!
//! A client calls `connect(fd, "/run/foo.sock")` to create a peer connection.
//! A server calls `listen()` then `accept()` to hand out new fds for each
//! incoming connection.
//!
//! Connected socket pairs share a [`SocketPeer`]: two ring buffers
//! (one per direction) plus four wait queues.  A `UnixSocketNode` holds an
//! `Arc<Mutex<SocketPeer>>` along with a side tag (`SideA` / `SideB`).
//!
//! # Supported socket types
//! Currently only `AF_UNIX + SOCK_STREAM` is implemented.  `SOCK_DGRAM` will
//! be added later and can share this module.
//!
//! # State machine (per socket fd)
//! ```text
//!  Unbound ──bind──► Bound ──listen──► Listening
//!                                          │
//!                                      accept()
//!                                          │
//!                                      Connected ◄──connect()
//! ```

#![allow(clippy::mutex_atomic)]

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use spin::Mutex;

use crate::sched::wait_queue::WaitQueue;
use crate::vfs::{VfsNode, VfsStat};

// ---------------------------------------------------------------------------
// Ring buffer (reused from pipe module pattern)
// ---------------------------------------------------------------------------

struct RingBuf {
    data: Vec<u8>,
    head: usize,
    tail: usize,
    len: usize,
    cap: usize,
}

impl RingBuf {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let mut data = Vec::with_capacity(cap);
        data.resize(cap, 0u8);
        Self { data, head: 0, tail: 0, len: 0, cap }
    }

    fn is_empty(&self) -> bool { self.len == 0 }
    fn is_full(&self)  -> bool { self.len == self.cap }
    fn available(&self) -> usize { self.len }
    fn free_space(&self) -> usize { self.cap - self.len }

    fn dequeue(&mut self, dst: &mut [u8]) -> usize {
        let n = dst.len().min(self.len);
        for b in dst[..n].iter_mut() {
            *b = self.data[self.head];
            self.head = (self.head + 1) % self.cap;
        }
        self.len -= n;
        n
    }

    fn enqueue(&mut self, src: &[u8]) -> usize {
        let n = src.len().min(self.free_space());
        for &b in &src[..n] {
            self.data[self.tail] = b;
            self.tail = (self.tail + 1) % self.cap;
        }
        self.len += n;
        n
    }
}

// ---------------------------------------------------------------------------
// Shared peer state for connected socket pair
// ---------------------------------------------------------------------------

/// Default socket buffer capacity in bytes.
const DEFAULT_SOCK_CAPACITY: usize = 4096;

/// State shared between the two ends of a connected socket pair.
///
/// Side A (the connecting/client side) writes to `a_to_b` and reads from
/// `b_to_a`.  Side B (the accepted/server side) is the mirror.
pub struct SocketPeer {
    /// Data written by A, readable by B.
    a_to_b: RingBuf,
    /// Data written by B, readable by A.
    b_to_a: RingBuf,
    /// Is A still alive (not shut down for writing)?
    a_alive: bool,
    /// Is B still alive (not shut down for writing)?
    b_alive: bool,
    /// A waiting to read (from b_to_a).
    a_read_waitq: WaitQueue,
    /// B waiting to read (from a_to_b).
    b_read_waitq: WaitQueue,
    /// A waiting to write (to a_to_b, which may be full).
    a_write_waitq: WaitQueue,
    /// B waiting to write (to b_to_a, which may be full).
    b_write_waitq: WaitQueue,
}

impl SocketPeer {
    fn new() -> Self {
        Self {
            a_to_b:        RingBuf::new(DEFAULT_SOCK_CAPACITY),
            b_to_a:        RingBuf::new(DEFAULT_SOCK_CAPACITY),
            a_alive:       true,
            b_alive:       true,
            a_read_waitq:  WaitQueue::new(),
            b_read_waitq:  WaitQueue::new(),
            a_write_waitq: WaitQueue::new(),
            b_write_waitq: WaitQueue::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Listening socket entry (server side)
// ---------------------------------------------------------------------------

/// Server-side listening state kept in the socket registry.
pub struct ListeningSocket {
    backlog: usize,
    /// Accepted-but-not-yet-accept()ed connections: each is an
    /// `Arc<UnixSocketNode>` in the Connected/SideB state.
    accept_queue: Mutex<VecDeque<Arc<dyn VfsNode>>>,
    /// Woken when a new connection is enqueued.
    accept_waitq: WaitQueue,
}

impl ListeningSocket {
    fn new(backlog: usize) -> Self {
        Self {
            backlog,
            accept_queue: Mutex::new(VecDeque::new()),
            accept_waitq: WaitQueue::new(),
        }
    }

    pub fn queue_len(&self) -> usize {
        self.accept_queue.lock().len()
    }
}

// ---------------------------------------------------------------------------
// Global socket registry — path → ListeningSocket
// ---------------------------------------------------------------------------

static SOCKET_REGISTRY: Mutex<BTreeMap<String, Arc<ListeningSocket>>> =
    Mutex::new(BTreeMap::new());

/// Register a path in the global socket registry, overwriting any stale entry.
fn registry_insert(path: &str, listener: Arc<ListeningSocket>) {
    SOCKET_REGISTRY.lock().insert(path.to_string(), listener);
}

/// Remove a path from the global socket registry.
fn registry_remove(path: &str) {
    SOCKET_REGISTRY.lock().remove(path);
}

/// Look up a listener by path; returns `None` if no such listener exists.
fn registry_get(path: &str) -> Option<Arc<ListeningSocket>> {
    SOCKET_REGISTRY.lock().get(path).cloned()
}

// ---------------------------------------------------------------------------
// UnixSocketNode — the VfsNode implementation
// ---------------------------------------------------------------------------

/// Which side of a connected [`SocketPeer`] this node represents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side { A, B }

/// Internal state of a [`UnixSocketNode`].
enum SocketState {
    /// Freshly created via `socket()`.
    Unbound,
    /// `bind()` called; socket is waiting for `listen()`.
    Bound { path: String },
    /// `listen()` called; server is ready to accept.
    Listening { path: String, listener: Arc<ListeningSocket> },
    /// `connect()` or `accept()` succeeded.
    Connected { side: Side, peer: Arc<Mutex<SocketPeer>>, shutdown_rd: bool, shutdown_wr: bool },
    /// Socket has been fully closed.
    Closed,
}

static NEXT_SOCKET_INO: AtomicU64 = AtomicU64::new(1);

/// A Unix domain stream socket exposed as a [`VfsNode`].
pub struct UnixSocketNode {
    ino:   u64,
    state: Mutex<SocketState>,
}

impl UnixSocketNode {
    /// Create a new unbound socket.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            ino:   NEXT_SOCKET_INO.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(SocketState::Unbound),
        })
    }

    /// Create a pre-connected pair for `socketpair()`.
    ///
    /// Returns `(side_a, side_b)` — both are in the Connected state and can
    /// be inserted directly into the calling process's fd table.
    pub fn new_pair() -> (Arc<Self>, Arc<Self>) {
        let peer = Arc::new(Mutex::new(SocketPeer::new()));
        let a = Arc::new(Self {
            ino: NEXT_SOCKET_INO.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(SocketState::Connected {
                side: Side::A,
                peer: peer.clone(),
                shutdown_rd: false,
                shutdown_wr: false,
            }),
        });
        let b = Arc::new(Self {
            ino: NEXT_SOCKET_INO.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(SocketState::Connected {
                side: Side::B,
                peer,
                shutdown_rd: false,
                shutdown_wr: false,
            }),
        });
        (a, b)
    }

    /// Bind this socket to `path`.  Returns `EADDRINUSE` if already bound or
    /// if the path is already registered.
    pub fn bind(&self, path: &str) -> abi::errors::SysResult<()> {
        let mut state = self.state.lock();
        match &*state {
            SocketState::Unbound => {}
            _ => return Err(abi::errors::Errno::EINVAL),
        }
        if SOCKET_REGISTRY.lock().contains_key(path) {
            return Err(abi::errors::Errno::EADDRINUSE);
        }
        *state = SocketState::Bound { path: path.to_string() };
        Ok(())
    }

    /// Mark the socket as listening.  Must be called after `bind()`.
    pub fn listen(&self, backlog: usize) -> abi::errors::SysResult<()> {
        let mut state = self.state.lock();
        let path = match &*state {
            SocketState::Bound { path } => path.clone(),
            _ => return Err(abi::errors::Errno::EINVAL),
        };
        let backlog = backlog.max(1).min(128);
        let listener = Arc::new(ListeningSocket::new(backlog));
        registry_insert(&path, listener.clone());
        *state = SocketState::Listening { path, listener };
        Ok(())
    }

    /// Accept one incoming connection.  Blocks until a connection arrives.
    ///
    /// Returns a new `Arc<dyn VfsNode>` for the server side of the connection.
    pub fn accept(&self) -> abi::errors::SysResult<Arc<dyn VfsNode>> {
        loop {
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }
            let tid = unsafe { crate::sched::current_tid_current() };
            {
                let state = self.state.lock();
                let listener = match &*state {
                    SocketState::Listening { listener, .. } => listener.clone(),
                    _ => return Err(abi::errors::Errno::EINVAL),
                };
                drop(state);

                let mut queue = listener.accept_queue.lock();
                if let Some(server_node) = queue.pop_front() {
                    return Ok(server_node);
                }
                // Register as waiter before dropping queue lock.
                listener.accept_waitq.push_back(tid as u64);
            }
            unsafe { crate::task::block_current_erased() };
            // After wakeup, loop and re-check.
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }
        }
    }

    /// Connect to a listening socket at `path`.
    pub fn connect(&self, path: &str) -> abi::errors::SysResult<()> {
        let listener = registry_get(path).ok_or(abi::errors::Errno::ECONNREFUSED)?;

        let peer = Arc::new(Mutex::new(SocketPeer::new()));

        // Server-side socket (SideB): pre-connected, placed into accept queue.
        let server_node = Arc::new(UnixSocketNode {
            ino: NEXT_SOCKET_INO.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(SocketState::Connected {
                side: Side::B,
                peer: peer.clone(),
                shutdown_rd: false,
                shutdown_wr: false,
            }),
        });

        {
            let mut queue = listener.accept_queue.lock();
            if queue.len() >= listener.backlog {
                return Err(abi::errors::Errno::ECONNREFUSED);
            }
            queue.push_back(server_node as Arc<dyn VfsNode>);
        }
        // Wake the accept() side.
        listener.accept_waitq.wake_one();

        // Transition self to Connected (SideA).
        let mut state = self.state.lock();
        match &*state {
            SocketState::Unbound => {}
            _ => return Err(abi::errors::Errno::EINVAL),
        }
        *state = SocketState::Connected {
            side: Side::A,
            peer,
            shutdown_rd: false,
            shutdown_wr: false,
        };
        Ok(())
    }

    /// Perform a shutdown in direction `how` (0=RD, 1=WR, 2=RDWR).
    pub fn shutdown(&self, how: u32) -> abi::errors::SysResult<()> {
        let mut state = self.state.lock();
        match &mut *state {
            SocketState::Connected { side, peer, shutdown_rd, shutdown_wr } => {
                let shut_rd = how == 0 || how == 2;
                let shut_wr = how == 1 || how == 2;
                if shut_rd { *shutdown_rd = true; }
                if shut_wr {
                    *shutdown_wr = true;
                    // Wake the peer's readers so they see EOF.
                    let p = peer.lock();
                    match side {
                        Side::A => p.b_read_waitq.wake_all(),
                        Side::B => p.a_read_waitq.wake_all(),
                    }
                }
                if shut_rd {
                    // Wake the peer's writers so they get EPIPE/ECONNRESET.
                    let p = peer.lock();
                    match side {
                        Side::A => p.b_write_waitq.wake_all(),
                        Side::B => p.a_write_waitq.wake_all(),
                    }
                }
                Ok(())
            }
            SocketState::Closed => Err(abi::errors::Errno::EBADF),
            _ => Err(abi::errors::Errno::ENOTCONN),
        }
    }
}

// ---------------------------------------------------------------------------
// VfsNode implementation
// ---------------------------------------------------------------------------

impl VfsNode for UnixSocketNode {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> abi::errors::SysResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }
            let tid = unsafe { crate::sched::current_tid_current() };
            {
                let state = self.state.lock();
                match &*state {
                    SocketState::Connected { side, peer, shutdown_rd, .. } => {
                        if *shutdown_rd {
                            return Ok(0);
                        }
                        let mut p = peer.lock();
                        // Extract bool copies before taking any mutable borrow.
                        let other_alive = match side { Side::A => p.b_alive, Side::B => p.a_alive };
                        let has_data = match side { Side::A => !p.b_to_a.is_empty(), Side::B => !p.a_to_b.is_empty() };

                        if has_data {
                            let n = match side {
                                Side::A => p.b_to_a.dequeue(buf),
                                Side::B => p.a_to_b.dequeue(buf),
                            };
                            match side {
                                Side::A => p.b_write_waitq.wake_one(),
                                Side::B => p.a_write_waitq.wake_one(),
                            }
                            return Ok(n);
                        }
                        if !other_alive {
                            return Ok(0); // EOF
                        }
                        // Register in read wait queue before blocking.
                        match side {
                            Side::A => p.a_read_waitq.push_back(tid as u64),
                            Side::B => p.b_read_waitq.push_back(tid as u64),
                        }
                    }
                    SocketState::Closed => return Err(abi::errors::Errno::EBADF),
                    _ => return Err(abi::errors::Errno::ENOTCONN),
                }
            }
            unsafe { crate::task::block_current_erased() };
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }
        }
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> abi::errors::SysResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }
            let tid = unsafe { crate::sched::current_tid_current() };
            {
                let state = self.state.lock();
                match &*state {
                    SocketState::Connected { side, peer, shutdown_wr, .. } => {
                        if *shutdown_wr {
                            return Err(abi::errors::Errno::EPIPE);
                        }
                        let mut p = peer.lock();
                        // Extract bool copies before taking any mutable borrow.
                        let other_alive = match side { Side::A => p.b_alive, Side::B => p.a_alive };
                        let tx_full = match side { Side::A => p.a_to_b.is_full(), Side::B => p.b_to_a.is_full() };

                        if !other_alive {
                            return Err(abi::errors::Errno::EPIPE);
                        }
                        if !tx_full {
                            let n = match side {
                                Side::A => p.a_to_b.enqueue(buf),
                                Side::B => p.b_to_a.enqueue(buf),
                            };
                            match side {
                                Side::A => p.b_read_waitq.wake_one(),
                                Side::B => p.a_read_waitq.wake_one(),
                            }
                            return Ok(n);
                        }
                        // Buffer is full — register and block.
                        match side {
                            Side::A => p.a_write_waitq.push_back(tid as u64),
                            Side::B => p.b_write_waitq.push_back(tid as u64),
                        }
                    }
                    SocketState::Closed => return Err(abi::errors::Errno::EBADF),
                    _ => return Err(abi::errors::Errno::ENOTCONN),
                }
            }
            unsafe { crate::task::block_current_erased() };
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }
        }
    }

    fn stat(&self) -> abi::errors::SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFSOCK | 0o600,
            ino: self.ino,
            nlink: 1,
            ..Default::default()
        })
    }

    fn close(&self) {
        let mut state = self.state.lock();
        match &*state {
            SocketState::Listening { path, .. } => {
                registry_remove(path);
            }
            SocketState::Connected { side, peer, .. } => {
                let mut p = peer.lock();
                match side {
                    Side::A => {
                        p.a_alive = false;
                        p.b_read_waitq.wake_all();
                        p.b_write_waitq.wake_all();
                    }
                    Side::B => {
                        p.b_alive = false;
                        p.a_read_waitq.wake_all();
                        p.a_write_waitq.wake_all();
                    }
                }
            }
            _ => {}
        }
        *state = SocketState::Closed;
    }

    fn poll(&self) -> u16 {
        use abi::syscall::poll_flags::{POLLERR, POLLHUP, POLLIN, POLLOUT};
        let state = self.state.lock();
        match &*state {
            SocketState::Connected { side, peer, shutdown_rd, shutdown_wr } => {
                let p = peer.lock();
                let mut events = 0u16;
                let (rx_buf, tx_buf, other_alive) = match side {
                    Side::A => (&p.b_to_a, &p.a_to_b, p.b_alive),
                    Side::B => (&p.a_to_b, &p.b_to_a, p.a_alive),
                };
                if !shutdown_rd && (!rx_buf.is_empty() || !other_alive) {
                    events |= POLLIN;
                }
                if !other_alive {
                    events |= POLLHUP;
                }
                if *shutdown_wr {
                    events |= POLLERR;
                } else if !tx_buf.is_full() && other_alive {
                    events |= POLLOUT;
                }
                events
            }
            SocketState::Listening { listener, .. } => {
                if listener.queue_len() > 0 {
                    POLLIN
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    fn add_waiter(&self, tid: u64) {
        let state = self.state.lock();
        match &*state {
            SocketState::Connected { side, peer, .. } => {
                let p = peer.lock();
                match side {
                    Side::A => p.a_read_waitq.push_back(tid),
                    Side::B => p.b_read_waitq.push_back(tid),
                }
            }
            SocketState::Listening { listener, .. } => {
                listener.accept_waitq.push_back(tid);
            }
            _ => {}
        }
    }

    fn remove_waiter(&self, tid: u64) {
        let state = self.state.lock();
        match &*state {
            SocketState::Connected { side, peer, .. } => {
                let p = peer.lock();
                match side {
                    Side::A => p.a_read_waitq.remove(tid),
                    Side::B => p.b_read_waitq.remove(tid),
                }
            }
            SocketState::Listening { listener, .. } => {
                listener.accept_waitq.remove(tid);
            }
            _ => {}
        }
    }

    // ── Socket-specific VfsNode methods ─────────────────────────────────────

    fn sock_bind(&self, path: &str) -> abi::errors::SysResult<()> {
        self.bind(path)
    }

    fn sock_listen(&self, backlog: usize) -> abi::errors::SysResult<()> {
        self.listen(backlog)
    }

    fn sock_accept(&self) -> abi::errors::SysResult<alloc::sync::Arc<dyn VfsNode>> {
        self.accept()
    }

    fn sock_connect(&self, path: &str) -> abi::errors::SysResult<()> {
        self.connect(path)
    }

    fn sock_shutdown(&self, how: u32) -> abi::errors::SysResult<()> {
        self.shutdown(how)
    }
}

// ---------------------------------------------------------------------------
// VFS marker node — placed at the bound path so stat()/ls work correctly
// ---------------------------------------------------------------------------

/// A lightweight VFS node that appears at the socket's bind path.
///
/// This node is purely a filesystem marker (S_IFSOCK).  Actual connect/accept
/// traffic is routed via the [`SOCKET_REGISTRY`], not through this node.
pub struct SocketFileMarker {
    ino: u64,
}

impl SocketFileMarker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            ino: NEXT_SOCKET_INO.fetch_add(1, Ordering::Relaxed),
        })
    }
}

impl VfsNode for SocketFileMarker {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> abi::errors::SysResult<usize> {
        Err(abi::errors::Errno::EOPNOTSUPP)
    }

    fn write(&self, _offset: u64, _buf: &[u8]) -> abi::errors::SysResult<usize> {
        Err(abi::errors::Errno::EOPNOTSUPP)
    }

    fn stat(&self) -> abi::errors::SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFSOCK | 0o600,
            ino: self.ino,
            nlink: 1,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use abi::syscall::poll_flags;

    // Wire up no-op scheduling hooks for tests.
    fn setup_test_hooks() {
        fn tid() -> u64 { 1 }
        fn no_interrupt() -> bool { false }
        unsafe {
            crate::sched::hooks::CURRENT_TID_HOOK = Some(tid);
            crate::sched::hooks::TAKE_PENDING_INTERRUPT_HOOK = Some(no_interrupt);
        }
    }

    // ── socketpair ──────────────────────────────────────────────────────────

    #[test]
    fn socketpair_write_a_read_b() {
        setup_test_hooks();
        let (a, b) = UnixSocketNode::new_pair();
        a.write(0, b"hello").expect("write");
        let mut buf = [0u8; 8];
        let n = b.read(0, &mut buf).expect("read");
        assert_eq!(&buf[..n], b"hello");
    }

    #[test]
    fn socketpair_write_b_read_a() {
        setup_test_hooks();
        let (a, b) = UnixSocketNode::new_pair();
        b.write(0, b"world").expect("write");
        let mut buf = [0u8; 8];
        let n = a.read(0, &mut buf).expect("read");
        assert_eq!(&buf[..n], b"world");
    }

    #[test]
    fn socketpair_close_a_gives_eof_on_b() {
        setup_test_hooks();
        let (a, b) = UnixSocketNode::new_pair();
        a.close();
        let mut buf = [0u8; 8];
        let n = b.read(0, &mut buf).expect("read after peer close");
        assert_eq!(n, 0, "expected EOF");
    }

    #[test]
    fn socketpair_poll_write_then_read() {
        setup_test_hooks();
        let (a, b) = UnixSocketNode::new_pair();

        // Both writable, neither readable initially.
        assert_ne!(a.poll() & poll_flags::POLLOUT, 0, "A writable");
        assert_eq!(a.poll() & poll_flags::POLLIN,  0, "A not readable");

        a.write(0, b"x").expect("write");

        assert_ne!(b.poll() & poll_flags::POLLIN, 0, "B readable after A writes");
    }

    #[test]
    fn socketpair_poll_hangup_after_close() {
        setup_test_hooks();
        let (a, b) = UnixSocketNode::new_pair();
        a.close();
        let flags = b.poll();
        assert_ne!(flags & poll_flags::POLLHUP, 0, "POLLHUP after peer close");
    }

    // ── bind / connect / accept  ────────────────────────────────────────────

    #[test]
    fn bind_and_connect_exchange_data() {
        setup_test_hooks();

        let server = UnixSocketNode::new();
        server.bind("/run/test_sock_1").expect("bind");
        server.listen(4).expect("listen");

        // Client connects.
        let client = UnixSocketNode::new();
        client.connect("/run/test_sock_1").expect("connect");

        // Server accepts.
        let accepted = server.accept().expect("accept");

        // Client → server.
        client.write(0, b"ping").expect("write");
        let mut buf = [0u8; 8];
        let n = accepted.read(0, &mut buf).expect("read");
        assert_eq!(&buf[..n], b"ping");

        // Server → client.
        accepted.write(0, b"pong").expect("write back");
        let n = client.read(0, &mut buf).expect("read back");
        assert_eq!(&buf[..n], b"pong");

        // Clean up registry entry.
        server.close();
    }

    #[test]
    fn duplicate_bind_returns_eaddrinuse() {
        setup_test_hooks();
        let s1 = UnixSocketNode::new();
        s1.bind("/run/dup_bind_test").expect("first bind");
        s1.listen(1).expect("listen");

        let s2 = UnixSocketNode::new();
        let err = s2.bind("/run/dup_bind_test").expect_err("duplicate bind");
        assert_eq!(err, abi::errors::Errno::EADDRINUSE);

        s1.close();
    }

    #[test]
    fn connect_to_nonexistent_path_returns_econnrefused() {
        setup_test_hooks();
        let client = UnixSocketNode::new();
        let err = client
            .connect("/run/no_such_socket")
            .expect_err("connect to missing");
        assert_eq!(err, abi::errors::Errno::ECONNREFUSED);
    }
}
