//! High-level `WaitSet` primitive — FD-centric readiness substrate.
//!
//! `WaitSet` lets a task block until *any* of a collection of event sources
//! becomes ready: file descriptors, ports, timers, task-exit signals, and IRQs.
//! Internally it builds a `WaitSpec` array and calls the `SYS_WAIT_MANY`
//! syscall, which parks the calling task in the kernel until at least one
//! source fires.
//!
//! This eliminates userspace polling loops.  Because all VFS-backed objects
//! (pipes, sockets, and channel ends bridged via `SYS_FS_FD_FROM_HANDLE`)
//! expose FD readiness, a single `WaitSet` can multiplex the complete set of
//! I/O the calling task cares about.
//!
//! # Example
//!
//! ```no_run
//! use stem::wait_set::{WaitSet, WaitEvent};
//! use core::time::Duration;
//!
//! let mut set = WaitSet::new();
//! let rx_port = 1u64;
//! let pipe_read_fd = 3u32;
//! let tok_rx   = set.add_port_readable(rx_port).unwrap();
//! let tok_pipe = set.add_fd_readable(pipe_read_fd).unwrap();
//!
//! for event in set.wait(Some(Duration::from_secs(5))).unwrap() {
//!     if event.token() == tok_rx && event.is_readable() {
//!         // port has data — call channel_recv
//!     } else if event.token() == tok_pipe && event.is_readable() {
//!         // pipe has data — call vfs_read
//!     }
//! }
//! ```

use abi::wait::{interest, WaitKind, WaitResult, WaitSpec, WAIT_MANY_MAX_ITEMS};
use alloc::vec::Vec;

use crate::errors::Errno;
use crate::syscall;
use crate::time::Duration;

// ─── Token ────────────────────────────────────────────────────────────────────

/// An opaque token that identifies a specific event source inside a [`WaitSet`].
///
/// Tokens are assigned at registration time and echoed back in every
/// [`WaitEvent`] so callers can dispatch by source without scanning by object
/// handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaitToken(u64);

// ─── WaitEvent ────────────────────────────────────────────────────────────────

/// A single ready event returned from [`WaitSet::wait`].
#[derive(Debug, Clone, Copy)]
pub struct WaitEvent {
    result: WaitResult,
}

impl WaitEvent {
    /// The token that was assigned when this source was added to the set.
    #[inline]
    pub fn token(&self) -> WaitToken {
        WaitToken(self.result.token)
    }

    /// Raw `WaitKind` discriminant.
    #[inline]
    pub fn kind(&self) -> u32 {
        self.result.kind
    }

    /// Raw ready-flag bitmask (see `abi::wait::ready`).
    #[inline]
    pub fn flags(&self) -> u32 {
        self.result.flags
    }

    /// The kernel object (port handle, watch id, tid, …) that became ready.
    #[inline]
    pub fn object(&self) -> u64 {
        self.result.object
    }

    /// Optional value payload:
    /// - port: bytes pending / space available
    /// - task exit: exit code
    /// - graph op: return value or errno
    #[inline]
    pub fn value(&self) -> i64 {
        self.result.value
    }

    /// Port (or watch) has data available to read.
    #[inline]
    pub fn is_readable(&self) -> bool {
        self.result.flags & abi::wait::ready::READABLE != 0
    }

    /// Port has space to write.
    #[inline]
    pub fn is_writable(&self) -> bool {
        self.result.flags & abi::wait::ready::WRITABLE != 0
    }

    /// The remote end has closed / hung up.
    #[inline]
    pub fn is_hangup(&self) -> bool {
        self.result.flags & abi::wait::ready::HANGUP != 0
    }

    /// A watched task has exited.
    #[inline]
    pub fn is_exited(&self) -> bool {
        self.result.flags & abi::wait::ready::EXITED != 0
    }

    /// The overall `wait_many` call timed out (global timeout, not a per-item
    /// timer source).
    #[inline]
    pub fn is_timeout(&self) -> bool {
        self.result.flags & abi::wait::ready::TIMEOUT != 0
    }

    /// A watch stream overflowed; some events were lost.
    #[inline]
    pub fn is_overflow(&self) -> bool {
        self.result.flags & abi::wait::ready::OVERFLOW != 0
    }

    /// An error occurred on this source.
    #[inline]
    pub fn is_error(&self) -> bool {
        self.result.flags & abi::wait::ready::ERROR != 0
    }

    /// An IRQ was delivered.
    #[inline]
    pub fn is_irq(&self) -> bool {
        self.result.flags & abi::wait::ready::IRQ != 0
    }

    /// An async graph operation completed (successfully or with an error).
    #[inline]
    pub fn is_done(&self) -> bool {
        self.result.flags & abi::wait::ready::DONE != 0
    }
}

// ─── WaitEvents ───────────────────────────────────────────────────────────────

/// Container of zero or more ready events returned from [`WaitSet::wait`].
#[derive(Debug)]
pub struct WaitEvents(Vec<WaitEvent>);

impl WaitEvents {
    /// Returns `true` when no sources fired (e.g. global timeout expired
    /// before any source became ready and no timeout-kind source was present).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of ready events.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate over the ready events.
    pub fn iter(&self) -> core::slice::Iter<'_, WaitEvent> {
        self.0.iter()
    }
}

impl IntoIterator for WaitEvents {
    type Item = WaitEvent;
    type IntoIter = alloc::vec::IntoIter<WaitEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// ─── WaitSet ──────────────────────────────────────────────────────────────────

/// Mutable set of waitable event sources.
///
/// Sources can be added and removed between calls to [`WaitSet::wait`] so
/// applications can dynamically track the exact set of things they care about.
/// Membership changes take effect on the next `wait` call.
///
/// Up to [`WAIT_MANY_MAX_ITEMS`] sources may be registered simultaneously.
/// Attempting to add more returns [`Errno::ENOSPC`].
pub struct WaitSet {
    specs: Vec<WaitSpec>,
    next_token: u64,
}

impl WaitSet {
    /// Create an empty `WaitSet`.
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
            next_token: 1,
        }
    }

    // ── internal helpers ───────────────────────────────────────────────────

    fn alloc_token(&mut self) -> WaitToken {
        let tok = WaitToken(self.next_token);
        // Skip 0 so that token 0 is never issued (easier to spot bugs where a
        // token field was left default-initialised). Wrapping is safe here
        // because push_spec() bounds the set to WAIT_MANY_MAX_ITEMS entries,
        // so a single WaitSet can hold at most 32 live tokens at any point in
        // time; token reuse after u64 wrap-around is therefore not a concern
        // in practice.
        self.next_token = self.next_token.wrapping_add(1).max(1);
        tok
    }

    fn push_spec(&mut self, kind: WaitKind, flags: u32, object: u64) -> Result<WaitToken, Errno> {
        if self.specs.len() >= WAIT_MANY_MAX_ITEMS {
            return Err(Errno::ENOSPC);
        }
        let tok = self.alloc_token();
        self.specs.push(WaitSpec {
            kind: kind as u32,
            flags,
            object,
            token: tok.0,
        });
        Ok(tok)
    }

    // ── registration API ──────────────────────────────────────────────────

    /// Watch a port for incoming data.
    ///
    /// `handle` is the **read** end of the port (as returned by the low
    /// half of `channel_create`).
    pub fn add_port_readable(&mut self, handle: u64) -> Result<WaitToken, Errno> {
        self.push_spec(WaitKind::Port, interest::READABLE, handle)
    }

    /// Watch a port for write space.
    ///
    /// `handle` is the **write** end of the port.
    pub fn add_port_writable(&mut self, handle: u64) -> Result<WaitToken, Errno> {
        self.push_spec(WaitKind::Port, interest::WRITABLE, handle)
    }

    /// Watch a VFS file descriptor for readability.
    ///
    /// `fd` is any open file descriptor: a pipe read-end, a socket, a channel
    /// end that was bridged via `SYS_FS_FD_FROM_HANDLE`, or a device node.
    /// The waiter wakes when the underlying node reports `POLLIN`.
    ///
    /// This is the recommended API for FD-based readiness.  The older
    /// [`add_vfs_watch`][Self::add_vfs_watch] alias is deprecated.
    pub fn add_fd_readable(&mut self, fd: u32) -> Result<WaitToken, Errno> {
        self.push_spec(WaitKind::Fd, interest::READABLE, fd as u64)
    }

    /// Watch a VFS file descriptor for writability.
    ///
    /// `fd` is any open file descriptor: a pipe write-end, a socket, a channel
    /// end that was bridged via `SYS_FS_FD_FROM_HANDLE`, or a device node.
    /// The waiter wakes when the underlying node reports `POLLOUT`.
    pub fn add_fd_writable(&mut self, fd: u32) -> Result<WaitToken, Errno> {
        self.push_spec(WaitKind::Fd, interest::WRITABLE, fd as u64)
    }

    /// Watch a VFS file descriptor for readability.
    ///
    /// # Deprecated
    ///
    /// Use [`add_fd_readable`][Self::add_fd_readable] instead.  This alias
    /// exists only for backward compatibility.
    #[deprecated(note = "Use add_fd_readable instead")]
    pub fn add_vfs_watch(&mut self, fd: u32) -> Result<WaitToken, Errno> {
        self.add_fd_readable(fd)
    }

    /// Watch for a task to exit.
    ///
    /// `tid` is the numeric task ID of the child task.
    pub fn add_task_exit(&mut self, tid: u64) -> Result<WaitToken, Errno> {
        self.push_spec(WaitKind::TaskExit, 0, tid)
    }

    /// Watch for an IRQ to fire.
    ///
    /// `irq_handle` is the handle returned by `device_irq_subscribe`.
    pub fn add_irq(&mut self, irq_handle: u64) -> Result<WaitToken, Errno> {
        self.push_spec(WaitKind::Irq, 0, irq_handle)
    }

    /// Watch for an async graph-operation to complete.
    ///
    /// # Deprecated
    ///
    /// Graph operations are removed.  The kernel returns `ENOSYS` for
    /// `WaitKind::GraphOp`.  There is no direct replacement: async I/O should
    /// be modelled as FD readiness via [`add_fd_readable`][Self::add_fd_readable].
    #[deprecated(note = "Graph ops are removed; model async I/O as FD readiness with add_fd_readable")]
    pub fn add_graph_op(&mut self, op_handle: u64) -> Result<WaitToken, Errno> {
        #[allow(deprecated)]
        self.push_spec(WaitKind::GraphOp, 0, op_handle)
    }

    // ── removal ──────────────────────────────────────────────────────────

    /// Remove the event source identified by `token`.
    ///
    /// Returns `true` if the token was found and removed, `false` if it was
    /// not present (already removed or never added).
    pub fn remove(&mut self, token: WaitToken) -> bool {
        if let Some(i) = self.specs.iter().position(|s| s.token == token.0) {
            self.specs.swap_remove(i);
            true
        } else {
            false
        }
    }

    // ── introspection ────────────────────────────────────────────────────

    /// Number of event sources currently registered.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// `true` when no event sources are registered.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    // ── wait ─────────────────────────────────────────────────────────────

    /// Block until at least one registered source becomes ready.
    ///
    /// - `timeout = None` — wait indefinitely.
    /// - `timeout = Some(d)` — return after at most `d`, even if nothing fired
    ///   (the returned [`WaitEvents`] may be empty in that case).
    ///
    /// The `timeout` parameter accepts any type that converts to
    /// [`Duration`][crate::time::Duration], including `core::time::Duration`.
    ///
    /// Returns an error if the set is empty or if the syscall fails.
    pub fn wait<D>(&self, timeout: Option<D>) -> Result<WaitEvents, Errno>
    where
        D: Into<Duration>,
    {
        if self.specs.is_empty() {
            return Err(Errno::EINVAL);
        }

        let timeout_dur = timeout.map(Into::into);
        let mut results = [WaitResult::default(); WAIT_MANY_MAX_ITEMS];
        // push_spec() guarantees specs.len() <= WAIT_MANY_MAX_ITEMS.
        let n =
            syscall::wait::wait_many(&self.specs, &mut results[..self.specs.len()], timeout_dur)?;
        if n > self.specs.len() || n > results.len() {
            return Err(Errno::EIO);
        }

        let events = results[..n]
            .iter()
            .map(|r| WaitEvent { result: *r })
            .collect();
        Ok(WaitEvents(events))
    }
}

impl Default for WaitSet {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use abi::wait::WaitKind;

    #[test]
    fn new_set_is_empty() {
        let set = WaitSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn add_increases_len() {
        let mut set = WaitSet::new();
        let t1 = set.add_port_readable(1).unwrap();
        assert_eq!(set.len(), 1);
        let _t2 = set.add_fd_readable(2).unwrap();
        assert_eq!(set.len(), 2);
        // tokens are unique
        assert_ne!(t1.0, _t2.0);
    }

    #[test]
    fn remove_returns_true_when_present() {
        let mut set = WaitSet::new();
        let tok = set.add_port_readable(1).unwrap();
        assert!(set.remove(tok));
        assert!(set.is_empty());
    }

    #[test]
    fn remove_returns_false_when_absent() {
        let mut set = WaitSet::new();
        assert!(!set.remove(WaitToken(99)));
    }

    #[test]
    fn wait_on_empty_set_returns_einval() {
        let set = WaitSet::new();
        let res = set.wait(None::<Duration>);
        assert_eq!(res.unwrap_err(), Errno::EINVAL);
    }

    #[test]
    fn exceeds_max_items_returns_enospc() {
        let mut set = WaitSet::new();
        for i in 0..WAIT_MANY_MAX_ITEMS as u64 {
            set.add_port_readable(i).unwrap();
        }
        // One more should fail
        let res = set.add_port_readable(99);
        assert_eq!(res.unwrap_err(), Errno::ENOSPC);
    }

    #[test]
    fn specs_have_correct_kind() {
        let mut set = WaitSet::new();
        let _ = set.add_port_readable(10).unwrap();
        assert_eq!(set.specs[0].kind, WaitKind::Port as u32);
        assert_eq!(set.specs[0].flags, interest::READABLE);
        assert_eq!(set.specs[0].object, 10);

        let _ = set.add_fd_readable(5).unwrap();
        assert_eq!(set.specs[1].kind, WaitKind::Fd as u32);
        assert_eq!(set.specs[1].object, 5);
    }

    #[test]
    fn wait_event_flag_helpers() {
        use abi::wait::ready;
        let r = WaitResult {
            kind: WaitKind::Port as u32,
            flags: ready::READABLE | ready::HANGUP,
            object: 1,
            token: 42,
            value: 128,
            reserved: 0,
        };
        let ev = WaitEvent { result: r };
        assert!(ev.is_readable());
        assert!(ev.is_hangup());
        assert!(!ev.is_writable());
        assert!(!ev.is_exited());
        assert_eq!(ev.token(), WaitToken(42));
        assert_eq!(ev.value(), 128);
    }

    #[test]
    fn add_fd_readable_creates_fd_spec_with_readable_interest() {
        let mut set = WaitSet::new();
        let tok = set.add_fd_readable(7).unwrap();
        assert_eq!(set.len(), 1);
        assert_eq!(set.specs[0].kind, WaitKind::Fd as u32);
        assert_eq!(set.specs[0].flags, interest::READABLE);
        assert_eq!(set.specs[0].object, 7);
        assert_eq!(set.specs[0].token, tok.0);
    }

    #[test]
    fn add_fd_writable_creates_fd_spec_with_writable_interest() {
        let mut set = WaitSet::new();
        let tok = set.add_fd_writable(9).unwrap();
        assert_eq!(set.len(), 1);
        assert_eq!(set.specs[0].kind, WaitKind::Fd as u32);
        assert_eq!(set.specs[0].flags, interest::WRITABLE);
        assert_eq!(set.specs[0].object, 9);
        assert_eq!(set.specs[0].token, tok.0);
    }

    #[test]
    fn fd_readable_and_writable_tokens_are_distinct() {
        let mut set = WaitSet::new();
        let tr = set.add_fd_readable(4).unwrap();
        let tw = set.add_fd_writable(4).unwrap();
        assert_ne!(tr, tw, "readable and writable registrations get unique tokens");
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn remove_fd_readable_entry() {
        let mut set = WaitSet::new();
        let tok = set.add_fd_readable(3).unwrap();
        assert!(set.remove(tok));
        assert!(set.is_empty());
    }
}
