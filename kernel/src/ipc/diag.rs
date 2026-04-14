//! IPC diagnostics — global counters for channels, pipes, and VFS RPC.
//!
//! All counters are `AtomicU64` incremented at the hot path (no locking).
//! Expose them via `/proc/ipc/channels` and `/proc/ipc/pipes`.
//!
//! # Usage
//!
//! Increment from the relevant hot path:
//! ```ignore
//! crate::ipc::diag::CHANNEL_SENDS.fetch_add(1, Ordering::Relaxed);
//! ```
//!
//! Read for display:
//! ```ignore
//! let s = crate::ipc::diag::channels_text();
//! ```

use core::sync::atomic::{AtomicU64, Ordering};

// ── Channel counters ──────────────────────────────────────────────────────────

/// Total `channel_send` calls that wrote ≥1 byte.
pub static CHANNEL_SENDS: AtomicU64 = AtomicU64::new(0);
/// Total `channel_recv` calls that read ≥1 byte.
pub static CHANNEL_RECVS: AtomicU64 = AtomicU64::new(0);
/// Cumulative bytes written via `channel_send`.
pub static CHANNEL_BYTES_SENT: AtomicU64 = AtomicU64::new(0);
/// Cumulative bytes read via `channel_recv`.
pub static CHANNEL_BYTES_RECV: AtomicU64 = AtomicU64::new(0);
/// Total capability handles enqueued via `channel_send_handle`.
pub static CHANNEL_HANDLES_SENT: AtomicU64 = AtomicU64::new(0);
/// Total capability handles dequeued via `channel_recv_handle`.
pub static CHANNEL_HANDLES_RECV: AtomicU64 = AtomicU64::new(0);
/// Times `channel_send_all` returned `EAGAIN` because the ring was full.
pub static CHANNEL_FULL_EVENTS: AtomicU64 = AtomicU64::new(0);
/// Times a peer closure was observed (writer or reader).
pub static CHANNEL_PEER_DEATHS: AtomicU64 = AtomicU64::new(0);

// ── Pipe counters ─────────────────────────────────────────────────────────────

/// Total pipe write calls that wrote ≥1 byte.
pub static PIPE_WRITES: AtomicU64 = AtomicU64::new(0);
/// Total pipe read calls that read ≥1 byte.
pub static PIPE_READS: AtomicU64 = AtomicU64::new(0);
/// Cumulative bytes written to pipes.
pub static PIPE_BYTES_WRITTEN: AtomicU64 = AtomicU64::new(0);
/// Cumulative bytes read from pipes.
pub static PIPE_BYTES_READ: AtomicU64 = AtomicU64::new(0);
/// Times a pipe write returned `EPIPE` (broken pipe).
pub static PIPE_BROKEN_PIPE: AtomicU64 = AtomicU64::new(0);

// ── VFS RPC counters ──────────────────────────────────────────────────────────

/// Total VFS RPC requests dispatched to providers.
pub static VFS_RPC_REQUESTS: AtomicU64 = AtomicU64::new(0);
/// Total VFS RPC requests that completed with an error.
pub static VFS_RPC_ERRORS: AtomicU64 = AtomicU64::new(0);
/// Total VFS RPC requests that failed because the provider was dead.
pub static VFS_RPC_DEAD_PROVIDER: AtomicU64 = AtomicU64::new(0);

// ── Text renderers ────────────────────────────────────────────────────────────

/// Record a dead-provider error: increments both `VFS_RPC_ERRORS` and
/// `VFS_RPC_DEAD_PROVIDER`.  Call this whenever a VFS RPC send fails or the
/// response port returns `EPIPE` (provider exited mid-call).
#[inline]
pub fn record_dead_provider_error() {
    VFS_RPC_ERRORS.fetch_add(1, Ordering::Relaxed);
    VFS_RPC_DEAD_PROVIDER.fetch_add(1, Ordering::Relaxed);
}

/// Render channel counters as a human-readable text block (for `/proc/ipc/channels`).
pub fn channels_text() -> alloc::string::String {
    alloc::format!(
        "sends:         {}\n\
         recvs:         {}\n\
         bytes_sent:    {}\n\
         bytes_recv:    {}\n\
         handles_sent:  {}\n\
         handles_recv:  {}\n\
         full_events:   {}\n\
         peer_deaths:   {}\n",
        CHANNEL_SENDS.load(Ordering::Relaxed),
        CHANNEL_RECVS.load(Ordering::Relaxed),
        CHANNEL_BYTES_SENT.load(Ordering::Relaxed),
        CHANNEL_BYTES_RECV.load(Ordering::Relaxed),
        CHANNEL_HANDLES_SENT.load(Ordering::Relaxed),
        CHANNEL_HANDLES_RECV.load(Ordering::Relaxed),
        CHANNEL_FULL_EVENTS.load(Ordering::Relaxed),
        CHANNEL_PEER_DEATHS.load(Ordering::Relaxed),
    )
}

/// Render pipe counters as a human-readable text block (for `/proc/ipc/pipes`).
pub fn pipes_text() -> alloc::string::String {
    alloc::format!(
        "writes:        {}\n\
         reads:         {}\n\
         bytes_written: {}\n\
         bytes_read:    {}\n\
         broken_pipe:   {}\n",
        PIPE_WRITES.load(Ordering::Relaxed),
        PIPE_READS.load(Ordering::Relaxed),
        PIPE_BYTES_WRITTEN.load(Ordering::Relaxed),
        PIPE_BYTES_READ.load(Ordering::Relaxed),
        PIPE_BROKEN_PIPE.load(Ordering::Relaxed),
    )
}

/// Render VFS RPC counters as a human-readable text block (for `/proc/ipc/vfs_rpc`).
pub fn vfs_rpc_text() -> alloc::string::String {
    alloc::format!(
        "requests:      {}\n\
         errors:        {}\n\
         dead_provider: {}\n",
        VFS_RPC_REQUESTS.load(Ordering::Relaxed),
        VFS_RPC_ERRORS.load(Ordering::Relaxed),
        VFS_RPC_DEAD_PROVIDER.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_text_contains_expected_keys() {
        let t = channels_text();
        assert!(t.contains("sends:"));
        assert!(t.contains("bytes_sent:"));
        assert!(t.contains("handles_sent:"));
        assert!(t.contains("peer_deaths:"));
    }

    #[test]
    fn pipes_text_contains_expected_keys() {
        let t = pipes_text();
        assert!(t.contains("writes:"));
        assert!(t.contains("broken_pipe:"));
    }

    #[test]
    fn vfs_rpc_text_contains_expected_keys() {
        let t = vfs_rpc_text();
        assert!(t.contains("requests:"));
        assert!(t.contains("dead_provider:"));
    }
}
