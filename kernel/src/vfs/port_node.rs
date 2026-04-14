//! VFS node wrapper for IPC ports.
//!
//! This allows IPC ports to be treated as VFS nodes, enabling them to be
//! passed across channels using the standard handle-passing mechanism.

use super::{VfsNode, VfsStat};
use crate::ipc::{HandleMode, Port};
use abi::errors::SysResult;
use alloc::sync::Arc;

/// A VFS node that wraps an IPC port.
pub struct PortNode {
    port: Arc<Port>,
    mode: HandleMode,
}

impl PortNode {
    pub fn new(port: Arc<Port>, mode: HandleMode) -> Self {
        Self { port, mode }
    }

    pub fn port(&self) -> &Arc<Port> {
        &self.port
    }
}

impl VfsNode for PortNode {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        if self.mode != HandleMode::Read {
            return Err(abi::errors::Errno::EBADF);
        }
        Ok(self.port.try_recv(buf))
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> SysResult<usize> {
        if self.mode != HandleMode::Write {
            return Err(abi::errors::Errno::EBADF);
        }
        Ok(self.port.send(buf))
    }

    fn stat(&self) -> SysResult<VfsStat> {
        let (r, w) = if self.mode == HandleMode::Read {
            (0o400, 0)
        } else {
            (0, 0o200)
        };
        Ok(VfsStat {
            mode: VfsStat::S_IFIFO | r | w,
            size: self.port.len() as u64,
            ino: 0,
            ..Default::default()
        })
    }

    fn poll(&self) -> u16 {
        use abi::syscall::poll_flags::{POLLERR, POLLHUP, POLLIN, POLLOUT};
        let mut revents = 0;
        match self.mode {
            HandleMode::Read => {
                if !self.port.is_empty() || !self.port.has_writers() {
                    revents |= POLLIN;
                }
                if !self.port.has_writers() {
                    revents |= POLLHUP;
                }
            }
            HandleMode::Write => {
                if !self.port.has_readers() {
                    revents |= POLLHUP | POLLERR;
                } else if !self.port.is_full() {
                    revents |= POLLOUT;
                }
            }
        }
        revents
    }

    fn close(&self) {
        match self.mode {
            HandleMode::Read => {
                self.port.close_reader();
            }
            HandleMode::Write => {
                self.port.close_writer();
            }
        }
    }

    fn add_waiter(&self, tid: u64) {
        match self.mode {
            HandleMode::Read => self.port.add_waiter_read(tid),
            HandleMode::Write => self.port.add_waiter_write(tid),
        }
    }

    fn remove_waiter(&self, tid: u64) {
        match self.mode {
            HandleMode::Read => self.port.remove_waiter_read(tid),
            HandleMode::Write => self.port.remove_waiter_write(tid),
        }
    }

    fn as_port(&self) -> Option<Arc<Port>> {
        Some(self.port.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi::syscall::poll_flags;

    fn make_channel(capacity: usize) -> (Arc<PortNode>, Arc<PortNode>) {
        let port = Arc::new(Port::new(capacity));
        let read_node = Arc::new(PortNode::new(Arc::clone(&port), HandleMode::Read));
        let write_node = Arc::new(PortNode::new(Arc::clone(&port), HandleMode::Write));
        (read_node, write_node)
    }

    // ── Read-handle poll semantics ─────────────────────────────────────────

    #[test]
    fn read_handle_not_ready_when_empty_with_writer() {
        let (r, _w) = make_channel(64);
        // Empty queue, writer still alive → not readable.
        assert_eq!(r.poll() & poll_flags::POLLIN, 0);
    }

    #[test]
    fn read_handle_ready_after_send() {
        let (r, w) = make_channel(64);
        w.write(0, b"hi").expect("write");
        assert_ne!(r.poll() & poll_flags::POLLIN, 0, "POLLIN set after send");
    }

    #[test]
    fn read_handle_reports_pollhup_when_writer_closed() {
        let (r, w) = make_channel(64);
        // Closing the write handle signals hangup on the read handle.
        w.close();
        assert_ne!(r.poll() & poll_flags::POLLIN, 0, "POLLIN set on EOF");
        assert_ne!(
            r.poll() & poll_flags::POLLHUP,
            0,
            "POLLHUP set when writer closed"
        );
    }

    #[test]
    fn read_handle_reports_both_pollin_and_pollhup_with_data_and_closed_writer() {
        let (r, w) = make_channel(64);
        w.write(0, b"x").expect("write");
        w.close();
        let flags = r.poll();
        assert_ne!(
            flags & poll_flags::POLLIN,
            0,
            "POLLIN set when data present"
        );
        assert_ne!(
            flags & poll_flags::POLLHUP,
            0,
            "POLLHUP set when writer gone"
        );
    }

    // ── Write-handle poll semantics ────────────────────────────────────────

    #[test]
    fn write_handle_ready_when_queue_has_space() {
        let (_r, w) = make_channel(64);
        assert_ne!(
            w.poll() & poll_flags::POLLOUT,
            0,
            "POLLOUT when space available"
        );
    }

    #[test]
    fn write_handle_reports_pollhup_when_reader_closed() {
        let (r, w) = make_channel(64);
        // Closing the read handle signals broken-pipe on the write handle.
        r.close();
        let flags = w.poll();
        assert_ne!(flags & poll_flags::POLLHUP, 0, "POLLHUP when reader closed");
        assert_ne!(flags & poll_flags::POLLERR, 0, "POLLERR when reader closed");
    }

    #[test]
    fn write_handle_not_pollout_when_queue_full() {
        // Use a small capacity so we can fill it.
        let (_r, w) = make_channel(16);
        // Fill the queue completely.
        w.write(0, &[0u8; 16]).expect("write");
        assert_eq!(
            w.poll() & poll_flags::POLLOUT,
            0,
            "POLLOUT clear when queue full"
        );
    }
}
