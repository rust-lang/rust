//! VFS provider server loop.
//!
//! [`ProviderLoop`] abstracts the channel framing for a userland VFS provider.
//! It reads requests from the provider channel, dispatches them to your
//! handler, and sends responses back to the kernel.
//!
//! # Usage
//!
//! ```ignore
//! use ipc_helpers::provider::{ProviderLoop, ProviderResponse};
//! use abi::vfs_rpc::VfsRpcOp;
//! use abi::errors::Errno;
//!
//! fn run(vfs_read: u32) {
//!     let mut lp = ProviderLoop::new(vfs_read);
//!     loop {
//!         let req = match lp.next_request() {
//!             Ok(r) => r,
//!             Err(_) => break,  // channel closed — exit cleanly
//!         };
//!         let resp = dispatch(&req);
//!         lp.send_response(req.resp_port, resp).ok();
//!     }
//! }
//!
//! fn dispatch(req: &ipc_helpers::provider::ProviderRequest) -> ProviderResponse {
//!     match req.op {
//!         VfsRpcOp::Lookup => ProviderResponse::ok_u64(1),
//!         VfsRpcOp::Read   => ProviderResponse::ok_bytes(b"Hello, world!\n"),
//!         VfsRpcOp::Stat   => ProviderResponse::ok_stat(0o100644, 14, 1),
//!         VfsRpcOp::Close  => ProviderResponse::ok_empty(),
//!         _                => ProviderResponse::err(Errno::ENOSYS),
//!     }
//! }
//! ```

use abi::errors::Errno;
use abi::vfs_rpc::{VfsRpcOp, VfsRpcReqHeader, VFS_RPC_MAX_REQ, VFS_RPC_MAX_RESP};
use stem::syscall::channel::{channel_recv, channel_send_all};

/// A decoded VFS RPC request from the kernel.
pub struct ProviderRequest {
    /// The response port the kernel is waiting on.  Pass this to
    /// [`ProviderLoop::send_response`].
    pub resp_port: u32,
    /// The operation code.
    pub op: VfsRpcOp,
    /// The op-specific payload bytes (after the 7-byte header).
    pub payload: alloc::vec::Vec<u8>,
}

/// A response to be sent back to the kernel.
#[derive(Debug)]
pub struct ProviderResponse {
    /// 0 = OK, non-zero = errno.
    pub status: u8,
    /// Payload bytes (only meaningful when `status == 0`).
    pub payload: alloc::vec::Vec<u8>,
}

impl ProviderResponse {
    /// Successful response with no payload (e.g. `Close`).
    pub fn ok_empty() -> Self {
        Self {
            status: 0,
            payload: alloc::vec![],
        }
    }

    /// Successful response with a raw byte payload.
    pub fn ok_bytes(data: &[u8]) -> Self {
        Self {
            status: 0,
            payload: alloc::vec::Vec::from(data),
        }
    }

    /// Successful `Lookup` response carrying a `u64` handle.
    pub fn ok_u64(value: u64) -> Self {
        let mut payload = alloc::vec![0u8; 8];
        payload[..8].copy_from_slice(&value.to_le_bytes());
        Self { status: 0, payload }
    }

    /// Successful `Stat` response carrying `mode`, `size`, `ino`.
    pub fn ok_stat(mode: u32, size: u64, ino: u64) -> Self {
        let mut payload = alloc::vec![0u8; 4 + 8 + 8];
        payload[0..4].copy_from_slice(&mode.to_le_bytes());
        payload[4..12].copy_from_slice(&size.to_le_bytes());
        payload[12..20].copy_from_slice(&ino.to_le_bytes());
        Self { status: 0, payload }
    }

    /// Successful `Read`/`Readdir` response.
    ///
    /// Prepends the 4-byte `bytes_read` count as required by the wire format.
    pub fn ok_read(data: &[u8]) -> Self {
        let mut payload = alloc::vec![0u8; 4 + data.len()];
        let len = data.len() as u32;
        payload[0..4].copy_from_slice(&len.to_le_bytes());
        payload[4..].copy_from_slice(data);
        Self { status: 0, payload }
    }

    /// Successful `Write` response with the number of bytes written.
    pub fn ok_written(n: u32) -> Self {
        let mut payload = alloc::vec![0u8; 4];
        payload[0..4].copy_from_slice(&n.to_le_bytes());
        Self { status: 0, payload }
    }

    /// Error response carrying an errno.
    pub fn err(e: Errno) -> Self {
        Self {
            status: e as u8,
            payload: alloc::vec![],
        }
    }
}

/// The VFS provider server loop.
///
/// Owns the read end of the provider channel.  Call [`next_request`] in a
/// loop to receive decoded requests, then call [`send_response`] to reply.
pub struct ProviderLoop {
    read_handle: u32,
    buf: alloc::vec::Vec<u8>,
}

impl ProviderLoop {
    /// Create a new loop bound to `vfs_read` — the read end of the provider
    /// channel (created with `channel_create` and passed to the supervisor).
    pub fn new(vfs_read: u32) -> Self {
        Self {
            read_handle: vfs_read,
            buf: alloc::vec![0u8; VFS_RPC_MAX_REQ],
        }
    }

    /// Block until the next request arrives and decode it.
    ///
    /// Returns `Err(Errno::EPIPE)` when the channel is closed (provider
    /// should exit cleanly).
    pub fn next_request(&mut self) -> Result<ProviderRequest, Errno> {
        let n = channel_recv(self.read_handle, &mut self.buf)?;

        let hdr_size = core::mem::size_of::<VfsRpcReqHeader>();
        if n < hdr_size {
            return Err(Errno::EINVAL);
        }

        // SAFETY: we checked n >= hdr_size; the header struct is repr(C,packed).
        let hdr: VfsRpcReqHeader = unsafe {
            core::ptr::read_unaligned(self.buf.as_ptr() as *const VfsRpcReqHeader)
        };

        let op = VfsRpcOp::from_u8(hdr.op).ok_or(Errno::EINVAL)?;
        let payload = self.buf[hdr_size..n].to_vec();

        Ok(ProviderRequest {
            resp_port: hdr.resp_port,
            op,
            payload,
        })
    }

    /// Send `response` back to the kernel on the given `resp_port`.
    pub fn send_response(
        &self,
        resp_port: u32,
        response: ProviderResponse,
    ) -> Result<(), Errno> {
        let total = 1 + response.payload.len();
        let mut buf = alloc::vec![0u8; total.min(VFS_RPC_MAX_RESP)];
        buf[0] = response.status;
        let payload_len = response.payload.len().min(buf.len() - 1);
        buf[1..1 + payload_len].copy_from_slice(&response.payload[..payload_len]);
        channel_send_all(resp_port, &buf[..1 + payload_len]).map(|_| ())
    }
}
