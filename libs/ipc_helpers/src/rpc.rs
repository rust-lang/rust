//! Typed request/reply RPC client and server helpers.
//!
//! These helpers layer [`abi::rpc::RpcHeader`] framing on top of channels to
//! provide a request/reply RPC substrate without the ceremony of manual header
//! encoding/decoding.
//!
//! # Example: server
//!
//! ```ignore
//! use ipc_helpers::rpc::RpcServer;
//!
//! let mut server = RpcServer::new(read_handle);
//! loop {
//!     let req = server.next().unwrap();
//!     let payload: &[u8] = &req.payload;
//!     server.reply(req.request_id, write_handle, b"pong").unwrap();
//! }
//! ```
//!
//! # Example: client
//!
//! ```ignore
//! use ipc_helpers::rpc::RpcClient;
//!
//! let mut client = RpcClient::new(write_handle, read_handle);
//! let reply_payload = client.call(b"ping").unwrap();
//! ```

use abi::errors::Errno;
use abi::rpc::{RpcHeader, RPC_FLAG_ERROR, RPC_FLAG_REPLY};
use core::sync::atomic::{AtomicU64, Ordering};
use stem::syscall::channel::{channel_recv, channel_send_all, ChannelHandle};

// ── RpcRequest ────────────────────────────────────────────────────────────────

/// A decoded RPC request received by a server.
pub struct RpcRequest {
    /// The correlation ID to echo back in the reply.
    pub request_id: u64,
    /// Application payload (everything after the RPC header).
    pub payload: alloc::vec::Vec<u8>,
}

// ── RpcServer ─────────────────────────────────────────────────────────────────

/// A simple RPC server bound to a single channel read handle.
///
/// Blocks in `next()` until a request arrives.
pub struct RpcServer {
    read_handle: ChannelHandle,
    buf: alloc::vec::Vec<u8>,
}

impl RpcServer {
    /// Create a new server reading from `read_handle`.
    pub fn new(read_handle: ChannelHandle) -> Self {
        Self {
            read_handle,
            buf: alloc::vec![0u8; 4096],
        }
    }

    /// Block until the next request arrives and decode it.
    pub fn next(&mut self) -> Result<RpcRequest, Errno> {
        let n = channel_recv(self.read_handle, &mut self.buf)?;
        if n < RpcHeader::WIRE_SIZE {
            return Err(Errno::EINVAL);
        }
        let hdr = RpcHeader::decode_le(&self.buf[..RpcHeader::WIRE_SIZE])
            .ok_or(Errno::EINVAL)?;
        if !hdr.is_request() {
            return Err(Errno::EINVAL);
        }
        let payload = self.buf[RpcHeader::WIRE_SIZE..n].to_vec();
        Ok(RpcRequest {
            request_id: hdr.request_id,
            payload,
        })
    }

    /// Send a successful reply for `request_id` with `payload`.
    pub fn reply(
        &self,
        request_id: u64,
        write_handle: ChannelHandle,
        payload: &[u8],
    ) -> Result<(), Errno> {
        send_reply(write_handle, request_id, RPC_FLAG_REPLY, payload)
    }

    /// Send an error reply for `request_id` with an errno payload.
    pub fn reply_err(
        &self,
        request_id: u64,
        write_handle: ChannelHandle,
        errno: Errno,
    ) -> Result<(), Errno> {
        let code = (errno as u32).to_le_bytes();
        send_reply(
            write_handle,
            request_id,
            RPC_FLAG_REPLY | RPC_FLAG_ERROR,
            &code,
        )
    }
}

// ── RpcClient ─────────────────────────────────────────────────────────────────

/// A simple synchronous RPC client.
///
/// Sends requests and blocks waiting for the matching reply.  Not suitable
/// for concurrent in-flight requests (use a higher-level multiplexer for
/// that).
pub struct RpcClient {
    write_handle: ChannelHandle,
    read_handle: ChannelHandle,
    next_id: AtomicU64,
    buf: spin::Mutex<alloc::vec::Vec<u8>>,
}

impl RpcClient {
    /// Create a new client using `write_handle` to send and `read_handle`
    /// to receive replies.
    pub fn new(write_handle: ChannelHandle, read_handle: ChannelHandle) -> Self {
        Self {
            write_handle,
            read_handle,
            next_id: AtomicU64::new(1),
            buf: spin::Mutex::new(alloc::vec![0u8; 4096]),
        }
    }

    /// Send `payload` as a request and block until the reply arrives.
    ///
    /// Returns the reply payload bytes.  Returns `Err(Errno::EREMOTEIO)` if
    /// the server replied with an error flag (use `call_raw` to inspect the
    /// reply header instead).
    pub fn call(&self, payload: &[u8]) -> Result<alloc::vec::Vec<u8>, Errno> {
        let (hdr, reply_payload) = self.call_raw(payload)?;
        if hdr.is_error() {
            // The error payload is a u32 errno; decode it if present.
            // We return EIO as a generic fallback if the code is unrecognised.
            if reply_payload.len() >= 4 {
                let _code = u32::from_le_bytes(reply_payload[..4].try_into().unwrap());
                // Best-effort mapping; callers that need the precise code use call_raw.
            }
            return Err(Errno::EIO);
        }
        Ok(reply_payload)
    }

    /// Like `call` but returns the raw `(RpcHeader, payload)` pair so the
    /// caller can inspect flags directly.
    pub fn call_raw(
        &self,
        payload: &[u8],
    ) -> Result<(RpcHeader, alloc::vec::Vec<u8>), Errno> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Encode request.
        let mut msg = alloc::vec![0u8; RpcHeader::WIRE_SIZE + payload.len()];
        let hdr = RpcHeader::request(request_id);
        hdr.encode_le(&mut msg[..RpcHeader::WIRE_SIZE]).unwrap();
        msg[RpcHeader::WIRE_SIZE..].copy_from_slice(payload);
        channel_send_all(self.write_handle, &msg).map_err(|e| e)?;

        // Block waiting for reply.
        let mut buf = self.buf.lock();
        loop {
            let n = channel_recv(self.read_handle, &mut buf)?;
            if n < RpcHeader::WIRE_SIZE {
                continue; // too short, skip
            }
            let reply_hdr =
                RpcHeader::decode_le(&buf[..RpcHeader::WIRE_SIZE]).ok_or(Errno::EINVAL)?;
            if reply_hdr.request_id != request_id {
                // Out-of-order reply (should not happen in synchronous usage) — skip.
                continue;
            }
            let reply_payload = buf[RpcHeader::WIRE_SIZE..n].to_vec();
            return Ok((reply_hdr, reply_payload));
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn send_reply(
    write_handle: ChannelHandle,
    request_id: u64,
    flags: u8,
    payload: &[u8],
) -> Result<(), Errno> {
    let mut msg = alloc::vec![0u8; RpcHeader::WIRE_SIZE + payload.len()];
    let hdr = RpcHeader {
        request_id,
        flags,
        _pad: [0; 5],
    };
    hdr.encode_le(&mut msg[..RpcHeader::WIRE_SIZE]).unwrap();
    msg[RpcHeader::WIRE_SIZE..].copy_from_slice(payload);
    channel_send_all(write_handle, &msg).map(|_| ())
}
