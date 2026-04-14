//! Structured request/reply header for channel-based RPC.
//!
//! This module provides [`RpcHeader`] — a small framing header prepended to
//! channel messages that participate in request/reply RPC.  It enables
//! correlation of replies to requests without requiring out-of-band signalling.
//!
//! # Doctrine
//!
//! Not every service needs RPC framing.  Use [`RpcHeader`] when:
//!
//! - A client needs to match replies to specific requests (request IDs).
//! - A service wants to support concurrent in-flight requests.
//! - Timeout or cancellation logic needs to identify individual operations.
//!
//! For fire-and-forget events and simple notification streams, omit the header
//! and use raw `channel_send` / `channel_recv`.
//!
//! # Wire Format
//!
//! ```text
//! offset  size  field
//!      0     8  request_id  – caller-assigned unique ID; echoed in reply
//!      8     1  flags       – RPC_FLAG_REQUEST / RPC_FLAG_REPLY / …
//!      9     5  _pad        – reserved, must be zero
//! ```
//!
//! Total: [`RpcHeader::WIRE_SIZE`] bytes (14 bytes).
//!
//! The application payload follows immediately after the header.
//!
//! # Usage
//!
//! **Sender (client)**
//! ```ignore
//! use abi::rpc::{RpcHeader, RPC_FLAG_REQUEST};
//!
//! let hdr = RpcHeader { request_id: 1, flags: RPC_FLAG_REQUEST, _pad: [0; 5] };
//! let mut buf = [0u8; 512];
//! hdr.encode_le(&mut buf[..RpcHeader::WIRE_SIZE]).unwrap();
//! buf[RpcHeader::WIRE_SIZE..].copy_from_slice(b"my-payload");
//! channel_send_all(write_handle, &buf[..RpcHeader::WIRE_SIZE + 10]).unwrap();
//! ```
//!
//! **Receiver (server)**
//! ```ignore
//! use abi::rpc::RpcHeader;
//!
//! let n = channel_recv(read_handle, &mut buf).unwrap();
//! let hdr = RpcHeader::decode_le(&buf[..RpcHeader::WIRE_SIZE]).unwrap();
//! let payload = &buf[RpcHeader::WIRE_SIZE..n];
//! // … process payload …
//! // Build reply with hdr.request_id echoed back.
//! ```

/// Flag bit: this message is a request.
pub const RPC_FLAG_REQUEST: u8 = 1 << 0;
/// Flag bit: this message is a reply to a prior request.
pub const RPC_FLAG_REPLY: u8 = 1 << 1;
/// Flag bit: this reply carries an error; the payload is a u32 errno value.
pub const RPC_FLAG_ERROR: u8 = 1 << 2;
/// Flag bit: this request is a one-way notification; no reply is expected.
pub const RPC_FLAG_ONEWAY: u8 = 1 << 3;

/// Framing header for channel-based request/reply RPC.
///
/// Embed this at the start of every channel message that participates in
/// request/reply correlation.  The server echoes `request_id` back in the
/// reply so the client can match it.
///
/// Wire size: [`RpcHeader::WIRE_SIZE`] bytes, little-endian.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpcHeader {
    /// Caller-assigned unique identifier for this request.
    ///
    /// The server must echo this value verbatim in the reply header.
    /// Clients may use any numbering scheme; monotonically increasing
    /// integers per channel are conventional.
    pub request_id: u64,
    /// Flags: `RPC_FLAG_REQUEST`, `RPC_FLAG_REPLY`, `RPC_FLAG_ERROR`, etc.
    pub flags: u8,
    /// Reserved, must be zero.
    pub _pad: [u8; 5],
}

impl RpcHeader {
    /// Byte length of the encoded header.
    pub const WIRE_SIZE: usize = 14;

    /// Encode this header into the first [`WIRE_SIZE`] bytes of `out`.
    ///
    /// Returns `Some(WIRE_SIZE)` on success, `None` if `out` is too short.
    pub fn encode_le(&self, out: &mut [u8]) -> Option<usize> {
        if out.len() < Self::WIRE_SIZE {
            return None;
        }
        out[0..8].copy_from_slice(&self.request_id.to_le_bytes());
        out[8] = self.flags;
        out[9..14].copy_from_slice(&self._pad);
        Some(Self::WIRE_SIZE)
    }

    /// Decode a header from the first [`WIRE_SIZE`] bytes of `src`.
    ///
    /// Returns `None` if `src` is too short.
    pub fn decode_le(src: &[u8]) -> Option<Self> {
        if src.len() < Self::WIRE_SIZE {
            return None;
        }
        let request_id = u64::from_le_bytes(src[0..8].try_into().unwrap());
        let flags = src[8];
        let _pad: [u8; 5] = src[9..14].try_into().unwrap();
        Some(Self {
            request_id,
            flags,
            _pad,
        })
    }

    /// Returns `true` if this is a request message.
    #[inline]
    pub fn is_request(&self) -> bool {
        self.flags & RPC_FLAG_REQUEST != 0
    }

    /// Returns `true` if this is a reply message.
    #[inline]
    pub fn is_reply(&self) -> bool {
        self.flags & RPC_FLAG_REPLY != 0
    }

    /// Returns `true` if this reply carries an error.
    #[inline]
    pub fn is_error(&self) -> bool {
        self.flags & RPC_FLAG_ERROR != 0
    }

    /// Returns `true` if no reply is expected.
    #[inline]
    pub fn is_oneway(&self) -> bool {
        self.flags & RPC_FLAG_ONEWAY != 0
    }

    /// Build a request header with the given `request_id`.
    #[inline]
    pub const fn request(request_id: u64) -> Self {
        Self {
            request_id,
            flags: RPC_FLAG_REQUEST,
            _pad: [0; 5],
        }
    }

    /// Build a reply header echoing `request_id`.
    #[inline]
    pub const fn reply(request_id: u64) -> Self {
        Self {
            request_id,
            flags: RPC_FLAG_REPLY,
            _pad: [0; 5],
        }
    }

    /// Build an error-reply header echoing `request_id`.
    #[inline]
    pub const fn error_reply(request_id: u64) -> Self {
        Self {
            request_id,
            flags: RPC_FLAG_REPLY | RPC_FLAG_ERROR,
            _pad: [0; 5],
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn wire_size_matches_struct_layout() {
        assert_eq!(
            RpcHeader::WIRE_SIZE,
            8 + 1 + 5, // request_id + flags + _pad
        );
    }

    #[test]
    fn round_trip_request() {
        let hdr = RpcHeader::request(0xDEAD_BEEF_0000_0001);
        let mut buf = [0u8; RpcHeader::WIRE_SIZE];
        let n = hdr.encode_le(&mut buf).unwrap();
        assert_eq!(n, RpcHeader::WIRE_SIZE);

        let decoded = RpcHeader::decode_le(&buf).unwrap();
        assert_eq!(decoded.request_id, 0xDEAD_BEEF_0000_0001);
        assert!(decoded.is_request());
        assert!(!decoded.is_reply());
        assert!(!decoded.is_error());
    }

    #[test]
    fn round_trip_reply() {
        let hdr = RpcHeader::reply(42);
        let mut buf = [0u8; RpcHeader::WIRE_SIZE];
        hdr.encode_le(&mut buf).unwrap();
        let decoded = RpcHeader::decode_le(&buf).unwrap();
        assert_eq!(decoded.request_id, 42);
        assert!(decoded.is_reply());
        assert!(!decoded.is_request());
    }

    #[test]
    fn round_trip_error_reply() {
        let hdr = RpcHeader::error_reply(99);
        let mut buf = [0u8; RpcHeader::WIRE_SIZE];
        hdr.encode_le(&mut buf).unwrap();
        let decoded = RpcHeader::decode_le(&buf).unwrap();
        assert!(decoded.is_reply());
        assert!(decoded.is_error());
        assert_eq!(decoded.request_id, 99);
    }

    #[test]
    fn decode_too_short_returns_none() {
        let buf = [0u8; RpcHeader::WIRE_SIZE - 1];
        assert!(RpcHeader::decode_le(&buf).is_none());
    }

    #[test]
    fn encode_too_small_buffer_returns_none() {
        let hdr = RpcHeader::request(1);
        let mut buf = [0u8; RpcHeader::WIRE_SIZE - 1];
        assert!(hdr.encode_le(&mut buf).is_none());
    }

    #[test]
    fn golden_bytes_request() {
        // request_id=1, flags=RPC_FLAG_REQUEST(0x01), _pad=0
        let hdr = RpcHeader::request(1);
        let mut buf = [0u8; RpcHeader::WIRE_SIZE];
        hdr.encode_le(&mut buf).unwrap();
        assert_eq!(
            buf,
            [
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // request_id = 1
                0x01, // flags = REQUEST
                0x00, 0x00, 0x00, 0x00, 0x00, // _pad
            ]
        );
    }
}
