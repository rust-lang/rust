//! RTL8168 driver protocol messages
//!
//! Defines the IPC message format between rtl8168d and netd.
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use alloc::vec::Vec;

/// Frame received from hardware (rtl8168d -> netd)
pub const MSG_FRAME_RX: u16 = 0x0001;
/// Frame to transmit to hardware (netd -> rtl8168d)
pub const MSG_FRAME_TX: u16 = 0x0002;
/// Request MAC address (netd -> rtl8168d)
pub const MSG_MAC_REQ: u16 = 0x0010;
/// MAC address response (rtl8168d -> netd)
pub const MSG_MAC_RESP: u16 = 0x0011;

#[derive(Debug)]
pub struct NetDriverMsg<'a> {
    pub msg_type: u16,
    pub payload: &'a [u8],
}

impl<'a> NetDriverMsg<'a> {
    pub fn new(msg_type: u16, payload: &'a [u8]) -> Self {
        Self { msg_type, payload }
    }

    pub fn encode(&self) -> Vec<u8> {
        let len = self.payload.len() as u16;
        let mut buf = Vec::with_capacity(4 + self.payload.len());
        buf.extend_from_slice(&self.msg_type.to_le_bytes());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(self.payload);
        buf
    }

    pub fn decode(data: &'a [u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let msg_type = u16::from_le_bytes([data[0], data[1]]);
        let len = u16::from_le_bytes([data[2], data[3]]) as usize;
        if data.len() < 4 + len {
            return None;
        }
        Some(Self {
            msg_type,
            payload: &data[4..4 + len],
        })
    }
}
