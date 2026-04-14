#![allow(dead_code)]

use core::mem::size_of;

pub const DISPLAY_MAGIC: u32 = 0x5448_4947;
pub const DISPLAY_VERSION: u16 = 0;

pub const MSG_HELLO: u16 = 1;
pub const MSG_INFO_REQ: u16 = 2;
pub const MSG_INFO_RESP: u16 = 3;
pub const MSG_BUFFER_REQ: u16 = 4;
pub const MSG_BUFFER_RESP: u16 = 5;
pub const MSG_PRESENT: u16 = 6;
pub const MSG_ACK: u16 = 7;
pub const MSG_ERR: u16 = 8;

pub const FORMAT_XRGB8888: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DisplayHeader {
    pub magic: u32,
    pub version: u16,
    pub msg_type: u16,
    pub payload_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InfoResp {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BufferResp {
    pub fd: u32,
    pub _pad: u32,
    pub size: u64,
    pub stride: u32,
    pub format: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PresentHeader {
    pub rect_count: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ErrResp {
    pub code: u32,
}

pub const HEADER_SIZE: usize = size_of::<DisplayHeader>();

pub fn encode_message(buf: &mut [u8], msg_type: u16, payload: &[u8]) -> Option<usize> {
    let total = HEADER_SIZE + payload.len();
    if buf.len() < total {
        return None;
    }

    let header = DisplayHeader {
        magic: DISPLAY_MAGIC,
        version: DISPLAY_VERSION,
        msg_type,
        payload_len: payload.len() as u32,
    };

    buf[0..4].copy_from_slice(&header.magic.to_le_bytes());
    buf[4..6].copy_from_slice(&header.version.to_le_bytes());
    buf[6..8].copy_from_slice(&header.msg_type.to_le_bytes());
    buf[8..12].copy_from_slice(&header.payload_len.to_le_bytes());

    if !payload.is_empty() {
        buf[HEADER_SIZE..total].copy_from_slice(payload);
    }

    Some(total)
}

pub fn parse_message(buf: &[u8]) -> Option<(DisplayHeader, &[u8])> {
    if buf.len() < HEADER_SIZE {
        return None;
    }

    let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
    let version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
    let msg_type = u16::from_le_bytes(buf[6..8].try_into().ok()?);
    let payload_len = u32::from_le_bytes(buf[8..12].try_into().ok()?);

    if magic != DISPLAY_MAGIC || version != DISPLAY_VERSION {
        return None;
    }

    let total = HEADER_SIZE + payload_len as usize;
    if buf.len() < total {
        return None;
    }

    let header = DisplayHeader {
        magic,
        version,
        msg_type,
        payload_len,
    };

    Some((header, &buf[HEADER_SIZE..total]))
}
