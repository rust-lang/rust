//! VirtIO-NET VFS provider.
//!
//! Serves `/dev/net/virtio0/` with the following files:
//!
//! ```text
//! /dev/net/virtio0/
//! ├── ctl        ← write: "up", "down", "set-mtu <n>"
//! ├── status     ← read: human-readable device state
//! ├── mac        ← read: MAC address
//! ├── mtu        ← read/write: MTU value
//! ├── rx         ← read: stream of length-prefixed raw Ethernet frames
//! ├── tx         ← write: length-prefixed raw Ethernet frames
//! ├── features   ← read: virtio feature flags (hex text)
//! └── events     ← pollable: newline-delimited link events
//! ```
//!
//! Frame stream format on `rx` and `tx`:
//! ```text
//! [4 bytes: frame_length_le][frame_length bytes: raw Ethernet frame]
//! ```
extern crate alloc;

use abi::vfs_rpc::{VfsRpcOp, VfsRpcReqHeader};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use stem::info;
use stem::syscall::{channel_send, ChannelHandle};

use crate::driver::VirtioNetDriver;

// ── Handle IDs ────────────────────────────────────────────────────────────────

/// Root directory handle.
pub const HANDLE_ROOT: u64 = 0;
const HANDLE_CTL: u64 = 1;
const HANDLE_STATUS: u64 = 2;
const HANDLE_MAC: u64 = 3;
const HANDLE_MTU: u64 = 4;
const HANDLE_RX: u64 = 5;
const HANDLE_TX: u64 = 6;
const HANDLE_FEATURES: u64 = 7;
const HANDLE_EVENTS: u64 = 8;

// ── Mode bits ─────────────────────────────────────────────────────────────────

const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;

// ── Errno values ─────────────────────────────────────────────────────────────

const E_OK: u8 = 0;
const E_NOENT: u8 = 2;
const E_IO: u8 = 5;
const E_INVAL: u8 = 22;
const E_ROFS: u8 = 30;
const E_NOTSUP: u8 = 38;

// ── Poll readiness bits ───────────────────────────────────────────────────────

const POLLIN: u32 = 0x0001;

// ── Shared driver state ───────────────────────────────────────────────────────

/// Mutable state shared between the hardware poll loop and the VFS handler.
pub struct NetVfsState {
    /// MAC address.
    pub mac: [u8; 6],
    /// Whether the link is considered up.
    pub link_up: bool,
    /// Current MTU (default 1500).
    pub mtu: u32,
    /// Negotiated virtio feature flags (read from device after init).
    pub features: u32,
    /// Buffered received Ethernet frames (raw frame bytes, no length prefix).
    pub rx_queue: VecDeque<Vec<u8>>,
    /// Buffered link-state events (newline-terminated strings).
    pub events_queue: VecDeque<Vec<u8>>,
}

impl NetVfsState {
    /// Create a new state instance.
    pub fn new(mac: [u8; 6], link_up: bool, features: u32) -> Self {
        Self {
            mac,
            link_up,
            mtu: 1500,
            features,
            rx_queue: VecDeque::new(),
            events_queue: VecDeque::new(),
        }
    }

    /// Push a received Ethernet frame onto the RX queue.
    pub fn push_rx_frame(&mut self, frame: Vec<u8>) {
        self.rx_queue.push_back(frame);
    }

    /// Push a newline-terminated link-state event string.
    pub fn push_event(&mut self, event: &str) {
        let mut ev = Vec::from(event.as_bytes());
        if ev.last() != Some(&b'\n') {
            ev.push(b'\n');
        }
        self.events_queue.push_back(ev);
    }
}

// ── Wire helpers ──────────────────────────────────────────────────────────────

fn send_resp(resp_port: ChannelHandle, data: &[u8]) {
    let _ = channel_send(resp_port, data);
}

fn send_err(resp_port: ChannelHandle, errno: u8) {
    send_resp(resp_port, &[errno]);
}

/// Send `[E_OK][data.len(): u32 LE][data...]`.
fn send_ok_data(resp_port: ChannelHandle, data: &[u8]) {
    let mut resp = Vec::with_capacity(5 + data.len());
    resp.push(E_OK);
    resp.extend_from_slice(&(data.len() as u32).to_le_bytes());
    resp.extend_from_slice(data);
    send_resp(resp_port, &resp);
}

/// Send `[E_OK][bytes_written: u32 LE]`.
fn send_ok_written(resp_port: ChannelHandle, n: u32) {
    let mut resp = [0u8; 5];
    resp[0] = E_OK;
    resp[1..5].copy_from_slice(&n.to_le_bytes());
    send_resp(resp_port, &resp);
}

// ── VFS RPC dispatch ──────────────────────────────────────────────────────────

/// Dispatch one VFS RPC request received from the kernel.
///
/// `buf` is the raw bytes read from the provider port (header + payload).
pub fn handle_vfs_rpc(state: &mut NetVfsState, driver: &mut VirtioNetDriver, buf: &[u8]) {
    let hdr_size = core::mem::size_of::<VfsRpcReqHeader>();
    if buf.len() < hdr_size {
        return;
    }

    let resp_port = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as ChannelHandle;
    let op_byte = buf[4];
    let payload = &buf[hdr_size..];

    let op = match VfsRpcOp::from_u8(op_byte) {
        Some(o) => o,
        None => {
            send_err(resp_port, E_NOTSUP);
            return;
        }
    };

    info!(
        "VIRTIO_NETD: rpc op={:?} payload_len={} resp_port={}",
        op,
        payload.len(),
        resp_port
    );

    match op {
        VfsRpcOp::Lookup => handle_lookup(resp_port, payload),
        VfsRpcOp::Read => handle_read(state, resp_port, payload),
        VfsRpcOp::Write => handle_write(state, driver, resp_port, payload),
        VfsRpcOp::Readdir => handle_readdir(resp_port, payload),
        VfsRpcOp::Stat => handle_stat(resp_port, payload),
        VfsRpcOp::Close => send_resp(resp_port, &[E_OK]),
        VfsRpcOp::Poll => handle_poll(state, resp_port, payload),
        VfsRpcOp::DeviceCall => send_err(resp_port, E_NOTSUP),
        VfsRpcOp::Rename => send_err(resp_port, E_NOTSUP),
        VfsRpcOp::SubscribeReady => send_resp(resp_port, &[E_OK]),
        VfsRpcOp::UnsubscribeReady => send_resp(resp_port, &[E_OK]),
    }
}

// ── Lookup ────────────────────────────────────────────────────────────────────

fn handle_lookup(resp_port: ChannelHandle, payload: &[u8]) {
    if payload.len() < 4 {
        send_err(resp_port, E_INVAL);
        return;
    }
    let path_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if payload.len() < 4 + path_len {
        send_err(resp_port, E_INVAL);
        return;
    }
    let path = match core::str::from_utf8(&payload[4..4 + path_len]) {
        Ok(s) => s,
        Err(_) => {
            send_err(resp_port, E_INVAL);
            return;
        }
    };
    let path = path.trim_matches('/');
    info!("VIRTIO_NETD: lookup '{}'", path);

    let handle: u64 = match path {
        "" => HANDLE_ROOT,
        "ctl" => HANDLE_CTL,
        "status" => HANDLE_STATUS,
        "mac" => HANDLE_MAC,
        "mtu" => HANDLE_MTU,
        "rx" => HANDLE_RX,
        "tx" => HANDLE_TX,
        "features" => HANDLE_FEATURES,
        "events" => HANDLE_EVENTS,
        _ => {
            send_err(resp_port, E_NOENT);
            return;
        }
    };

    let mut resp = [0u8; 9]; // E_OK + u64
    resp[0] = E_OK;
    resp[1..9].copy_from_slice(&handle.to_le_bytes());
    send_resp(resp_port, &resp);
}

// ── Stat ──────────────────────────────────────────────────────────────────────

fn handle_stat(resp_port: ChannelHandle, payload: &[u8]) {
    if payload.len() < 8 {
        send_err(resp_port, E_INVAL);
        return;
    }
    let handle = u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ]);

    let (mode, size): (u32, u64) = match handle {
        HANDLE_ROOT => (S_IFDIR | 0o755, 0),
        HANDLE_CTL | HANDLE_TX => (S_IFREG | 0o200, 0), // write-only
        HANDLE_RX | HANDLE_EVENTS => (S_IFREG | 0o400, 0), // read-only stream
        HANDLE_STATUS | HANDLE_MAC | HANDLE_FEATURES => (S_IFREG | 0o444, 0),
        HANDLE_MTU => (S_IFREG | 0o644, 0),
        _ => {
            send_err(resp_port, E_NOENT);
            return;
        }
    };

    let mut resp = [0u8; 21]; // 1 + 4 + 8 + 8
    resp[0] = E_OK;
    resp[1..5].copy_from_slice(&mode.to_le_bytes());
    resp[5..13].copy_from_slice(&size.to_le_bytes());
    resp[13..21].copy_from_slice(&handle.to_le_bytes());
    send_resp(resp_port, &resp);
}

// ── Readdir ───────────────────────────────────────────────────────────────────

struct DirEntry {
    name: &'static str,
    handle: u64,
}

const DIR_ENTRIES: &[DirEntry] = &[
    DirEntry {
        name: "ctl",
        handle: HANDLE_CTL,
    },
    DirEntry {
        name: "status",
        handle: HANDLE_STATUS,
    },
    DirEntry {
        name: "mac",
        handle: HANDLE_MAC,
    },
    DirEntry {
        name: "mtu",
        handle: HANDLE_MTU,
    },
    DirEntry {
        name: "rx",
        handle: HANDLE_RX,
    },
    DirEntry {
        name: "tx",
        handle: HANDLE_TX,
    },
    DirEntry {
        name: "features",
        handle: HANDLE_FEATURES,
    },
    DirEntry {
        name: "events",
        handle: HANDLE_EVENTS,
    },
];

fn handle_readdir(resp_port: ChannelHandle, payload: &[u8]) {
    if payload.len() < 20 {
        send_err(resp_port, E_INVAL);
        return;
    }
    let handle = u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ]);
    let offset = u64::from_le_bytes([
        payload[8],
        payload[9],
        payload[10],
        payload[11],
        payload[12],
        payload[13],
        payload[14],
        payload[15],
    ]) as usize;
    let max_bytes =
        u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]) as usize;

    if handle != HANDLE_ROOT {
        send_err(resp_port, E_INVAL);
        return;
    }

    let mut out: Vec<u8> = Vec::new();
    for entry in DIR_ENTRIES.iter().skip(offset) {
        let name_bytes = entry.name.as_bytes();
        let name_len = name_bytes.len().min(255) as u8;
        // DT_REG = 8
        let file_type: u8 = 8;
        let entry_size = 10 + name_len as usize;
        if out.len() + entry_size > max_bytes {
            break;
        }
        // DirentWire: [ino: u64][file_type: u8][name_len: u8][name bytes]
        out.extend_from_slice(&entry.handle.to_le_bytes());
        out.push(file_type);
        out.push(name_len);
        out.extend_from_slice(&name_bytes[..name_len as usize]);
    }

    send_ok_data(resp_port, &out);
}

// ── Read ──────────────────────────────────────────────────────────────────────

fn handle_read(state: &mut NetVfsState, resp_port: ChannelHandle, payload: &[u8]) {
    if payload.len() < 20 {
        send_err(resp_port, E_INVAL);
        return;
    }
    let handle = u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ]);
    let offset = u64::from_le_bytes([
        payload[8],
        payload[9],
        payload[10],
        payload[11],
        payload[12],
        payload[13],
        payload[14],
        payload[15],
    ]) as usize;
    let len = u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]) as usize;

    match handle {
        HANDLE_STATUS => {
            info!("VIRTIO_NETD: read status");
            let text =
                alloc::format!(
                "state: {}\nlink: {}\nmac: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}\nmtu: {}\n",
                if state.link_up { "up" } else { "down" },
                if state.link_up { "up" } else { "down" },
                state.mac[0], state.mac[1], state.mac[2],
                state.mac[3], state.mac[4], state.mac[5],
                state.mtu,
            );
            send_text_slice(resp_port, text.as_bytes(), offset, len);
        }
        HANDLE_MAC => {
            info!("VIRTIO_NETD: read mac");
            let text = alloc::format!(
                "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}\n",
                state.mac[0], state.mac[1], state.mac[2], state.mac[3], state.mac[4], state.mac[5],
            );
            send_text_slice(resp_port, text.as_bytes(), offset, len);
        }
        HANDLE_MTU => {
            info!("VIRTIO_NETD: read mtu");
            let text = alloc::format!("{}\n", state.mtu);
            send_text_slice(resp_port, text.as_bytes(), offset, len);
        }
        HANDLE_FEATURES => {
            info!("VIRTIO_NETD: read features");
            let text = alloc::format!("0x{:08x}\n", state.features);
            send_text_slice(resp_port, text.as_bytes(), offset, len);
        }
        HANDLE_RX => {
            info!("VIRTIO_NETD: read rx queued={}", state.rx_queue.len());
            // Return one length-prefixed frame, or empty if none available.
            if let Some(frame) = state.rx_queue.pop_front() {
                let frame_len = frame.len() as u32;
                let mut out = Vec::with_capacity(4 + frame.len());
                out.extend_from_slice(&frame_len.to_le_bytes());
                out.extend_from_slice(&frame);
                send_ok_data(resp_port, &out);
            } else {
                send_ok_data(resp_port, &[]);
            }
        }
        HANDLE_EVENTS => {
            info!(
                "VIRTIO_NETD: read events queued={}",
                state.events_queue.len()
            );
            // Return one newline-terminated event, or empty if none queued.
            if let Some(event) = state.events_queue.pop_front() {
                send_ok_data(resp_port, &event);
            } else {
                send_ok_data(resp_port, &[]);
            }
        }
        HANDLE_CTL | HANDLE_TX => {
            send_err(resp_port, E_INVAL); // write-only
        }
        _ => {
            send_err(resp_port, E_NOENT);
        }
    }
}

/// Send a slice of `text` starting at `offset`, capped at `max_len` bytes.
fn send_text_slice(resp_port: ChannelHandle, text: &[u8], offset: usize, max_len: usize) {
    let start = offset.min(text.len());
    let slice = &text[start..];
    let out = &slice[..slice.len().min(max_len)];
    send_ok_data(resp_port, out);
}

// ── Write ─────────────────────────────────────────────────────────────────────

fn handle_write(
    state: &mut NetVfsState,
    driver: &mut VirtioNetDriver,
    resp_port: ChannelHandle,
    payload: &[u8],
) {
    if payload.len() < 20 {
        send_err(resp_port, E_INVAL);
        return;
    }
    let handle = u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ]);
    // bytes 8..16: offset (ignored for these files)
    let data_len =
        u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]) as usize;
    if payload.len() < 20 + data_len {
        send_err(resp_port, E_INVAL);
        return;
    }
    let data = &payload[20..20 + data_len];

    match handle {
        HANDLE_CTL => {
            info!("VIRTIO_NETD: write ctl len={}", data_len);
            let cmd = core::str::from_utf8(data).unwrap_or("").trim();
            if cmd == "up" {
                state.link_up = true;
                state.push_event("link-up");
            } else if cmd == "down" {
                state.link_up = false;
                state.push_event("link-down");
            } else if let Some(rest) = cmd.strip_prefix("set-mtu ") {
                if let Ok(mtu) = rest.trim().parse::<u32>() {
                    state.mtu = mtu;
                } else {
                    send_err(resp_port, E_INVAL);
                    return;
                }
            }
            send_ok_written(resp_port, data_len as u32);
        }
        HANDLE_MTU => {
            info!("VIRTIO_NETD: write mtu len={}", data_len);
            let text = core::str::from_utf8(data).unwrap_or("").trim();
            if let Ok(mtu) = text.parse::<u32>() {
                state.mtu = mtu;
                send_ok_written(resp_port, data_len as u32);
            } else {
                send_err(resp_port, E_INVAL);
            }
        }
        HANDLE_TX => {
            info!("VIRTIO_NETD: write tx len={}", data_len);
            // Expect length-prefixed frame: [4 bytes: len][len bytes: frame data]
            if data.len() < 4 {
                send_err(resp_port, E_INVAL);
                return;
            }
            let frame_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
            if data.len() < 4 + frame_len {
                send_err(resp_port, E_INVAL);
                return;
            }
            let frame = &data[4..4 + frame_len];
            match driver.tx(frame) {
                Ok(()) => send_ok_written(resp_port, data_len as u32),
                Err(_) => send_err(resp_port, E_IO),
            }
        }
        HANDLE_STATUS | HANDLE_MAC | HANDLE_FEATURES | HANDLE_RX | HANDLE_EVENTS => {
            send_err(resp_port, E_ROFS);
        }
        _ => {
            send_err(resp_port, E_NOENT);
        }
    }
}

// ── Poll ──────────────────────────────────────────────────────────────────────

fn handle_poll(state: &NetVfsState, resp_port: ChannelHandle, payload: &[u8]) {
    if payload.len() < 12 {
        send_err(resp_port, E_INVAL);
        return;
    }
    let handle = u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ]);
    // bytes 8..12: requested events (ignored — we check readiness unconditionally)

    let revents: u32 = match handle {
        HANDLE_RX => {
            if !state.rx_queue.is_empty() {
                POLLIN
            } else {
                0
            }
        }
        HANDLE_EVENTS => {
            if !state.events_queue.is_empty() {
                POLLIN
            } else {
                0
            }
        }
        // These text files are always readable.
        HANDLE_STATUS | HANDLE_MAC | HANDLE_MTU | HANDLE_FEATURES => POLLIN,
        // Write-only files are never readable.
        HANDLE_CTL | HANDLE_TX => 0,
        _ => 0,
    };

    let mut resp = [0u8; 5]; // E_OK + u32
    resp[0] = E_OK;
    resp[1..5].copy_from_slice(&revents.to_le_bytes());
    send_resp(resp_port, &resp);
}
