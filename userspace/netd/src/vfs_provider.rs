//! `/net/` VFS provider for netd (issue #541)
//!
//! Implements the full `/net/` filesystem tree as a userland VFS provider.
//! Applications interact with the network stack using ordinary file operations.
//!
//! ## Tree layout
//! ```text
//! /net/
//! ├── interfaces/
//! │   └── eth0/
//! │       ├── status    ← read:  multiline interface state
//! │       ├── addr      ← read/write: CIDR "192.168.1.50/24\n"
//! │       ├── flags     ← write: "up" / "down"
//! │       ├── mtu       ← read/write: decimal MTU
//! │       ├── stats     ← read:  rx/tx counts
//! │       └── events    ← pollable link-state stream
//! ├── routes            ← read:  table; write: "add default via 1.2.3.4 dev eth0"
//! ├── tcp/
//! │   ├── new           ← read:  allocates socket, returns id
//! │   └── <id>/
//! │       ├── ctl       ← write: "connect IP PORT" / "listen PORT [BACKLOG]" / "close"
//! │       ├── data      ← read/write: TCP byte stream
//! │       ├── accept    ← read:  "<conn_id> <ip> <port>\n" (listener sockets only)
//! │       ├── status    ← read:  state text
//! │       └── events    ← pollable connection events
//! ├── udp/
//! │   ├── new           ← read:  allocates socket, returns id
//! │   └── <id>/
//! │       ├── ctl       ← write: "bind PORT" / "connect IP PORT" / "broadcast 0|1" / "close"
//! │       ├── data      ← write: [4: dest_ipv4][2: dest_port_le][4: len_le][payload]
//! │       │               read:  [4: src_ipv4][2: src_port_le][4: len_le][payload]
//! │       └── status    ← read:  state text, including `broadcast: true|false`
//! └── dns/
//!     └── lookup        ← write: hostname; read: dotted-decimal IPv4 address
//! ```
extern crate alloc;
use alloc::string::ToString;

use abi::vfs_rpc::{VfsRpcOp, VfsRpcReqHeader, VFS_RPC_MAX_REQ};
use alloc::string::String;
use alloc::{vec, vec::Vec};
use smoltcp::iface::{Interface, SocketSet};
use smoltcp::socket::tcp::{Socket as TcpSocket, SocketBuffer};
use smoltcp::wire::{IpAddress, IpCidr, Ipv4Address};
use stem::syscall::channel::{channel_create, channel_send, channel_try_recv, ChannelHandle};
use stem::syscall::vfs::vfs_mount;
use stem::{info, warn};

use crate::socket_api::{SocketApi, CONN_RX, CONN_TX};

// ── errno shorthands ─────────────────────────────────────────────────────────

const E_OK: u8 = 0;
const E_NOENT: u8 = 2;
const E_IO: u8 = 5;
const E_INVAL: u8 = 22;
const E_ROFS: u8 = 30;
const E_NOTSUP: u8 = 38;

// ── file-type bits ────────────────────────────────────────────────────────────

const S_IFDIR: u32 = 0o040_000;
const S_IFREG: u32 = 0o100_000;

// ── poll interest bits ───────────────────────────────────────────────────────

const POLLIN: u32 = 0x0001;
#[allow(dead_code)]
const POLLOUT: u32 = 0x0004;

// ── static handle constants ──────────────────────────────────────────────────

const HANDLE_ROOT: u64 = 1;
const HANDLE_INTERFACES_DIR: u64 = 2;
const HANDLE_ETH0_DIR: u64 = 3;
const HANDLE_ETH0_STATUS: u64 = 4;
const HANDLE_ETH0_ADDR: u64 = 5;
const HANDLE_ETH0_FLAGS: u64 = 6;
const HANDLE_ETH0_MTU: u64 = 7;
const HANDLE_ETH0_STATS: u64 = 8;
const HANDLE_ETH0_EVENTS: u64 = 9;
const HANDLE_ROUTES: u64 = 10;
const HANDLE_TCP_DIR: u64 = 11;
const HANDLE_TCP_NEW: u64 = 12;
const HANDLE_UDP_DIR: u64 = 13;
const HANDLE_UDP_NEW: u64 = 14;
const HANDLE_DNS_DIR: u64 = 15;
const HANDLE_DNS_LOOKUP: u64 = 16;
const HANDLE_DNS_SERVER: u64 = 17;

/// Dynamic handle base for TCP socket sub-files.
/// Handle = TCP_DYN_BASE | ((api_handle as u64) << 8) | subfile_id
const TCP_DYN_BASE: u64 = 0x0001_0000;
const UDP_DYN_BASE: u64 = 0x0100_0000;

// subfile IDs
const SF_DIR: u8 = 0;
const SF_CTL: u8 = 1;
const SF_DATA: u8 = 2;
const SF_STATUS: u8 = 3;
const SF_EVENTS: u8 = 4;
/// Listener-only: returns accepted connection id on read.
const SF_ACCEPT: u8 = 5;

// ── IP config ────────────────────────────────────────────────────────────────

/// Snapshot of the current IPv4 configuration published by DHCP or static
/// assignment.  Stored inside `NetVfsProvider` and updated when the interface
/// configuration changes.
#[derive(Clone, Copy)]
pub struct IpConfig {
    pub ip: Ipv4Address,
    pub prefix_len: u8,
    pub gateway: Ipv4Address,
    pub dns_server: Ipv4Address,
}

// ── main provider struct ──────────────────────────────────────────────────────

/// Userland VFS provider that serves the `/net/` namespace.
pub struct NetVfsProvider {
    /// Read-end of the VFS RPC port (provider reads requests from here).
    req_read: ChannelHandle,
    /// MAC address of the first interface (eth0).
    pub mac: [u8; 6],
    /// MTU of eth0.
    pub mtu: usize,
    /// Current link state.
    pub link_up: bool,
    /// Current IP configuration (populated after DHCP).
    pub ip_config: Option<IpConfig>,
    /// Monoton rx/tx byte counters for stats.
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    pub rx_packets: u64,
    pub tx_packets: u64,
    /// RPC request staging buffer.
    req_buf: Vec<u8>,
    /// Pending DNS hostname to resolve (written by op_write to HANDLE_DNS_LOOKUP).
    dns_pending: Option<alloc::string::String>,
    /// Resolved DNS result (dotted-decimal IPv4 or error text).
    dns_result: Option<alloc::string::String>,
}

impl NetVfsProvider {
    /// Create and mount the `/net/` provider.
    ///
    /// Returns `None` if the port creation or mount fails.
    pub fn new(mac: [u8; 6], mtu: usize, link_up: bool) -> Option<Self> {
        let (req_write, req_read) = match channel_create(VFS_RPC_MAX_REQ * 8) {
            Ok(p) => p,
            Err(e) => {
                warn!("NetVfsProvider: failed to create RPC port: {:?}", e);
                return None;
            }
        };

        match vfs_mount(req_write, "/net") {
            Ok(()) => {
                info!(
                    "NetVfsProvider: mounted at /net (port w={} r={})",
                    req_write, req_read
                );
            }
            Err(e) => {
                warn!("NetVfsProvider: vfs_mount failed: {:?}", e);
                return None;
            }
        }

        Some(Self {
            req_read,
            mac,
            mtu,
            link_up,
            ip_config: None,
            rx_bytes: 0,
            tx_bytes: 0,
            rx_packets: 0,
            tx_packets: 0,
            req_buf: alloc::vec![0u8; VFS_RPC_MAX_REQ],
            dns_pending: None,
            dns_result: None,
        })
    }

    /// The port handle the RPC loop reads from (pass to `port_wait` / `port_len`).
    pub fn req_read_port(&self) -> ChannelHandle {
        self.req_read
    }

    /// Update the IP configuration after DHCP completes.
    pub fn set_ip_config(
        &mut self,
        ip: Ipv4Address,
        prefix_len: u8,
        gateway: Ipv4Address,
        dns_server: Ipv4Address,
    ) {
        self.ip_config = Some(IpConfig {
            ip,
            prefix_len,
            gateway,
            dns_server,
        });
    }

    /// Process all pending VFS RPC messages (non-blocking drain).
    ///
    /// `socket_api` and `socket_set` are borrowed so that socket operations
    /// (open/connect/send/recv) can be dispatched inline without locking.
    ///
    /// Also performs any pending DNS resolution synchronously when a hostname
    /// has been written to `/net/dns/lookup`.
    pub fn drain_rpcs<D: smoltcp::phy::Device>(
        &mut self,
        iface: &mut Interface,
        device: &mut D,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) {
        loop {
            match channel_try_recv(self.req_read, &mut self.req_buf) {
                Ok(n) if n > 0 => {
                    let buf: &[u8] = unsafe {
                        // Extend lifetime: we're about to pass it to handle_one which
                        // completes before the next iteration touches req_buf.
                        core::slice::from_raw_parts(self.req_buf.as_ptr(), n)
                    };
                    self.handle_one(iface, device, socket_set, socket_api, buf);
                }
                _ => break,
            }
        }

        // If a DNS lookup was requested, resolve it now (synchronously in netd).
        if self.dns_result.is_none() {
            if let Some(hostname) = self.dns_pending.take() {
                if let Some(dns_ip) = self.ip_config.as_ref().map(|c| c.gateway) {
                    // Use gateway as DNS server if not configured separately.
                    match crate::dns::lookup_a(iface, device, dns_ip, &hostname) {
                        Ok(ip) => {
                            let b = ip.as_bytes();
                            self.dns_result =
                                Some(alloc::format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3]));
                        }
                        Err(e) => {
                            warn!("DNS lookup for '{}' failed: {:?}", hostname, e);
                            self.dns_result = Some("error".into());
                        }
                    }
                } else {
                    self.dns_result = Some("error".into());
                }
            }
        }
    }

    // ── RPC dispatch ─────────────────────────────────────────────────────────

    fn handle_one<D: smoltcp::phy::Device>(
        &mut self,
        iface: &mut Interface,
        device: &mut D,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
        buf: &[u8],
    ) {
        let hdr_sz = core::mem::size_of::<VfsRpcReqHeader>();
        if buf.len() < hdr_sz {
            return;
        }

        let resp_port = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as ChannelHandle;
        let op_byte = buf[4];
        let payload = &buf[hdr_sz..];

        let op = match VfsRpcOp::from_u8(op_byte) {
            Some(o) => o,
            None => {
                send_err(resp_port, E_NOTSUP);
                return;
            }
        };

        match op {
            VfsRpcOp::Lookup => self.op_lookup(resp_port, payload),
            VfsRpcOp::Read => self.op_read(resp_port, payload, socket_set, socket_api),
            VfsRpcOp::Write => {
                self.op_write(resp_port, payload, iface, device, socket_set, socket_api)
            }
            VfsRpcOp::Readdir => self.op_readdir(resp_port, payload, socket_api),
            VfsRpcOp::Stat => self.op_stat(resp_port, payload, socket_api),
            VfsRpcOp::Close => self.op_close(resp_port, payload, socket_set, socket_api),
            VfsRpcOp::Poll => self.op_poll(resp_port, payload, socket_set, socket_api),
            VfsRpcOp::DeviceCall => send_err(resp_port, E_NOTSUP),
            VfsRpcOp::Rename => send_err(resp_port, E_NOTSUP),
            VfsRpcOp::SubscribeReady => send_resp(resp_port, &[E_OK]),
            VfsRpcOp::UnsubscribeReady => send_resp(resp_port, &[E_OK]),
        }
    }

    // ── Lookup ────────────────────────────────────────────────────────────────

    fn op_lookup(&self, resp_port: ChannelHandle, payload: &[u8]) {
        if payload.len() < 4 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let path_len =
            u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        if payload.len() < 4 + path_len {
            send_err(resp_port, E_INVAL);
            return;
        }
        let path = match core::str::from_utf8(&payload[4..4 + path_len]) {
            Ok(s) => s.trim_matches('/'),
            Err(_) => {
                send_err(resp_port, E_INVAL);
                return;
            }
        };

        match self.resolve_path(path) {
            Some(handle) => send_handle(resp_port, handle),
            None => send_err(resp_port, E_NOENT),
        }
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    fn op_read(
        &mut self,
        resp_port: ChannelHandle,
        payload: &[u8],
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) {
        if payload.len() < 20 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let offset = u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let len = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;

        let data = self.read_handle(handle, offset, len, socket_set, socket_api);
        match data {
            ReadResult::Data(bytes) => send_data(resp_port, &bytes),
            ReadResult::EOF => send_data(resp_port, &[]),
            ReadResult::Again => send_err(resp_port, 11), // EAGAIN
            ReadResult::Error => send_err(resp_port, E_IO),
            ReadResult::NotSupported => send_err(resp_port, E_NOTSUP),
        }
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    fn op_write<D: smoltcp::phy::Device>(
        &mut self,
        resp_port: ChannelHandle,
        payload: &[u8],
        iface: &mut Interface,
        device: &mut D,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) {
        if payload.len() < 20 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let _offset = u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let data_len = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;
        if payload.len() < 20 + data_len {
            send_err(resp_port, E_INVAL);
            return;
        }
        let data = &payload[20..20 + data_len];

        let result = self.write_handle(handle, data, iface, device, socket_set, socket_api);
        match result {
            WriteResult::Ok(n) => send_write_ok(resp_port, n as u32),
            WriteResult::Error => send_err(resp_port, E_IO),
            WriteResult::ReadOnly => send_err(resp_port, E_ROFS),
            WriteResult::NotSupported => send_err(resp_port, E_NOTSUP),
        }
    }

    // ── Readdir ───────────────────────────────────────────────────────────────

    fn op_readdir(&self, resp_port: ChannelHandle, payload: &[u8], socket_api: &SocketApi) {
        if payload.len() < 20 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let offset = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
        let max_bytes = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;

        let entries = self.list_dir(handle, socket_api);
        let entries_slice = entries.as_slice();

        let start = offset.min(entries_slice.len());
        let tail = &entries_slice[start..];

        let mut out: Vec<u8> = Vec::new();
        for (name, ino, file_type) in tail {
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len().min(255) as u8;
            let entry_size = 10 + name_len as usize;
            if out.len() + entry_size > max_bytes {
                break;
            }
            out.extend_from_slice(&ino.to_le_bytes());
            out.push(*file_type); // DT_DIR=4 or DT_REG=8
            out.push(name_len);
            out.extend_from_slice(&name_bytes[..name_len as usize]);
        }

        send_data(resp_port, &out);
    }

    // ── Stat ─────────────────────────────────────────────────────────────────

    fn op_stat(&self, resp_port: ChannelHandle, payload: &[u8], socket_api: &SocketApi) {
        if payload.len() < 8 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let (mode, size) = self.stat_handle(handle, socket_api);
        if mode == 0 {
            send_err(resp_port, E_NOENT);
            return;
        }

        let mut resp = [0u8; 21]; // 1 + 4 + 8 + 8
        resp[0] = E_OK;
        resp[1..5].copy_from_slice(&mode.to_le_bytes());
        resp[5..13].copy_from_slice(&(size as u64).to_le_bytes());
        resp[13..21].copy_from_slice(&handle.to_le_bytes());
        let _ = channel_send(resp_port, &resp);
    }

    // ── Close ─────────────────────────────────────────────────────────────────

    fn op_close(
        &self,
        resp_port: ChannelHandle,
        payload: &[u8],
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) {
        if payload.len() < 8 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        // If it's a TCP or UDP socket *directory* handle, close the underlying socket.
        if handle >= TCP_DYN_BASE && handle < UDP_DYN_BASE {
            let sf = (handle & 0xFF) as u8;
            let api_handle = ((handle - TCP_DYN_BASE) >> 8) as u32;
            if sf == SF_DIR {
                let _ = socket_api.handle_close(socket_set, api_handle);
            }
        } else if handle >= UDP_DYN_BASE {
            let sf = (handle & 0xFF) as u8;
            let api_handle = ((handle - UDP_DYN_BASE) >> 8) as u32;
            if sf == SF_DIR {
                let _ = socket_api.handle_close(socket_set, api_handle);
            }
        }

        send_resp(resp_port, &[E_OK]);
    }

    // ── Poll ─────────────────────────────────────────────────────────────────

    fn op_poll(
        &self,
        resp_port: ChannelHandle,
        payload: &[u8],
        socket_set: &mut SocketSet,
        socket_api: &SocketApi,
    ) {
        if payload.len() < 12 {
            send_err(resp_port, E_INVAL);
            return;
        }
        let handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let _events = u32::from_le_bytes(payload[8..12].try_into().unwrap());

        let revents = self.poll_handle(handle, socket_set, socket_api);

        let mut resp = [0u8; 5];
        resp[0] = E_OK;
        resp[1..5].copy_from_slice(&revents.to_le_bytes());
        let _ = channel_send(resp_port, &resp);
    }

    // ── Path resolution ───────────────────────────────────────────────────────

    /// Resolve a path (relative to mount point, leading/trailing slashes stripped)
    /// to a u64 handle.  Returns `None` for unknown paths.
    fn resolve_path(&self, path: &str) -> Option<u64> {
        match path {
            "" | "/" => Some(HANDLE_ROOT),
            "interfaces" => Some(HANDLE_INTERFACES_DIR),
            "interfaces/eth0" => Some(HANDLE_ETH0_DIR),
            "interfaces/eth0/status" => Some(HANDLE_ETH0_STATUS),
            "interfaces/eth0/addr" => Some(HANDLE_ETH0_ADDR),
            "interfaces/eth0/flags" => Some(HANDLE_ETH0_FLAGS),
            "interfaces/eth0/mtu" => Some(HANDLE_ETH0_MTU),
            "interfaces/eth0/stats" => Some(HANDLE_ETH0_STATS),
            "interfaces/eth0/events" => Some(HANDLE_ETH0_EVENTS),
            "routes" => Some(HANDLE_ROUTES),
            "tcp" => Some(HANDLE_TCP_DIR),
            "tcp/new" => Some(HANDLE_TCP_NEW),
            "udp" => Some(HANDLE_UDP_DIR),
            "udp/new" => Some(HANDLE_UDP_NEW),
            "dns" => Some(HANDLE_DNS_DIR),
            "dns/lookup" => Some(HANDLE_DNS_LOOKUP),
            "dns/server" => Some(HANDLE_DNS_SERVER),
            other => self.resolve_dynamic_path(other),
        }
    }

    fn resolve_dynamic_path(&self, path: &str) -> Option<u64> {
        // tcp/<id>[/<subfile>]
        if let Some(rest) = path.strip_prefix("tcp/") {
            let (id_str, sub) = match rest.find('/') {
                Some(pos) => (&rest[..pos], &rest[pos + 1..]),
                None => (rest, ""),
            };
            let id: u32 = id_str.parse().ok()?;
            let sf: u8 = match sub {
                "" => SF_DIR,
                "ctl" => SF_CTL,
                "data" => SF_DATA,
                "status" => SF_STATUS,
                "events" => SF_EVENTS,
                "accept" => SF_ACCEPT,
                _ => return None,
            };
            return Some(TCP_DYN_BASE | ((id as u64) << 8) | sf as u64);
        }

        // udp/<id>[/<subfile>]
        if let Some(rest) = path.strip_prefix("udp/") {
            let (id_str, sub) = match rest.find('/') {
                Some(pos) => (&rest[..pos], &rest[pos + 1..]),
                None => (rest, ""),
            };
            let id: u32 = id_str.parse().ok()?;
            let sf: u8 = match sub {
                "" => SF_DIR,
                "ctl" => SF_CTL,
                "data" => SF_DATA,
                "status" => SF_STATUS,
                _ => return None,
            };
            return Some(UDP_DYN_BASE | ((id as u64) << 8) | sf as u64);
        }

        None
    }

    // ── Handle reads ─────────────────────────────────────────────────────────

    fn read_handle(
        &mut self,
        handle: u64,
        offset: u64,
        _len: usize,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) -> ReadResult {
        match handle {
            HANDLE_ETH0_STATUS => ReadResult::text_offset(&self.eth0_status(), offset),
            HANDLE_ETH0_ADDR => ReadResult::text_offset(&self.eth0_addr_text(), offset),
            HANDLE_ETH0_MTU => ReadResult::text_offset(&alloc::format!("{}\n", self.mtu), offset),
            HANDLE_ETH0_STATS => ReadResult::text_offset(&self.eth0_stats(), offset),
            HANDLE_ETH0_EVENTS => {
                // Events are single-shot; subsequent reads return EOF until next event.
                if offset == 0 {
                    let s = if self.link_up {
                        "link-up\n"
                    } else {
                        "link-down\n"
                    };
                    ReadResult::Data(s.as_bytes().to_vec())
                } else {
                    ReadResult::EOF
                }
            }
            HANDLE_ROUTES => ReadResult::text_offset(&self.routes_text(socket_api), offset),
            HANDLE_DNS_SERVER => ReadResult::text_offset(&self.dns_server_text(), offset),
            // tcp/new: allocate a new TCP socket, return its id as text
            HANDLE_TCP_NEW => {
                if offset > 0 {
                    return ReadResult::EOF;
                }
                let Some(buf_idx) = socket_api.alloc_buffer() else {
                    warn!("NetVfsProvider: out of socket buffers for tcp/new");
                    return ReadResult::Error;
                };
                let rx_buf = SocketBuffer::new(unsafe { &mut CONN_RX[buf_idx][..] });
                let tx_buf = SocketBuffer::new(unsafe { &mut CONN_TX[buf_idx][..] });
                let socket = TcpSocket::new(rx_buf, tx_buf);
                let shdl = socket_set.add(socket);
                let api_handle = socket_api.alloc_socket_raw(shdl, buf_idx, false, 0);
                let text = alloc::format!("{}\n", api_handle);
                ReadResult::Data(text.into_bytes())
            }
            // udp/new: allocate a new UDP socket
            HANDLE_UDP_NEW => {
                if offset > 0 {
                    return ReadResult::EOF;
                }
                let Some(buf_idx) = socket_api.alloc_buffer() else {
                    warn!("NetVfsProvider: out of socket buffers for udp/new");
                    return ReadResult::Error;
                };
                let api_handle = socket_api.alloc_udp_socket_raw(socket_set, buf_idx);
                match api_handle {
                    Some(id) => {
                        let text = alloc::format!("{}\n", id);
                        ReadResult::Data(text.into_bytes())
                    }
                    None => ReadResult::Error,
                }
            }
            // Dynamic TCP data
            h if h >= TCP_DYN_BASE && h < UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - TCP_DYN_BASE) >> 8) as u32;
                self.read_tcp(api_handle, sf, offset, socket_set, socket_api)
            }
            // Dynamic UDP data
            h if h >= UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - UDP_DYN_BASE) >> 8) as u32;
                self.read_udp(api_handle, sf, offset, socket_set, socket_api)
            }
            // Directories are not readable as byte streams
            HANDLE_ROOT
            | HANDLE_INTERFACES_DIR
            | HANDLE_ETH0_DIR
            | HANDLE_TCP_DIR
            | HANDLE_UDP_DIR
            | HANDLE_DNS_DIR => ReadResult::NotSupported,
            // dns/lookup: returns the resolved IP (EAGAIN if not yet resolved)
            HANDLE_DNS_LOOKUP => {
                if offset > 0 {
                    return ReadResult::EOF;
                }
                match &self.dns_result {
                    Some(result) if result != "error" => {
                        let text = result.clone() + "\n";
                        self.dns_result = None; // consume result
                        self.dns_pending = None;
                        ReadResult::Data(text.into_bytes())
                    }
                    Some(_) => {
                        self.dns_result = None;
                        self.dns_pending = None;
                        ReadResult::Error
                    }
                    None => ReadResult::Again, // EAGAIN: resolution in progress
                }
            }
            _ => ReadResult::Error,
        }
    }

    fn read_tcp(
        &mut self,
        api_handle: u32,
        sf: u8,
        offset: u64,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) -> ReadResult {
        match sf {
            SF_DIR => ReadResult::NotSupported,
            SF_STATUS => {
                ReadResult::text_offset(&socket_api.tcp_status_text(api_handle, socket_set), offset)
            }
            SF_DATA => {
                let recv = socket_api.handle_recv(socket_set, api_handle, 8192);
                // handle_recv returns [resp_type: u16][data...]
                if recv.len() < 2 {
                    return ReadResult::Error;
                }
                let resp_type = u16::from_le_bytes([recv[0], recv[1]]);
                match resp_type {
                    crate::socket_api::RESP_DATA => {
                        self.rx_bytes += (recv.len() - 2) as u64;
                        self.rx_packets += 1;
                        ReadResult::Data(recv[2..].to_vec())
                    }
                    crate::socket_api::RESP_EMPTY => ReadResult::Again,
                    crate::socket_api::RESP_CLOSED => ReadResult::EOF,
                    _ => ReadResult::Error,
                }
            }
            SF_EVENTS => {
                ReadResult::text_offset(&socket_api.tcp_events_text(api_handle, socket_set), offset)
            }
            SF_ACCEPT => {
                // Listener sockets: returns "<conn_id> <ip> <port>\n" when a
                // connection is ready, or EAGAIN when none is queued.
                let Some(buf_idx) = socket_api.alloc_buffer() else {
                    return ReadResult::Error;
                };
                let result = socket_api.handle_accept(socket_set, api_handle, 0, buf_idx);
                if result.len() < 2 {
                    socket_api.free_buffer(buf_idx);
                    return ReadResult::Error;
                }
                let resp_type = u16::from_le_bytes([result[0], result[1]]);
                match resp_type {
                    crate::socket_api::RESP_ACCEPT => {
                        // [2: RESP_ACCEPT][4: conn_handle][4: remote_ip][2: remote_port]
                        if result.len() < 12 {
                            return ReadResult::Error;
                        }
                        let conn_handle = u32::from_le_bytes(result[2..6].try_into().unwrap());
                        let ip = &result[6..10];
                        let port = u16::from_le_bytes(result[10..12].try_into().unwrap());
                        let text = alloc::format!(
                            "{} {}.{}.{}.{} {}\n",
                            conn_handle,
                            ip[0],
                            ip[1],
                            ip[2],
                            ip[3],
                            port
                        );
                        ReadResult::Data(text.into_bytes())
                    }
                    crate::socket_api::RESP_EMPTY => ReadResult::Again,
                    _ => ReadResult::Error,
                }
            }
            _ => ReadResult::Error,
        }
    }

    fn read_udp(
        &mut self,
        api_handle: u32,
        sf: u8,
        offset: u64,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) -> ReadResult {
        match sf {
            SF_DIR => ReadResult::NotSupported,
            SF_STATUS => ReadResult::text_offset(&socket_api.udp_status_text(api_handle), offset),
            SF_DATA => {
                let recv = socket_api.handle_udp_recv_from(socket_set, api_handle);
                if recv.len() < 2 {
                    return ReadResult::Error;
                }
                let resp_type = u16::from_le_bytes([recv[0], recv[1]]);
                match resp_type {
                    crate::socket_api::RESP_DATA => {
                        // encode_udp_data format: [2: RESP_DATA][4: src_ip][2: src_port][payload]
                        // New wire format: [4: src_ip][2: src_port][4: payload_len][payload]
                        if recv.len() < 8 {
                            return ReadResult::Error;
                        }
                        let src_ip = &recv[2..6];
                        let src_port = u16::from_le_bytes([recv[6], recv[7]]);
                        let payload = &recv[8..];
                        self.rx_bytes += payload.len() as u64;
                        self.rx_packets += 1;
                        let mut out = alloc::vec![0u8; 4 + 2 + 4 + payload.len()];
                        out[..4].copy_from_slice(src_ip);
                        out[4..6].copy_from_slice(&src_port.to_le_bytes());
                        out[6..10].copy_from_slice(&(payload.len() as u32).to_le_bytes());
                        out[10..].copy_from_slice(payload);
                        ReadResult::Data(out)
                    }
                    crate::socket_api::RESP_EMPTY => ReadResult::Again,
                    _ => ReadResult::Error,
                }
            }
            _ => ReadResult::Error,
        }
    }

    // ── Handle writes ─────────────────────────────────────────────────────────

    fn write_handle<D: smoltcp::phy::Device>(
        &mut self,
        handle: u64,
        data: &[u8],
        iface: &mut Interface,
        device: &mut D,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) -> WriteResult {
        let text = match core::str::from_utf8(data) {
            Ok(s) => s.trim(),
            Err(_) => "",
        };

        match handle {
            HANDLE_ETH0_ADDR => self.write_eth0_addr(text, iface),
            HANDLE_ETH0_FLAGS => self.write_eth0_flags(text),
            HANDLE_ETH0_MTU => {
                if let Ok(n) = text.parse::<usize>() {
                    self.mtu = n;
                    WriteResult::Ok(data.len())
                } else {
                    WriteResult::Error
                }
            }
            HANDLE_ROUTES => self.write_routes(text, iface),
            // Read-only files
            HANDLE_ETH0_STATUS | HANDLE_ETH0_STATS | HANDLE_ETH0_EVENTS => WriteResult::ReadOnly,
            // dns/lookup: write hostname, clear any previous result
            HANDLE_DNS_LOOKUP => {
                let hostname = text.trim().to_string();
                if hostname.is_empty() {
                    return WriteResult::Error;
                }
                self.dns_pending = Some(hostname);
                self.dns_result = None;
                WriteResult::Ok(data.len())
            }
            // Dynamic TCP
            h if h >= TCP_DYN_BASE && h < UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - TCP_DYN_BASE) >> 8) as u32;
                self.write_tcp(
                    api_handle, sf, data, text, iface, device, socket_set, socket_api,
                )
            }
            // Dynamic UDP
            h if h >= UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - UDP_DYN_BASE) >> 8) as u32;
                self.write_udp(api_handle, sf, data, text, socket_set, socket_api)
            }
            _ => WriteResult::NotSupported,
        }
    }

    fn write_eth0_addr(&mut self, text: &str, iface: &mut Interface) -> WriteResult {
        // "192.168.1.50/24" or "192.168.1.50"
        let (ip_str, prefix_str) = match text.find('/') {
            Some(pos) => (&text[..pos], &text[pos + 1..]),
            None => (text, "24"),
        };
        let prefix_len: u8 = prefix_str.parse().unwrap_or(24);
        let ip = match parse_ipv4(ip_str) {
            Some(ip) => ip,
            None => return WriteResult::Error,
        };
        iface.update_ip_addrs(|addrs| {
            if let Some(slot) = addrs.iter_mut().next() {
                *slot = IpCidr::new(IpAddress::Ipv4(ip), prefix_len);
            }
        });
        if let Some(cfg) = &mut self.ip_config {
            cfg.ip = ip;
            cfg.prefix_len = prefix_len;
        } else {
            self.ip_config = Some(IpConfig {
                ip,
                prefix_len,
                gateway: Ipv4Address::new(0, 0, 0, 0),
                dns_server: Ipv4Address::new(0, 0, 0, 0),
            });
        }
        WriteResult::Ok(text.len())
    }

    fn write_eth0_flags(&mut self, text: &str) -> WriteResult {
        match text {
            "up" | "1" => {
                self.link_up = true;
                WriteResult::Ok(2)
            }
            "down" | "0" => {
                self.link_up = false;
                WriteResult::Ok(4)
            }
            _ => WriteResult::Error,
        }
    }

    fn write_routes(&self, text: &str, iface: &mut Interface) -> WriteResult {
        // "add default via 1.2.3.4 dev eth0"
        if let Some(rest) = text.strip_prefix("add default via ") {
            let gw_str = rest.split_whitespace().next().unwrap_or("");
            if let Some(gw) = parse_ipv4(gw_str) {
                iface.routes_mut().add_default_ipv4_route(gw).ok();
                return WriteResult::Ok(text.len());
            }
        }
        WriteResult::Error
    }

    fn write_tcp<D: smoltcp::phy::Device>(
        &mut self,
        api_handle: u32,
        sf: u8,
        raw: &[u8],
        text: &str,
        iface: &mut Interface,
        _device: &mut D,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) -> WriteResult {
        match sf {
            SF_CTL => {
                // "connect IP PORT", "listen PORT [BACKLOG]", or "close"
                if let Some(rest) = text.strip_prefix("connect ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let (Some(ip), Ok(port)) =
                            (parse_ipv4(parts[0]), parts[1].parse::<u16>())
                        {
                            let r = socket_api
                                .handle_connect_existing(iface, socket_set, api_handle, ip, port);
                            return if r {
                                WriteResult::Ok(text.len())
                            } else {
                                WriteResult::Error
                            };
                        }
                    }
                } else if let Some(rest) = text.strip_prefix("listen ") {
                    // "listen PORT [BACKLOG]"
                    let mut parts = rest.split_whitespace();
                    if let Some(port_str) = parts.next() {
                        if let Ok(port) = port_str.parse::<u16>() {
                            let backlog: u16 =
                                parts.next().and_then(|s| s.parse().ok()).unwrap_or(4);
                            let r = socket_api
                                .handle_listen_existing(socket_set, api_handle, port, backlog);
                            return if r {
                                WriteResult::Ok(text.len())
                            } else {
                                WriteResult::Error
                            };
                        }
                    }
                } else if text == "close" {
                    socket_api.handle_close(socket_set, api_handle);
                    return WriteResult::Ok(5);
                }
                WriteResult::Error
            }
            SF_DATA => {
                let result = socket_api.handle_send(socket_set, api_handle, raw);
                if result.len() >= 4 {
                    let sent = u16::from_le_bytes([result[2], result[3]]) as usize;
                    self.tx_bytes += sent as u64;
                    self.tx_packets += 1;
                    WriteResult::Ok(sent)
                } else {
                    WriteResult::Error
                }
            }
            _ => WriteResult::ReadOnly,
        }
    }

    fn write_udp(
        &mut self,
        api_handle: u32,
        sf: u8,
        raw: &[u8],
        text: &str,
        socket_set: &mut SocketSet,
        socket_api: &mut SocketApi,
    ) -> WriteResult {
        match sf {
            SF_CTL => {
                // "bind PORT", "connect IP PORT", "broadcast 0|1", or "close"
                if let Some(rest) = text.strip_prefix("bind ") {
                    if let Ok(port) = rest.trim().parse::<u16>() {
                        let r = socket_api.handle_udp_bind_port(socket_set, api_handle, port);
                        return if r {
                            WriteResult::Ok(text.len())
                        } else {
                            WriteResult::Error
                        };
                    }
                } else if let Some(rest) = text.strip_prefix("connect ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let (Some(ip), Ok(port)) =
                            (parse_ipv4(parts[0]), parts[1].parse::<u16>())
                        {
                            let r = socket_api.handle_udp_connect(socket_set, api_handle, ip, port);
                            return if r {
                                WriteResult::Ok(text.len())
                            } else {
                                WriteResult::Error
                            };
                        }
                    }
                } else if let Some(rest) = text.strip_prefix("broadcast ") {
                    let enabled = match rest.trim() {
                        "1" | "true" => true,
                        "0" | "false" => false,
                        _ => return WriteResult::Error,
                    };
                    let r = socket_api.handle_udp_set_broadcast(api_handle, enabled);
                    return if r {
                        WriteResult::Ok(text.len())
                    } else {
                        WriteResult::Error
                    };
                } else if text == "close" {
                    socket_api.handle_close(socket_set, api_handle);
                    return WriteResult::Ok(5);
                }
                WriteResult::Error
            }
            SF_DATA => {
                // New wire format: [4: dest_ip][2: dest_port_le][4: payload_len_le][payload]
                if raw.len() < 10 {
                    return WriteResult::Error;
                }
                let dest_ip = Ipv4Address::from_bytes(&raw[..4]);
                let dest_port = u16::from_le_bytes([raw[4], raw[5]]);
                let payload_len = u32::from_le_bytes([raw[6], raw[7], raw[8], raw[9]]) as usize;
                if raw.len() < 10 + payload_len {
                    return WriteResult::Error;
                }
                let payload = &raw[10..10 + payload_len];
                let r = socket_api
                    .handle_udp_send_to(socket_set, api_handle, dest_ip, dest_port, payload);
                if r.len() >= 4 {
                    let sent = u16::from_le_bytes([r[2], r[3]]) as usize;
                    self.tx_bytes += sent as u64;
                    self.tx_packets += 1;
                    WriteResult::Ok(10 + sent)
                } else {
                    WriteResult::Error
                }
            }
            _ => WriteResult::ReadOnly,
        }
    }

    // ── Readdir / stat helpers ────────────────────────────────────────────────

    /// List directory entries for a given directory handle.
    /// Returns `(name, ino, file_type)` triples; file_type 4=DT_DIR, 8=DT_REG.
    fn list_dir(&self, handle: u64, socket_api: &SocketApi) -> Vec<(String, u64, u8)> {
        match handle {
            HANDLE_ROOT => vec![
                ("interfaces".into(), HANDLE_INTERFACES_DIR, 4),
                ("routes".into(), HANDLE_ROUTES, 8),
                ("tcp".into(), HANDLE_TCP_DIR, 4),
                ("udp".into(), HANDLE_UDP_DIR, 4),
                ("dns".into(), HANDLE_DNS_DIR, 4),
            ],
            HANDLE_INTERFACES_DIR => vec![("eth0".into(), HANDLE_ETH0_DIR, 4)],
            HANDLE_ETH0_DIR => vec![
                ("status".into(), HANDLE_ETH0_STATUS, 8),
                ("addr".into(), HANDLE_ETH0_ADDR, 8),
                ("flags".into(), HANDLE_ETH0_FLAGS, 8),
                ("mtu".into(), HANDLE_ETH0_MTU, 8),
                ("stats".into(), HANDLE_ETH0_STATS, 8),
                ("events".into(), HANDLE_ETH0_EVENTS, 8),
            ],
            HANDLE_TCP_DIR => {
                let mut entries = vec![("new".into(), HANDLE_TCP_NEW, 8)];
                for id in socket_api.tcp_socket_ids() {
                    let dh = TCP_DYN_BASE | ((id as u64) << 8) | SF_DIR as u64;
                    entries.push((alloc::format!("{}", id), dh, 4));
                }
                entries
            }
            HANDLE_UDP_DIR => {
                let mut entries = vec![("new".into(), HANDLE_UDP_NEW, 8)];
                for id in socket_api.udp_socket_ids() {
                    let dh = UDP_DYN_BASE | ((id as u64) << 8) | SF_DIR as u64;
                    entries.push((alloc::format!("{}", id), dh, 4));
                }
                entries
            }
            HANDLE_DNS_DIR => {
                vec![("server".into(), HANDLE_DNS_SERVER, 8)]
            }
            // Dynamic TCP socket directory
            h if h >= TCP_DYN_BASE && h < UDP_DYN_BASE && (h & 0xFF) == SF_DIR as u64 => {
                let bid = TCP_DYN_BASE | (h & !0xFF);
                vec![
                    ("ctl".into(), bid | SF_CTL as u64, 8),
                    ("data".into(), bid | SF_DATA as u64, 8),
                    ("accept".into(), bid | SF_ACCEPT as u64, 8),
                    ("status".into(), bid | SF_STATUS as u64, 8),
                    ("events".into(), bid | SF_EVENTS as u64, 8),
                ]
            }
            // Dynamic UDP socket directory
            h if h >= UDP_DYN_BASE && (h & 0xFF) == SF_DIR as u64 => {
                let bid = UDP_DYN_BASE | (h & !0xFF);
                vec![
                    ("ctl".into(), bid | SF_CTL as u64, 8),
                    ("data".into(), bid | SF_DATA as u64, 8),
                    ("status".into(), bid | SF_STATUS as u64, 8),
                ]
            }
            HANDLE_DNS_DIR => vec![("lookup".into(), HANDLE_DNS_LOOKUP, 8)],
            _ => vec![],
        }
    }

    fn stat_handle(&self, handle: u64, socket_api: &SocketApi) -> (u32, usize) {
        match handle {
            HANDLE_ROOT
            | HANDLE_INTERFACES_DIR
            | HANDLE_ETH0_DIR
            | HANDLE_TCP_DIR
            | HANDLE_UDP_DIR
            | HANDLE_DNS_DIR => (S_IFDIR | 0o555, 0),
            HANDLE_ETH0_STATUS => (S_IFREG | 0o444, self.eth0_status().len()),
            HANDLE_ETH0_ADDR => (S_IFREG | 0o644, self.eth0_addr_text().len()),
            HANDLE_ETH0_FLAGS => (S_IFREG | 0o222, 0),
            HANDLE_ETH0_MTU => (S_IFREG | 0o644, alloc::format!("{}\n", self.mtu).len()),
            HANDLE_ETH0_STATS => (S_IFREG | 0o444, self.eth0_stats().len()),
            HANDLE_ETH0_EVENTS => (S_IFREG | 0o444, 0),
            HANDLE_ROUTES => (S_IFREG | 0o644, 0),
            HANDLE_TCP_NEW | HANDLE_UDP_NEW => (S_IFREG | 0o444, 0),
            HANDLE_DNS_LOOKUP => (S_IFREG | 0o644, 0),
            HANDLE_DNS_SERVER => (S_IFREG | 0o444, self.dns_server_text().len()),
            h if h >= TCP_DYN_BASE && h < UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - TCP_DYN_BASE) >> 8) as u32;
                if socket_api.has_socket(api_handle) {
                    if sf == SF_DIR {
                        (S_IFDIR | 0o555, 0)
                    } else {
                        (S_IFREG | 0o644, 0)
                    }
                } else {
                    (0, 0) // not found
                }
            }
            h if h >= UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - UDP_DYN_BASE) >> 8) as u32;
                if socket_api.has_socket(api_handle) {
                    if sf == SF_DIR {
                        (S_IFDIR | 0o555, 0)
                    } else {
                        (S_IFREG | 0o644, 0)
                    }
                } else {
                    (0, 0) // not found
                }
            }
            _ => (0, 0),
        }
    }

    fn poll_handle(&self, handle: u64, socket_set: &mut SocketSet, socket_api: &SocketApi) -> u32 {
        match handle {
            HANDLE_ETH0_EVENTS => POLLIN,
            HANDLE_ETH0_STATUS | HANDLE_ETH0_ADDR | HANDLE_ETH0_MTU | HANDLE_ETH0_STATS
            | HANDLE_ROUTES | HANDLE_TCP_NEW | HANDLE_UDP_NEW => POLLIN,
            h if h >= TCP_DYN_BASE && h < UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - TCP_DYN_BASE) >> 8) as u32;
                socket_api.tcp_poll_ready(api_handle, sf, socket_set)
            }
            h if h >= UDP_DYN_BASE => {
                let sf = (h & 0xFF) as u8;
                let api_handle = ((h - UDP_DYN_BASE) >> 8) as u32;
                socket_api.udp_poll_ready(api_handle, sf, socket_set)
            }
            // dns/lookup is readable when a result is available
            HANDLE_DNS_LOOKUP => {
                if self.dns_result.is_some() {
                    POLLIN
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    // ── Text generators ───────────────────────────────────────────────────────

    fn eth0_status(&self) -> String {
        let state = if self.link_up { "up" } else { "down" };
        let link = if self.link_up { "up" } else { "down" };
        let mac = self.mac;
        let ip_line = match &self.ip_config {
            Some(c) => {
                let b = c.ip.as_bytes();
                alloc::format!(
                    "ipv4: {}.{}.{}.{}/{}\n",
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    c.prefix_len
                )
            }
            None => "ipv4: unassigned\n".into(),
        };
        alloc::format!(
            "state: {}\nlink: {}\nmac: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}\nmtu: {}\n{}",
            state,
            link,
            mac[0],
            mac[1],
            mac[2],
            mac[3],
            mac[4],
            mac[5],
            self.mtu,
            ip_line
        )
    }

    fn eth0_addr_text(&self) -> String {
        match &self.ip_config {
            Some(c) => {
                let b = c.ip.as_bytes();
                alloc::format!("{}.{}.{}.{}/{}\n", b[0], b[1], b[2], b[3], c.prefix_len)
            }
            None => "0.0.0.0/0\n".into(),
        }
    }

    fn eth0_stats(&self) -> String {
        alloc::format!(
            "rx_bytes: {}\ntx_bytes: {}\nrx_packets: {}\ntx_packets: {}\n",
            self.rx_bytes,
            self.tx_bytes,
            self.rx_packets,
            self.tx_packets
        )
    }

    fn routes_text(&self, _socket_api: &SocketApi) -> String {
        match &self.ip_config {
            Some(c) => {
                let gw = c.gateway.as_bytes();
                let net_b = c.ip.as_bytes();
                // Derive network address by masking
                alloc::format!(
                    "default via {}.{}.{}.{} dev eth0\n{}.{}.{}.0/{} dev eth0\n",
                    gw[0],
                    gw[1],
                    gw[2],
                    gw[3],
                    net_b[0],
                    net_b[1],
                    net_b[2],
                    c.prefix_len
                )
            }
            None => "# no routes\n".into(),
        }
    }

    fn dns_server_text(&self) -> String {
        match &self.ip_config {
            Some(c) => {
                let d = c.dns_server.as_bytes();
                alloc::format!("{}.{}.{}.{}\n", d[0], d[1], d[2], d[3])
            }
            None => "0.0.0.0\n".into(),
        }
    }
}

// ── Result types ─────────────────────────────────────────────────────────────

enum ReadResult {
    Data(Vec<u8>),
    EOF,
    Again,
    Error,
    NotSupported,
}

impl ReadResult {
    fn text_offset(text: &str, offset: u64) -> Self {
        let bytes = text.as_bytes();
        let off = offset as usize;
        if off >= bytes.len() {
            ReadResult::EOF
        } else {
            ReadResult::Data(bytes[off..].to_vec())
        }
    }
}

enum WriteResult {
    Ok(usize),
    Error,
    ReadOnly,
    NotSupported,
}

// ── Wire helpers ─────────────────────────────────────────────────────────────

fn send_resp(port: ChannelHandle, data: &[u8]) {
    let _ = channel_send(port, data);
}

fn send_err(port: ChannelHandle, errno: u8) {
    let _ = channel_send(port, &[errno]);
}

fn send_handle(port: ChannelHandle, handle: u64) {
    let mut resp = [0u8; 9];
    resp[0] = E_OK;
    resp[1..9].copy_from_slice(&handle.to_le_bytes());
    let _ = channel_send(port, &resp);
}

fn send_data(port: ChannelHandle, data: &[u8]) {
    let mut resp = Vec::with_capacity(5 + data.len());
    resp.push(E_OK);
    resp.extend_from_slice(&(data.len() as u32).to_le_bytes());
    resp.extend_from_slice(data);
    let _ = channel_send(port, &resp);
}

fn send_write_ok(port: ChannelHandle, bytes_written: u32) {
    let mut resp = [0u8; 5];
    resp[0] = E_OK;
    resp[1..5].copy_from_slice(&bytes_written.to_le_bytes());
    let _ = channel_send(port, &resp);
}

// ── IPv4 parse helper ─────────────────────────────────────────────────────────

fn parse_ipv4(s: &str) -> Option<Ipv4Address> {
    let parts: Vec<&str> = s.trim().split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    Some(Ipv4Address::new(
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
        parts[3].parse().ok()?,
    ))
}
