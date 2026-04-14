//! Socket API module for netd
//!
//! Provides a high-level socket management API used by the VFS provider
//! to implement the `/net/` tree with smoltcp TCP/UDP sockets.
extern crate alloc;
use alloc::string::ToString;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::{vec, vec::Vec};
use smoltcp::iface::{Interface, SocketHandle, SocketSet};
use smoltcp::socket::tcp::{Socket as TcpSocket, SocketBuffer, State as TcpState};
use smoltcp::time::Instant;
use smoltcp::wire::{IpAddress, IpEndpoint, IpListenEndpoint, Ipv4Address};

use crate::dns;
use stem::{debug, info, trace, warn};

pub static mut CONN_RX: [[u8; 8192]; 256] = [[0; 8192]; 256];
pub static mut CONN_TX: [[u8; 32768]; 256] = [[0; 32768]; 256];

// Response types
pub const RESP_OK: u16 = 0x0000;
pub const RESP_ERROR: u16 = 0x0001;
pub const RESP_HANDLE: u16 = 0x0002;
pub const RESP_DATA: u16 = 0x0003;
pub const RESP_ACCEPT: u16 = 0x0004;
pub const RESP_EMPTY: u16 = 0x0005;
pub const RESP_CLOSED: u16 = 0x0006;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SocketType {
    Tcp,
    Udp,
}

#[derive(Debug, Clone, Copy)]
struct EndpointV4 {
    ip: Ipv4Address,
    port: u16,
}

#[derive(Debug)]
struct ManagedSocket {
    handle: SocketHandle,
    kind: SocketType,
    #[allow(dead_code)]
    is_listener: bool,
    local: Option<EndpointV4>,
    remote: Option<EndpointV4>,
    #[allow(dead_code)]
    owner_tid: u64,
    bytes_tx: u64,
    bytes_rx: u64,
    packets_tx: u64,
    packets_rx: u64,
    last_error_sym: u64,
    last_seen_ms: u64,
    /// Index of the backing static buffer, if any
    pub buf_idx: Option<usize>,
    /// Additional socket handles for listener pool backlog
    pub listen_pool: Vec<(SocketHandle, usize)>,
    /// Whether limited IPv4 broadcast sends are enabled for this UDP socket.
    pub udp_broadcast: bool,
}

impl ManagedSocket {
    #[allow(dead_code)]
    fn proto_str(&self) -> &'static str {
        match self.kind {
            SocketType::Tcp => "tcp",
            SocketType::Udp => "udp",
        }
    }
}

/// Socket API manager
pub struct SocketApi {
    /// Next handle ID to assign
    next_handle: u32,
    /// Map of API handles to managed sockets
    sockets: BTreeMap<u32, ManagedSocket>,
    /// Pending accepted connections (listen_handle -> Vec<(conn_handle, remote_ip, remote_port)>)
    pending_accepts: BTreeMap<u32, Vec<(u32, Ipv4Address, u16)>>,
    /// Socket handles pending removal from SocketSet (after TCP close completes)
    pending_removal: Vec<(SocketHandle, Option<usize>)>,
    /// Reusable scratch buffer for receive operations
    recv_scratch: Vec<u8>,
    /// List of free buffer indices (0..256)
    pub free_buffers: Vec<usize>,
}

impl SocketApi {
    pub fn new() -> Self {
        let mut free_buffers = Vec::with_capacity(256);
        for i in (0..256).rev() {
            free_buffers.push(i);
        }
        Self {
            next_handle: 1,
            sockets: BTreeMap::new(),
            pending_accepts: BTreeMap::new(),
            pending_removal: Vec::new(),
            recv_scratch: Vec::with_capacity(32768), // Large enough for most frames
            free_buffers,
        }
    }

    pub fn alloc_buffer(&mut self) -> Option<usize> {
        self.free_buffers.pop()
    }

    pub fn free_buffer(&mut self, idx: usize) {
        self.free_buffers.push(idx);
    }

    fn alloc_handle(&mut self) -> u32 {
        let h = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1);
        if self.next_handle == 0 {
            self.next_handle = 1;
        }
        h
    }

    /// Public handle allocator — used by the VFS provider to pre-register socket ids.
    #[allow(dead_code)]
    pub fn alloc_handle_pub(&mut self) -> u32 {
        self.alloc_handle()
    }

    fn now_ms() -> u64 {
        stem::time::now().as_millis() as u64
    }

    #[allow(dead_code)]
    fn endpoint_ip_string(ep: EndpointV4) -> alloc::string::String {
        let b = ep.ip.as_bytes();
        alloc::format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3])
    }

    fn tcp_state_label(state: TcpState) -> &'static str {
        match state {
            TcpState::Closed => "closed",
            TcpState::Listen => "listen",
            TcpState::SynSent => "syn-sent",
            TcpState::SynReceived => "syn-received",
            TcpState::Established => "established",
            TcpState::FinWait1 => "fin-wait-1",
            TcpState::FinWait2 => "fin-wait-2",
            TcpState::CloseWait => "close-wait",
            TcpState::Closing => "closing",
            TcpState::LastAck => "last-ack",
            TcpState::TimeWait => "time-wait",
        }
    }

    #[allow(dead_code)]
    fn socket_state_label(managed: &ManagedSocket, tcp_state: Option<TcpState>) -> &'static str {
        if managed.is_listener {
            return "listening";
        }
        match managed.kind {
            SocketType::Udp => {
                if managed.remote.is_some() {
                    "connected"
                } else if managed.local.is_some() {
                    "bound"
                } else {
                    "created"
                }
            }
            SocketType::Tcp => match tcp_state {
                Some(TcpState::Closed) => "closed",
                Some(TcpState::Listen) => "listening",
                Some(TcpState::Established)
                | Some(TcpState::FinWait1)
                | Some(TcpState::FinWait2)
                | Some(TcpState::CloseWait)
                | Some(TcpState::Closing)
                | Some(TcpState::LastAck)
                | Some(TcpState::TimeWait) => "connected",
                Some(TcpState::SynSent) | Some(TcpState::SynReceived) => "bound",
                None => {
                    if managed.local.is_some() {
                        "bound"
                    } else {
                        "created"
                    }
                }
            },
        }
    }

    fn set_last_error(managed: &mut ManagedSocket, _text: &str) {
        managed.last_error_sym = 0;
    }

    fn new_managed_socket(
        &mut self,
        socket_handle: SocketHandle,
        kind: SocketType,
        is_listener: bool,
        local: Option<EndpointV4>,
        owner_tid: u64,
        _api_handle: u32,
        now_ms: u64,
        buf_idx: Option<usize>,
    ) -> ManagedSocket {
        ManagedSocket {
            handle: socket_handle,
            kind,
            is_listener,
            local,
            remote: None,
            owner_tid,
            bytes_tx: 0,
            bytes_rx: 0,
            packets_tx: 0,
            packets_rx: 0,
            last_error_sym: 0,
            last_seen_ms: now_ms,
            buf_idx,
            listen_pool: Vec::new(),
            udp_broadcast: false,
        }
    }

    #[allow(dead_code)]
    fn sync_local_edge(_managed: &ManagedSocket) {}

    #[allow(dead_code)]
    fn sync_remote_edge(_managed: &ManagedSocket) {}

    #[allow(dead_code)]
    fn ensure_tcp_connection(_managed: &mut ManagedSocket, _now_ms: u64, _initial_state: &str) {}

    #[allow(dead_code)]
    fn flush_managed_socket_tcp<'a>(
        _managed: &mut ManagedSocket,
        _socket_set: &mut SocketSet<'a>,
        _now_ms: u64,
    ) {
    }

    #[allow(dead_code)]
    fn flush_managed_socket_udp(_managed: &ManagedSocket, _now_ms: u64) {}

    #[allow(dead_code)]
    pub fn flush_graph<'a>(&mut self, _socket_set: &mut SocketSet<'a>, _now_ms: u64) {}

    // ── VFS provider helpers ─────────────────────────────────────────────────

    /// Register a pre-created TCP socket handle with the pool.
    /// Returns the API handle (u32 id used in /net/tcp/<id>).
    pub fn alloc_socket_raw(
        &mut self,
        socket_handle: SocketHandle,
        buf_idx: usize,
        is_listener: bool,
        owner_tid: u64,
    ) -> u32 {
        let api_handle = self.alloc_handle();
        let now_ms = Self::now_ms();
        let managed = self.new_managed_socket(
            socket_handle,
            SocketType::Tcp,
            is_listener,
            None,
            owner_tid,
            api_handle,
            now_ms,
            Some(buf_idx),
        );
        self.sockets.insert(api_handle, managed);
        api_handle
    }

    /// Create a new UDP socket and register it. Returns the API handle, or None on error.
    pub fn alloc_udp_socket_raw<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        buf_idx: usize,
    ) -> Option<u32> {
        use smoltcp::socket::udp::{PacketBuffer, PacketMetadata};

        let (rx_meta, rx_payload) = unsafe {
            let meta_size = core::mem::size_of::<PacketMetadata>() * 8;
            let (m, p) = CONN_RX[buf_idx].split_at_mut(meta_size);
            let meta_ptr = m.as_mut_ptr() as *mut PacketMetadata;
            for i in 0..8 {
                *meta_ptr.add(i) = PacketMetadata::EMPTY;
            }
            (core::slice::from_raw_parts_mut(meta_ptr, 8), p)
        };
        let (tx_meta, tx_payload) = unsafe {
            let meta_size = core::mem::size_of::<PacketMetadata>() * 8;
            let (m, p) = CONN_TX[buf_idx].split_at_mut(meta_size);
            let meta_ptr = m.as_mut_ptr() as *mut PacketMetadata;
            for i in 0..8 {
                *meta_ptr.add(i) = PacketMetadata::EMPTY;
            }
            (core::slice::from_raw_parts_mut(meta_ptr, 8), p)
        };

        let rx_buf = PacketBuffer::new(rx_meta, rx_payload);
        let tx_buf = PacketBuffer::new(tx_meta, tx_payload);
        let socket = smoltcp::socket::udp::Socket::new(rx_buf, tx_buf);
        let socket_handle = socket_set.add(socket);

        let api_handle = self.alloc_handle();
        let now_ms = Self::now_ms();
        let managed = self.new_managed_socket(
            socket_handle,
            SocketType::Udp,
            false,
            None,
            0,
            api_handle,
            now_ms,
            Some(buf_idx),
        );
        self.sockets.insert(api_handle, managed);
        Some(api_handle)
    }

    /// Return the API handles of all active TCP sockets.
    pub fn tcp_socket_ids(&self) -> Vec<u32> {
        self.sockets
            .iter()
            .filter(|(_, s)| s.kind == SocketType::Tcp)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Return the API handles of all active UDP sockets.
    pub fn udp_socket_ids(&self) -> Vec<u32> {
        self.sockets
            .iter()
            .filter(|(_, s)| s.kind == SocketType::Udp)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns `true` if an API handle exists in the socket pool.
    pub fn has_socket(&self, api_handle: u32) -> bool {
        self.sockets.contains_key(&api_handle)
    }

    /// Return a human-readable status string for a TCP socket.
    pub fn tcp_status_text(&self, api_handle: u32, socket_set: &mut SocketSet) -> String {
        let managed = match self.sockets.get(&api_handle) {
            Some(s) if s.kind == SocketType::Tcp => s,
            _ => return "error: unknown\n".into(),
        };
        let socket = socket_set.get_mut::<TcpSocket>(managed.handle);
        let state = socket.state();
        let local = match socket.local_endpoint() {
            Some(ep) => alloc::format!("{}:{}", ep.addr, ep.port),
            None => "none".into(),
        };
        let remote = match socket.remote_endpoint() {
            Some(ep) => alloc::format!("{}:{}", ep.addr, ep.port),
            None => "none".into(),
        };
        alloc::format!(
            "state: {}\nlocal: {}\nremote: {}\n",
            Self::tcp_state_label(state),
            local,
            remote
        )
    }

    /// Return a pollable events string for a TCP socket.
    pub fn tcp_events_text(&self, api_handle: u32, socket_set: &mut SocketSet) -> String {
        let managed = match self.sockets.get(&api_handle) {
            Some(s) if s.kind == SocketType::Tcp => s,
            _ => return "error\n".into(),
        };
        let socket = socket_set.get_mut::<TcpSocket>(managed.handle);
        let state = socket.state();
        match state {
            TcpState::Established => "connected\n".into(),
            TcpState::CloseWait | TcpState::TimeWait => "peer_closed\n".into(),
            TcpState::Closed => "closed\n".into(),
            _ => "connecting\n".into(),
        }
    }

    /// Return a human-readable status string for a UDP socket.
    pub fn udp_status_text(&self, api_handle: u32) -> String {
        let managed = match self.sockets.get(&api_handle) {
            Some(s) if s.kind == SocketType::Udp => s,
            _ => return "error: unknown\n".into(),
        };
        let local_str = match managed.local {
            Some(ep) => {
                let b = ep.ip.as_bytes();
                alloc::format!("{}.{}.{}.{}:{}", b[0], b[1], b[2], b[3], ep.port)
            }
            None => "unbound".into(),
        };
        let remote_str = match managed.remote {
            Some(ep) => {
                let b = ep.ip.as_bytes();
                alloc::format!("{}.{}.{}.{}:{}", b[0], b[1], b[2], b[3], ep.port)
            }
            None => "none".into(),
        };
        alloc::format!(
            "local: {}\nremote: {}\nbroadcast: {}\n",
            local_str,
            remote_str,
            managed.udp_broadcast
        )
    }

    /// Return the stored remote endpoint for a UDP socket.
    ///
    /// May be used for diagnostic or future "connected UDP" fast-path.
    #[allow(dead_code)]
    pub fn udp_remote(&self, api_handle: u32) -> Option<(Ipv4Address, u16)> {
        let managed = self.sockets.get(&api_handle)?;
        let remote = managed.remote?;
        Some((remote.ip, remote.port))
    }

    /// Check poll readiness for a TCP socket's sub-file.
    pub fn tcp_poll_ready(&self, api_handle: u32, sf: u8, socket_set: &mut SocketSet) -> u32 {
        // SF constants: DIR=0, CTL=1, DATA=2, STATUS=3, EVENTS=4, ACCEPT=5
        const SF_DATA: u8 = 2;
        const SF_EVENTS: u8 = 4;
        const SF_ACCEPT: u8 = 5;
        let managed = match self.sockets.get(&api_handle) {
            Some(s) if s.kind == SocketType::Tcp => s,
            _ => return 0,
        };
        let socket = socket_set.get_mut::<TcpSocket>(managed.handle);
        match sf {
            SF_DATA => {
                let mut ready = 0u32;
                if socket.can_recv() {
                    ready |= 0x0001; // POLLIN
                }
                if socket.can_send() {
                    ready |= 0x0004; // POLLOUT
                }
                ready
            }
            SF_EVENTS => {
                let state = socket.state();
                if state == TcpState::Established
                    || state == TcpState::CloseWait
                    || state == TcpState::Closed
                {
                    0x0001 // POLLIN — there is something to read from events
                } else {
                    0
                }
            }
            SF_ACCEPT => {
                // Check if any connection is established on listener or pool
                if socket.state() == TcpState::Established {
                    return 0x0001; // POLLIN
                }
                for &(pool_handle, _) in &managed.listen_pool {
                    if socket_set.get_mut::<TcpSocket>(pool_handle).state() == TcpState::Established
                    {
                        return 0x0001; // POLLIN
                    }
                }
                // Also check pending_accepts
                if let Some(pending) = self.pending_accepts.get(&api_handle) {
                    if !pending.is_empty() {
                        return 0x0001;
                    }
                }
                0
            }
            _ => 0x0001, // status/ctl always readable/writable
        }
    }

    /// Check poll readiness for a UDP socket's sub-file.
    pub fn udp_poll_ready(&self, api_handle: u32, sf: u8, socket_set: &mut SocketSet) -> u32 {
        // SF constants: DIR=0, CTL=1, DATA=2, STATUS=3
        const SF_DATA: u8 = 2;
        let managed = match self.sockets.get(&api_handle) {
            Some(s) if s.kind == SocketType::Udp => s,
            _ => return 0,
        };
        match sf {
            SF_DATA => {
                let socket = socket_set.get_mut::<smoltcp::socket::udp::Socket>(managed.handle);
                let mut ready = 0u32;
                if socket.can_recv() {
                    ready |= 0x0001;
                }
                if socket.can_send() {
                    ready |= 0x0004;
                }
                ready
            }
            _ => 0x0001, // status/ctl always readable/writable
        }
    }

    /// Connect an *already-allocated* TCP socket (handle points to a socket in
    /// the pool created via `alloc_socket_raw`).
    pub fn handle_connect_existing<'a>(
        &mut self,
        iface: &mut Interface,
        socket_set: &mut SocketSet<'a>,
        api_handle: u32,
        remote_ip: Ipv4Address,
        remote_port: u16,
    ) -> bool {
        let socket_handle = match self.sockets.get(&api_handle) {
            Some(s) if s.kind == SocketType::Tcp => s.handle,
            _ => return false,
        };

        let endpoint = IpEndpoint::new(IpAddress::Ipv4(remote_ip), remote_port);
        let local_port = 49152 + (self.next_handle as u16 % 16384);

        let socket = socket_set.get_mut::<TcpSocket>(socket_handle);
        match socket.connect(iface.context(), endpoint, local_port) {
            Ok(()) => {
                if let Some(m) = self.sockets.get_mut(&api_handle) {
                    m.remote = Some(EndpointV4 {
                        ip: remote_ip,
                        port: remote_port,
                    });
                    m.local = Some(EndpointV4 {
                        ip: Ipv4Address::new(0, 0, 0, 0),
                        port: local_port,
                    });
                }
                info!(
                    "SOCKET_API: connect_existing handle={} to {}:{}",
                    api_handle, remote_ip, remote_port
                );
                true
            }
            Err(e) => {
                warn!("SOCKET_API: connect_existing failed: {:?}", e);
                false
            }
        }
    }

    /// Connect an *already-allocated* UDP socket to a remote address (sets
    /// the default send destination).
    pub fn handle_udp_connect<'a>(
        &mut self,
        _socket_set: &mut SocketSet<'a>,
        api_handle: u32,
        remote_ip: Ipv4Address,
        remote_port: u16,
    ) -> bool {
        if let Some(m) = self.sockets.get_mut(&api_handle) {
            if m.kind == SocketType::Udp {
                m.remote = Some(EndpointV4 {
                    ip: remote_ip,
                    port: remote_port,
                });
                return true;
            }
        }
        false
    }

    pub fn handle_udp_set_broadcast(&mut self, api_handle: u32, enabled: bool) -> bool {
        if let Some(managed) = self.sockets.get_mut(&api_handle) {
            if managed.kind == SocketType::Udp {
                managed.udp_broadcast = enabled;
                return true;
            }
        }
        false
    }

    /// Handle a TCP_LISTEN request
    #[allow(dead_code)]
    pub fn handle_listen<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        owner_tid: u64,
        port: u16,
        backlog: u16,
        buf_idx: usize,
    ) -> Vec<u8> {
        info!(
            "SOCKET_API: TCP_LISTEN on port {} with backlog {}",
            port, backlog
        );

        let rx_buffer = SocketBuffer::new(unsafe { &mut CONN_RX[buf_idx][..] });
        let tx_buffer = SocketBuffer::new(unsafe { &mut CONN_TX[buf_idx][..] });
        let mut socket = TcpSocket::new(rx_buffer, tx_buffer);

        let endpoint = IpListenEndpoint::from(port);
        if let Err(e) = socket.listen(endpoint) {
            warn!("SOCKET_API: Failed to listen: {:?}", e);
            self.free_buffer(buf_idx);
            return encode_error();
        }

        let socket_handle = socket_set.add(socket);
        let api_handle = self.alloc_handle();
        let now_ms = Self::now_ms();

        let mut managed = self.new_managed_socket(
            socket_handle,
            SocketType::Tcp,
            true,
            Some(EndpointV4 {
                ip: Ipv4Address::new(0, 0, 0, 0),
                port,
            }),
            owner_tid,
            api_handle,
            now_ms,
            Some(buf_idx),
        );

        let fill_backlog = core::cmp::min(core::cmp::max(backlog, 1), 16) - 1;
        for _ in 0..fill_backlog {
            let Some(b_idx) = self.alloc_buffer() else {
                break;
            };
            let rx_buf = SocketBuffer::new(unsafe { &mut CONN_RX[b_idx][..] });
            let tx_buf = SocketBuffer::new(unsafe { &mut CONN_TX[b_idx][..] });
            let mut sock = TcpSocket::new(rx_buf, tx_buf);
            if sock.listen(endpoint).is_ok() {
                managed.listen_pool.push((socket_set.add(sock), b_idx));
            } else {
                self.free_buffer(b_idx);
            }
        }

        self.sockets.insert(api_handle, managed);
        self.pending_accepts.insert(api_handle, Vec::new());

        info!(
            "SOCKET_API: Listening on port {}, handle={}",
            port, api_handle
        );
        encode_handle(api_handle)
    }

    /// Handle a TCP_CONNECT request
    #[allow(dead_code)]
    pub fn handle_connect<'a, D: smoltcp::phy::Device>(
        &mut self,
        iface: &mut Interface,
        _device: &mut D,
        socket_set: &mut SocketSet<'a>,
        owner_tid: u64,
        remote_ip: Ipv4Address,
        remote_port: u16,
        buf_idx: usize,
    ) -> Vec<u8> {
        info!("SOCKET_API: TCP_CONNECT to {}:{}", remote_ip, remote_port);

        let rx_buffer = SocketBuffer::new(unsafe { &mut CONN_RX[buf_idx][..] });
        let tx_buffer = SocketBuffer::new(unsafe { &mut CONN_TX[buf_idx][..] });
        let mut socket = TcpSocket::new(rx_buffer, tx_buffer);

        let endpoint = IpEndpoint::new(IpAddress::Ipv4(remote_ip), remote_port);
        // Ephemeral port generation is handled by smoltcp if local_port is not specified (unspecified endpoint)
        let local_port = 49152 + (self.next_handle as u16 % 16384);

        if let Err(e) = socket.connect(iface.context(), endpoint, local_port) {
            warn!("SOCKET_API: Failed to connect: {:?}", e);
            return encode_error();
        }

        let socket_handle = socket_set.add(socket);
        let api_handle = self.alloc_handle();
        let now_ms = Self::now_ms();

        let mut managed = self.new_managed_socket(
            socket_handle,
            SocketType::Tcp,
            false,
            Some(EndpointV4 {
                ip: Ipv4Address::new(0, 0, 0, 0),
                port: local_port,
            }),
            owner_tid,
            api_handle,
            now_ms,
            Some(buf_idx),
        );
        managed.remote = Some(EndpointV4 {
            ip: remote_ip,
            port: remote_port,
        });
        // DEFERRED: No blocking graph operations in network hot-path
        // Self::sync_remote_edge(&managed);
        // Self::ensure_tcp_connection(&mut managed, now_ms, "syn-sent");

        self.sockets.insert(api_handle, managed);

        info!(
            "SOCKET_API: Connected handle={} local_port={}",
            api_handle, local_port
        );
        encode_handle(api_handle)
    }

    /// Handle a DNS query
    #[allow(dead_code)]
    pub fn handle_dns_query<D: smoltcp::phy::Device>(
        &mut self,
        iface: &mut Interface,
        device: &mut D,
        dns_server: Ipv4Address,
        hostname: &str,
    ) -> Vec<u8> {
        match dns::lookup_a(iface, device, dns_server, hostname) {
            Ok(ip) => {
                let mut v = Vec::with_capacity(6);
                v.extend_from_slice(&RESP_DATA.to_le_bytes());
                v.extend_from_slice(ip.as_bytes());
                v
            }
            Err(_) => encode_error(),
        }
    }

    /// Handle a TCP_ACCEPT request (non-blocking)
    #[allow(dead_code)]
    pub fn handle_accept<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        listen_handle: u32,
        _owner_tid: u64,
        buf_idx: usize,
    ) -> Vec<u8> {
        // Check if we have a pending accepted connection
        if let Some(pending) = self.pending_accepts.get_mut(&listen_handle) {
            if let Some((conn_handle, remote_ip, remote_port)) = pending.pop() {
                debug!(
                    "SOCKET_API: TCP_ACCEPT returning connection handle={} from {}:{}",
                    conn_handle, remote_ip, remote_port
                );
                return encode_accept(conn_handle, remote_ip, remote_port);
            }
        }

        // No pending connections - check if listener socket has a connection ready
        let mut connected_slot: Option<(SocketHandle, usize, bool, usize)> = None;
        let (listen_port, socket_owner) = {
            let s = match self.sockets.get(&listen_handle) {
                Some(s) if s.is_listener && s.kind == SocketType::Tcp => s,
                Some(_) => return encode_error(),
                None => return encode_error(),
            };
            let lp = s.local.map(|e| e.port).unwrap_or(80);
            let ot = s.owner_tid;

            let state = socket_set.get_mut::<TcpSocket>(s.handle).state();
            if state == TcpState::Established {
                connected_slot = Some((s.handle, s.buf_idx.unwrap_or(0), true, 0));
            } else {
                for (i, &(pool_handle, pool_bidx)) in s.listen_pool.iter().enumerate() {
                    if socket_set.get_mut::<TcpSocket>(pool_handle).state() == TcpState::Established
                    {
                        connected_slot = Some((pool_handle, pool_bidx, false, i));
                        break;
                    }
                }
            }
            (lp, ot)
        };

        let (listener_socket_handle, bidx_connected, is_main, pool_idx) = match connected_slot {
            Some(slot) => slot,
            None => return encode_empty(),
        };

        let socket = socket_set.get_mut::<TcpSocket>(listener_socket_handle);

        // Diagnostic: log the listener socket state on every accept attempt
        let state = socket.state();
        if state != TcpState::Listen {
            info!(
                "SOCKET_API: TCP_ACCEPT handle={} socket state={:?}",
                listen_handle, state
            );
        }

        // Check socket state - if it's established, we have a connection
        if socket.state() == TcpState::Established {
            let remote = socket.remote_endpoint();
            if let Some(ep) = remote {
                let IpAddress::Ipv4(remote_ip) = ep.addr;
                let remote_port = ep.port;

                debug!(
                    "SOCKET_API: Connection established from {}:{} on listener {}",
                    remote_ip, remote_port, listen_handle
                );

                let conn_handle = self.alloc_handle();

                // Set up the connection's ManagedSocket
                let now_ms = Self::now_ms();
                let mut conn_managed = self.new_managed_socket(
                    listener_socket_handle,
                    SocketType::Tcp,
                    false,
                    Some(EndpointV4 {
                        ip: Ipv4Address::new(0, 0, 0, 0),
                        port: listen_port,
                    }),
                    socket_owner,
                    conn_handle,
                    now_ms,
                    Some(bidx_connected),
                );
                conn_managed.remote = Some(EndpointV4 {
                    ip: remote_ip,
                    port: remote_port,
                });
                self.sockets.insert(conn_handle, conn_managed);
                self.pending_accepts.remove(&listen_handle);

                // Create a new listener socket on the same port
                let rx_buffer = SocketBuffer::new(unsafe { &mut CONN_RX[buf_idx][..] });
                let tx_buffer = SocketBuffer::new(unsafe { &mut CONN_TX[buf_idx][..] });
                let mut new_listener = TcpSocket::new(rx_buffer, tx_buffer);

                let endpoint = IpListenEndpoint::from(listen_port);
                if let Err(e) = new_listener.listen(endpoint) {
                    warn!("SOCKET_API: Failed to respawn listener: {:?}", e);
                    self.free_buffer(buf_idx);
                } else {
                    let new_socket_handle = socket_set.add(new_listener);
                    if let Some(s) = self.sockets.get_mut(&listen_handle) {
                        if is_main {
                            s.handle = new_socket_handle;
                            s.buf_idx = Some(buf_idx);
                        } else {
                            s.listen_pool[pool_idx] = (new_socket_handle, buf_idx);
                        }
                    } else {
                        self.free_buffer(buf_idx);
                    }
                }

                return encode_accept(conn_handle, remote_ip, remote_port);
            }
        }

        // No connection ready
        encode_empty()
    }

    /// Put an already-allocated TCP socket into listening mode.
    ///
    /// Used by the VFS provider when the client writes `"listen PORT [BACKLOG]"`
    /// to `/net/tcp/<id>/ctl`.  The socket must have been previously allocated
    /// via `alloc_socket_raw`.
    ///
    /// Returns `true` on success, `false` if the socket was not found or the
    /// listen call failed.
    pub fn handle_listen_existing<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        api_handle: u32,
        port: u16,
        backlog: u16,
    ) -> bool {
        let managed = match self.sockets.get_mut(&api_handle) {
            Some(s) if s.kind == SocketType::Tcp => s,
            _ => return false,
        };

        let endpoint = IpListenEndpoint::from(port);
        let socket = socket_set.get_mut::<TcpSocket>(managed.handle);
        if let Err(e) = socket.listen(endpoint) {
            warn!(
                "SOCKET_API: handle_listen_existing: listen failed on port {}: {:?}",
                port, e
            );
            return false;
        }

        managed.is_listener = true;
        if let Some(local) = managed.local.as_mut() {
            local.port = port;
        } else {
            managed.local = Some(EndpointV4 {
                ip: Ipv4Address::new(0, 0, 0, 0),
                port,
            });
        }

        // Fill the backlog pool with additional listener sockets.
        let fill_backlog = core::cmp::min(core::cmp::max(backlog, 1), 16).saturating_sub(1);
        for _ in 0..fill_backlog {
            let Some(b_idx) = self.alloc_buffer() else {
                break;
            };
            let rx_buf = SocketBuffer::new(unsafe { &mut CONN_RX[b_idx][..] });
            let tx_buf = SocketBuffer::new(unsafe { &mut CONN_TX[b_idx][..] });
            let mut sock = TcpSocket::new(rx_buf, tx_buf);
            if sock.listen(endpoint).is_ok() {
                if let Some(m) = self.sockets.get_mut(&api_handle) {
                    m.listen_pool.push((socket_set.add(sock), b_idx));
                } else {
                    self.free_buffer(b_idx);
                    break;
                }
            } else {
                self.free_buffer(b_idx);
            }
        }

        self.pending_accepts.insert(api_handle, Vec::new());

        info!(
            "SOCKET_API: Listening on port {}, handle={} (backlog={})",
            port, api_handle, backlog
        );
        true
    }

    /// Bind an already-allocated UDP socket to a local port.
    ///
    /// Used by the VFS provider when the client writes `"bind PORT"` to
    /// `/net/udp/<id>/ctl`.  The socket must have been previously allocated
    /// via `alloc_udp_socket_raw`.
    ///
    /// Returns `true` on success, `false` if the socket was not found or
    /// the bind call failed.
    pub fn handle_udp_bind_port<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        api_handle: u32,
        port: u16,
    ) -> bool {
        let managed = match self.sockets.get_mut(&api_handle) {
            Some(s) if s.kind == SocketType::Udp => s,
            _ => return false,
        };

        let socket = socket_set.get_mut::<smoltcp::socket::udp::Socket>(managed.handle);
        if let Err(e) = socket.bind(port) {
            warn!(
                "SOCKET_API: handle_udp_bind_port: bind failed on port {}: {:?}",
                port, e
            );
            return false;
        }

        managed.local = Some(EndpointV4 {
            ip: Ipv4Address::new(0, 0, 0, 0),
            port,
        });

        info!(
            "SOCKET_API: Bound UDP on port {}, handle={}",
            port, api_handle
        );
        true
    }

    /// Handle a UDP_BIND request
    #[allow(dead_code)]
    pub fn handle_udp_bind<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        owner_tid: u64,
        port: u16,
        rx_metadata_storage: &'a mut [smoltcp::socket::udp::PacketMetadata],
        rx_payload_storage: &'a mut [u8],
        tx_metadata_storage: &'a mut [smoltcp::socket::udp::PacketMetadata],
        tx_payload_storage: &'a mut [u8],
        buf_idx: usize,
    ) -> Vec<u8> {
        info!("SOCKET_API: UDP_BIND on port {}", port);

        let rx_buffer =
            smoltcp::socket::udp::PacketBuffer::new(rx_metadata_storage, rx_payload_storage);
        let tx_buffer =
            smoltcp::socket::udp::PacketBuffer::new(tx_metadata_storage, tx_payload_storage);
        let mut socket = smoltcp::socket::udp::Socket::new(rx_buffer, tx_buffer);

        if let Err(e) = socket.bind(port) {
            warn!("SOCKET_API: Failed to bind UDP: {:?}", e);
            return encode_error();
        }

        let socket_handle = socket_set.add(socket);
        let api_handle = self.alloc_handle();
        let now_ms = Self::now_ms();

        let managed = self.new_managed_socket(
            socket_handle,
            SocketType::Udp,
            false,
            Some(EndpointV4 {
                ip: Ipv4Address::new(0, 0, 0, 0),
                port,
            }),
            owner_tid,
            api_handle,
            now_ms,
            Some(buf_idx),
        );

        self.sockets.insert(api_handle, managed);

        info!(
            "SOCKET_API: Bound UDP on port {}, handle={}",
            port, api_handle
        );
        encode_handle(api_handle)
    }

    /// Handle a UDP_SEND_TO request
    pub fn handle_udp_send_to<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        handle: u32,
        remote_ip: Ipv4Address,
        remote_port: u16,
        data: &[u8],
    ) -> Vec<u8> {
        let managed = match self.sockets.get(&handle) {
            Some(s) if s.kind == SocketType::Udp => s,
            _ => return encode_error(),
        };

        let socket = socket_set.get_mut::<smoltcp::socket::udp::Socket>(managed.handle);
        let endpoint = IpEndpoint::new(IpAddress::Ipv4(remote_ip), remote_port);

        if !socket.can_send() {
            return encode_send_result(0);
        }

        let result = match socket.send_slice(data, endpoint) {
            Ok(_) => {
                trace!(
                    "SOCKET_API: UDP_SEND_TO handle={} sent {} bytes to {}:{}",
                    handle,
                    data.len(),
                    remote_ip,
                    remote_port
                );
                encode_send_result(data.len() as u16)
            }
            Err(e) => {
                warn!("SOCKET_API: UDP_SEND_TO error: {:?}", e);
                encode_send_result(0)
            }
        };

        if let Some(m) = self.sockets.get_mut(&handle) {
            if result.len() >= 4 && u16::from_le_bytes([result[2], result[3]]) != 0 {
                m.bytes_tx = m.bytes_tx.saturating_add(data.len() as u64);
                m.packets_tx = m.packets_tx.saturating_add(1);
                m.last_seen_ms = Self::now_ms();
            }
            m.remote = Some(EndpointV4 {
                ip: remote_ip,
                port: remote_port,
            });
            Self::sync_remote_edge(m);
        }

        result
    }

    /// Handle a UDP_RECV_FROM request
    pub fn handle_udp_recv_from<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        handle: u32,
    ) -> Vec<u8> {
        let managed = match self.sockets.get(&handle) {
            Some(s) if s.kind == SocketType::Udp => s,
            _ => return encode_error(),
        };

        let socket = socket_set.get_mut::<smoltcp::socket::udp::Socket>(managed.handle);

        if !socket.can_recv() {
            return encode_empty();
        }

        match socket.recv() {
            Ok((data, endpoint)) => {
                let IpAddress::Ipv4(remote_ip) = endpoint.endpoint.addr;
                let remote_port = endpoint.endpoint.port;

                trace!(
                    "SOCKET_API: UDP_RECV_FROM handle={} got {} bytes from {}:{}",
                    handle,
                    data.len(),
                    remote_ip,
                    remote_port
                );

                if let Some(m) = self.sockets.get_mut(&handle) {
                    m.bytes_rx = m.bytes_rx.saturating_add(data.len() as u64);
                    m.packets_rx = m.packets_rx.saturating_add(1);
                    m.last_seen_ms = Self::now_ms();
                    m.remote = Some(EndpointV4 {
                        ip: remote_ip,
                        port: remote_port,
                    });
                    Self::sync_remote_edge(m);
                }

                encode_udp_data(remote_ip, remote_port, data)
            }
            Err(e) => {
                warn!("SOCKET_API: UDP_RECV_FROM error: {:?}", e);
                if let Some(m) = self.sockets.get_mut(&handle) {
                    Self::set_last_error(m, &alloc::format!("udp_recv:{:?}", e));
                }
                encode_empty()
            }
        }
    }

    /// Handle a NET_JOIN_MULTICAST request
    #[allow(dead_code)]
    pub fn handle_multicast_join<'a, D: smoltcp::phy::Device>(
        &mut self,
        iface: &mut Interface,
        device: &mut D,
        multicast_ip: Ipv4Address,
    ) -> Vec<u8> {
        info!("SOCKET_API: Joining multicast group {}", multicast_ip);
        let now = Instant::from_millis(stem::time::now().as_millis() as i64);
        iface
            .join_multicast_group(device, IpAddress::Ipv4(multicast_ip), now)
            .ok();
        encode_ok()
    }

    /// Handle a TCP_SEND request
    pub fn handle_send<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        handle: u32,
        data: &[u8],
    ) -> Vec<u8> {
        let managed = match self.sockets.get(&handle) {
            Some(s) if s.kind == SocketType::Tcp => s,
            _ => return encode_error(),
        };

        let socket = socket_set.get_mut::<TcpSocket>(managed.handle);

        if !socket.can_send() {
            return encode_send_result(0);
        }

        let response = match socket.send_slice(data) {
            Ok(sent) => {
                trace!("SOCKET_API: TCP_SEND handle={} sent {} bytes", handle, sent);
                encode_send_result(sent as u16)
            }
            Err(e) => {
                warn!("SOCKET_API: TCP_SEND error: {:?}", e);
                encode_send_result(0)
            }
        };

        if let Some(m) = self.sockets.get_mut(&handle) {
            if response.len() >= 4 {
                let sent = u16::from_le_bytes([response[2], response[3]]) as u64;
                if sent != 0 {
                    m.bytes_tx = m.bytes_tx.saturating_add(sent);
                    m.packets_tx = m.packets_tx.saturating_add(1);
                    m.last_seen_ms = Self::now_ms();
                }
            }
        }

        response
    }

    /// Handle a TCP_RECV request (non-blocking)
    pub fn handle_recv<'a>(
        &mut self,
        socket_set: &mut SocketSet<'a>,
        handle: u32,
        max_len: u16,
    ) -> Vec<u8> {
        let managed = match self.sockets.get(&handle) {
            Some(s) if s.kind == SocketType::Tcp => s,
            _ => return encode_error(),
        };

        let socket = socket_set.get_mut::<TcpSocket>(managed.handle);

        // Use reusable scratch buffer
        let max_len = max_len as usize;
        if self.recv_scratch.len() < max_len {
            self.recv_scratch.resize(max_len, 0);
        }

        match socket.recv_slice(&mut self.recv_scratch[..max_len]) {
            Ok(len) => {
                trace!("SOCKET_API: TCP_RECV handle={} got {} bytes", handle, len);
                if let Some(m) = self.sockets.get_mut(&handle) {
                    m.bytes_rx = m.bytes_rx.saturating_add(len as u64);
                    m.packets_rx = m.packets_rx.saturating_add(1);
                    m.last_seen_ms = Self::now_ms();
                }
                encode_data(&self.recv_scratch[..len])
            }
            Err(_) => {
                // If we can't receive, check if it's because the socket is empty or closed
                if socket.state() == TcpState::Established {
                    // Still established but no data
                    encode_empty()
                } else if socket.may_recv() {
                    // Still potentially receiving (e.g. FIN received but buffer not empty,
                    // though recv_slice would have returned data in that case)
                    encode_empty()
                } else {
                    // Socket closed or EOF reached
                    encode_closed()
                }
            }
        }
    }

    /// Handle a TCP_CLOSE request
    pub fn handle_close<'a>(&mut self, socket_set: &mut SocketSet<'a>, handle: u32) -> Vec<u8> {
        if let Some(managed) = self.sockets.get(&handle) {
            if managed.kind == SocketType::Tcp {
                let socket = socket_set.get_mut::<TcpSocket>(managed.handle);
                socket.close();
                debug!("SOCKET_API: TCP_CLOSE handle={} (initiating close)", handle);
                self.pending_removal.push((managed.handle, managed.buf_idx));
                // Listen pools need to be closed/freed too
                for &(pool_handle, pool_idx) in &managed.listen_pool {
                    socket_set.get_mut::<TcpSocket>(pool_handle).close();
                    self.pending_removal.push((pool_handle, Some(pool_idx)));
                }
            } else {
                // UDP sockets can be removed immediately
                socket_set.remove(managed.handle);
                debug!("SOCKET_API: UDP close handle={} (removed)", handle);
                if let Some(bidx) = managed.buf_idx {
                    self.free_buffer(bidx);
                }
            }
        }
        self.sockets.remove(&handle);
        self.pending_accepts.remove(&handle);
        encode_ok()
    }

    /// Garbage collect closed sockets from the SocketSet
    pub fn gc_closed_sockets<'a>(&mut self, socket_set: &mut SocketSet<'a>) {
        let mut new_pending = Vec::new();
        let mut to_free = Vec::new();
        for (socket_handle, buf_idx) in self.pending_removal.drain(..) {
            let socket = socket_set.get_mut::<TcpSocket>(socket_handle);
            if socket.state() == TcpState::Closed {
                socket_set.remove(socket_handle);
                debug!("SOCKET_API: GC removed explicitly closed TCP socket");
                if let Some(bidx) = buf_idx {
                    to_free.push(bidx);
                }
            } else {
                new_pending.push((socket_handle, buf_idx));
            }
        }
        self.pending_removal = new_pending;
        for bidx in to_free {
            self.free_buffer(bidx);
        }

        let mut tracked_handles = BTreeSet::new();
        for managed in self.sockets.values() {
            tracked_handles.insert(managed.handle);
        }

        let all_handles: Vec<SocketHandle> = socket_set.iter().map(|(h, _)| h).collect();
        let to_remove = Vec::new();
        for handle in all_handles {
            if tracked_handles.contains(&handle) {
                continue;
            }

            // For now, only GC TCP sockets that reached Closed state
            // (e.g., remotely closed ones we haven't handled yet)
            // This is a bit tricky with smoltcp's type system in a loop,
            // so we'll stick to the explicit pending_removal for now as primary GC.
        }

        for handle in to_remove {
            socket_set.remove(handle);
            debug!("SOCKET_API: GC removed orphaned socket");
        }
    }

    /// Return the number of currently tracked sockets
    pub fn socket_count(&self) -> usize {
        self.sockets.len()
    }
}

// Encoding helpers
fn encode_ok() -> Vec<u8> {
    RESP_OK.to_le_bytes().to_vec()
}

fn encode_error() -> Vec<u8> {
    RESP_ERROR.to_le_bytes().to_vec()
}

#[allow(dead_code)]
fn encode_handle(handle: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(6);
    v.extend_from_slice(&RESP_HANDLE.to_le_bytes());
    v.extend_from_slice(&handle.to_le_bytes());
    v
}

fn encode_send_result(bytes_sent: u16) -> Vec<u8> {
    let mut v = Vec::with_capacity(4);
    v.extend_from_slice(&RESP_OK.to_le_bytes());
    v.extend_from_slice(&bytes_sent.to_le_bytes());
    v
}

fn encode_data(data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(2 + data.len());
    v.extend_from_slice(&RESP_DATA.to_le_bytes());
    v.extend_from_slice(data);
    v
}

fn encode_udp_data(remote_ip: Ipv4Address, remote_port: u16, data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(2 + 4 + 2 + data.len());
    v.extend_from_slice(&RESP_DATA.to_le_bytes());
    v.extend_from_slice(remote_ip.as_bytes());
    v.extend_from_slice(&remote_port.to_le_bytes());
    v.extend_from_slice(data);
    v
}

#[allow(dead_code)]
fn encode_accept(conn_handle: u32, remote_ip: Ipv4Address, remote_port: u16) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&RESP_ACCEPT.to_le_bytes());
    v.extend_from_slice(&conn_handle.to_le_bytes());
    let ip_bytes = remote_ip.as_bytes();
    v.extend_from_slice(ip_bytes);
    v.extend_from_slice(&remote_port.to_le_bytes());
    v
}

fn encode_empty() -> Vec<u8> {
    RESP_EMPTY.to_le_bytes().to_vec()
}

fn encode_closed() -> Vec<u8> {
    RESP_CLOSED.to_le_bytes().to_vec()
}

/// Helper to split a large buffer into packet metadata and payload
#[allow(dead_code)]
unsafe fn split_packet_buffer(
    buf: &mut [u8],
) -> (&mut [smoltcp::socket::udp::PacketMetadata], &mut [u8]) {
    use core::mem::size_of;
    let meta_count = 8;
    let meta_size = size_of::<smoltcp::socket::udp::PacketMetadata>() * meta_count;

    let (meta_bytes, payload) = buf.split_at_mut(meta_size);
    let meta_ptr = meta_bytes.as_mut_ptr() as *mut smoltcp::socket::udp::PacketMetadata;
    let meta = core::slice::from_raw_parts_mut(meta_ptr, meta_count);

    // Initialize metadata
    for m in meta.iter_mut() {
        *m = smoltcp::socket::udp::PacketMetadata::EMPTY;
    }

    (meta, payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use smoltcp::wire::Ipv4Address;

    #[test]
    fn test_encode_ok() {
        let v = encode_ok();
        assert_eq!(v.len(), 2);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_OK);
    }

    #[test]
    fn test_encode_error() {
        let v = encode_error();
        assert_eq!(v.len(), 2);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_ERROR);
    }

    #[test]
    fn test_encode_empty() {
        let v = encode_empty();
        assert_eq!(v.len(), 2);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_EMPTY);
    }

    #[test]
    fn test_encode_closed() {
        let v = encode_closed();
        assert_eq!(v.len(), 2);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_CLOSED);
    }

    #[test]
    fn test_encode_data() {
        let payload = b"hello";
        let v = encode_data(payload);
        // 2 bytes type + 5 bytes payload
        assert_eq!(v.len(), 7);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_DATA);
        assert_eq!(&v[2..], b"hello");
    }

    #[test]
    fn test_encode_data_empty_payload() {
        let v = encode_data(b"");
        assert_eq!(v.len(), 2);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_DATA);
    }

    #[test]
    fn test_encode_udp_data() {
        let ip = Ipv4Address::new(192, 168, 1, 1);
        let port = 8080u16;
        let data = b"test";
        let v = encode_udp_data(ip, port, data);
        // 2 type + 4 ip + 2 port + 4 data
        assert_eq!(v.len(), 12);
        assert_eq!(u16::from_le_bytes([v[0], v[1]]), RESP_DATA);
        assert_eq!(&v[2..6], ip.as_bytes());
        assert_eq!(u16::from_le_bytes([v[6], v[7]]), port);
        assert_eq!(&v[8..], data);
    }

    #[test]
    fn test_resp_constants_distinct() {
        // Verify all response type constants are distinct
        let constants = [
            RESP_OK,
            RESP_ERROR,
            RESP_HANDLE,
            RESP_DATA,
            RESP_ACCEPT,
            RESP_EMPTY,
            RESP_CLOSED,
        ];
        for i in 0..constants.len() {
            for j in (i + 1)..constants.len() {
                assert_ne!(
                    constants[i], constants[j],
                    "constants[{}] == constants[{}]",
                    i, j
                );
            }
        }
    }
}
