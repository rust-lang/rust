//! Network platform abstraction.
//!
//! ## Model
//!
//! Thing-OS networking is **VFS-backed**: every socket is a subtree of files
//! under `/net/`.  The kernel exposes no socket syscalls of its own; instead
//! the `netd` userland service mounts a VFS provider at `/net/` and services
//! all socket operations through ordinary file I/O.
//!
//! ### Network file tree
//!
//! ```text
//! /net/
//! ├── tcp/
//! │   ├── new              ← read: allocates a socket, returns its numeric id
//! │   └── <id>/
//! │       ├── ctl          ← write: "connect IP PORT"
//! │       │                          "listen PORT [BACKLOG]"
//! │       │                          "close"
//! │       ├── data         ← read/write: TCP byte stream
//! │       ├── accept       ← read: "<conn_id> <src_ip> <src_port>\n" (listener only)
//! │       ├── status       ← read: human-readable state / local / remote
//! │       └── events       ← pollable: "connected", "peer_closed", etc.
//! ├── udp/
//! │   ├── new              ← read: allocates a socket, returns its numeric id
//! │   └── <id>/
//! │       ├── ctl          ← write: "bind PORT"
//! │       │                          "connect IP PORT"  (sets default destination)
//! │       │                          "close"
//! │       ├── data         ← write: [4: dest_ipv4][2: dest_port_le][4: len_le][payload]
//! │       │                   read: [4: src_ipv4][2: src_port_le][4: len_le][payload]
//! │       └── status       ← read: human-readable state
//! └── dns/
//!     └── lookup           ← write: hostname; read: dotted-decimal IPv4 address
//! ```
//!
//! ### Connection lifecycle (TCP client)
//!
//! 1. Read `/net/tcp/new` → socket id `N`
//! 2. Open `/net/tcp/N/ctl` (write-only) and `/net/tcp/N/data` (rdwr)
//! 3. Write `"connect ADDR PORT"` to ctl
//! 4. Poll `/net/tcp/N/status` until `state: established`
//! 5. I/O on data fd; write `"close"` to ctl when done
//!
//! ### Connection lifecycle (TCP server)
//!
//! 1. Read `/net/tcp/new` → socket id `L`
//! 2. Open `/net/tcp/L/ctl` (write-only)
//! 3. Write `"listen PORT BACKLOG"` to ctl
//! 4. Poll `/net/tcp/L/accept` for POLLIN; read returns `"<conn_id> <ip> <port>"`
//! 5. Open `/net/tcp/<conn_id>/data` for I/O with the peer
//! 6. Repeat step 4 for subsequent connections; write `"close"` to ctl when done
//!
//! ### IPv6 policy
//!
//! Only IPv4 is supported.  All operations that receive an IPv6 address return
//! `EAFNOSUPPORT`.  This is an explicit, tested policy, not an oversight.
//!
//! ### Nonblocking mode
//!
//! All data file descriptors are opened with `O_NONBLOCK`.  The caller decides
//! whether to spin-wait, sleep-poll, or use `sys_fs_poll` for readiness.
//! Blocking wrappers (`tcp_connect`, `TcpListenerHandle::accept`) perform
//! their own timed polling loop and return `ETIMEDOUT` on expiry.
//!
//! ### Polling / readiness
//!
//! Future integration with a poll/wait_many mechanism should go through
//! the underlying VFS fds (`data_fd`, `accept_fd`), which the kernel VFS
//! layer already supports via `sys_fs_poll`.  No structural changes are
//! needed in this module; callers just poll the raw fds directly.

use super::vfs_flags::{O_NONBLOCK, O_RDONLY, O_RDWR, O_WRONLY};
use crate::syscall;
use crate::syscall::vfs::{vfs_close, vfs_open, vfs_poll, vfs_read, vfs_write};
use abi::errors::{Errno, SysResult};
use abi::syscall::{poll_flags, PollFd};
use spin::Mutex;

/// A VFS file descriptor returned by [`vfs_open`].
pub type Fd = u32;

// poll intervals

/// Nanoseconds to wait between connection-state polls.
const CONNECT_POLL_NS: u64 = 5_000_000;

/// Nanoseconds to wait between accept polls.
const ACCEPT_POLL_NS: u64 = 5_000_000;

fn wait_fd_or_sleep(
    fd: Fd,
    events: u16,
    deadline_ns: u64,
    fallback_sleep_ns: u64,
) -> SysResult<()> {
    match wait_fd(fd, events, deadline_ns) {
        Ok(()) => Ok(()),
        Err(Errno::ENOSYS) => {
            syscall::sleep_ns(fallback_sleep_ns);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

fn wait_fd(fd: Fd, events: u16, deadline_ns: u64) -> SysResult<()> {
    loop {
        let timeout_ms = if deadline_ns == 0 {
            u64::MAX
        } else {
            let now = syscall::monotonic_ns();
            if now >= deadline_ns {
                return Err(Errno::ETIMEDOUT);
            }
            ns_to_timeout_ms(deadline_ns.saturating_sub(now))
        };

        let mut pollfd = [PollFd {
            fd: fd as i32,
            events,
            revents: 0,
        }];

        match vfs_poll(&mut pollfd, timeout_ms) {
            Ok(0) => {
                if deadline_ns != 0 {
                    return Err(Errno::ETIMEDOUT);
                }
            }
            Ok(_) => return Ok(()),
            Err(Errno::EINTR) => continue,
            Err(e) => return Err(e),
        }
    }
}

fn ns_to_timeout_ms(ns: u64) -> u64 {
    let ms = ns.saturating_add(999_999) / 1_000_000;
    ms.max(1)
}

// IPv6 rejection

/// Returns `Err(EAFNOSUPPORT)` for any IPv6 address literal.
///
/// Thing-OS is explicitly IPv4-only for now.  Call this before any address
/// parse to produce a clean, documented error rather than a mysterious
/// parse failure.
pub fn reject_ipv6(addr: &str) -> SysResult<()> {
    if addr.contains(':') {
        Err(Errno::EAFNOSUPPORT)
    } else {
        Ok(())
    }
}

// TcpHandle

/// An open TCP connection backed by the `/net/tcp/<id>/` VFS subtree.
///
/// Obtained via [`tcp_connect`] (client) or [`TcpListenerHandle::accept`]
/// (server).  I/O goes through `data_fd`; control commands (`"close"`) go
/// through `ctl_fd`.
pub struct TcpHandle {
    /// Socket ID assigned by netd (index under `/net/tcp/`).
    pub id: u32,
    /// Read-write data channel: `/net/tcp/<id>/data`.
    pub data_fd: Fd,
    /// Write-only control channel: `/net/tcp/<id>/ctl`.
    pub ctl_fd: Fd,
    /// Whether the data fd is in nonblocking mode (PAL-level flag).
    nonblocking: bool,
    /// Userspace peek buffer: bytes read from VFS but not yet consumed.
    peek_buf: Mutex<alloc::vec::Vec<u8>>,
}

impl TcpHandle {
    /// Read up to `buf.len()` bytes from the TCP data stream.
    ///
    /// Drains any bytes staged by a prior [`peek`](Self::peek) call before
    /// reading from the VFS.  Returns `Err(EAGAIN)` when nonblocking and no
    /// data is available.
    pub fn read(&self, buf: &mut [u8]) -> SysResult<usize> {
        // Drain peek buffer first.
        let mut peek_buf = self.peek_buf.lock();
        if !peek_buf.is_empty() {
            let copy_len = buf.len().min(peek_buf.len());
            buf[..copy_len].copy_from_slice(&peek_buf[..copy_len]);
            let _ = peek_buf.drain(..copy_len);
            return Ok(copy_len);
        }
        drop(peek_buf);
        vfs_read(self.data_fd, buf)
    }

    /// Peek up to `buf.len()` bytes from the TCP data stream without consuming them.
    ///
    /// If the internal peek buffer is empty a VFS read is issued to fill it.
    /// The data is copied into `buf` but remains in the peek buffer so that
    /// the next [`read`](Self::read) call will return the same bytes.
    ///
    /// Returns `Err(EAGAIN)` when nonblocking and no data is available.
    pub fn peek(&self, buf: &mut [u8]) -> SysResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut peek_buf = self.peek_buf.lock();
        if peek_buf.is_empty() {
            let mut tmp = alloc::vec![0u8; buf.len()];
            let n = vfs_read(self.data_fd, &mut tmp)?;
            if n == 0 {
                return Ok(0);
            }
            peek_buf.extend_from_slice(&tmp[..n]);
        }
        let copy_len = buf.len().min(peek_buf.len());
        buf[..copy_len].copy_from_slice(&peek_buf[..copy_len]);
        Ok(copy_len)
    }

    /// Write `buf` to the TCP data stream.
    pub fn write(&self, buf: &[u8]) -> SysResult<usize> {
        vfs_write(self.data_fd, buf)
    }

    /// Send a raw control command to the TCP socket (e.g. `b"close"`).
    pub fn control(&self, cmd: &[u8]) -> SysResult<usize> {
        vfs_write(self.ctl_fd, cmd)
    }

    /// Returns `true` if this handle is in nonblocking mode.
    pub fn is_nonblocking(&self) -> bool {
        self.nonblocking
    }

    /// Switch to blocking (`false`) or nonblocking (`true`) mode.
    ///
    /// The underlying VFS fd is always `O_NONBLOCK`.  This flag is a
    /// PAL-level hint used by higher layers (e.g. `std::net::TcpStream`).
    pub fn set_nonblocking(&mut self, nonblocking: bool) {
        self.nonblocking = nonblocking;
    }

    /// Close the connection and release VFS file descriptors.
    pub fn close(self) {
        let _ = vfs_write(self.ctl_fd, b"close");
        let _ = vfs_close(self.data_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

// TcpListenerHandle

/// A TCP listening socket backed by the `/net/tcp/<id>/` VFS subtree.
///
/// Obtained via [`tcp_bind`].  Use [`accept`](Self::accept) to retrieve
/// incoming connections one at a time.
pub struct TcpListenerHandle {
    /// Socket ID assigned by netd.
    pub id: u32,
    /// Port this listener is bound to.
    pub port: u16,
    /// Write-only control channel.
    ctl_fd: Fd,
    /// Read-only accept channel: `/net/tcp/<id>/accept`.
    accept_fd: Fd,
    /// Whether accept is in nonblocking mode.
    nonblocking: bool,
}

impl TcpListenerHandle {
    /// Accept the next incoming connection.
    ///
    /// - **Blocking mode** (default): polls until a connection arrives, or
    ///   returns `Err(ETIMEDOUT)` after `deadline_ns` nanoseconds from boot.
    ///   Pass `0` for no deadline.
    /// - **Nonblocking mode**: returns `Err(EAGAIN)` immediately if no
    ///   connection is queued.
    ///
    /// On success, returns `(TcpHandle, remote_ipv4, remote_port)`.
    pub fn accept(&self, deadline_ns: u64) -> SysResult<(TcpHandle, [u8; 4], u16)> {
        use alloc::format;

        loop {
            let mut buf = [0u8; 64];
            match vfs_read(self.accept_fd, &mut buf) {
                Ok(n) if n > 0 => {
                    let text = core::str::from_utf8(&buf[..n])
                        .map_err(|_| Errno::EINVAL)?
                        .trim();
                    let (conn_id, remote_ip, remote_port) = parse_accept_response(text)?;

                    let data_path = format!("/net/tcp/{conn_id}/data");
                    let ctl_path = format!("/net/tcp/{conn_id}/ctl");
                    let data_fd = vfs_open(&data_path, O_RDWR | O_NONBLOCK)?;
                    let ctl_fd = match vfs_open(&ctl_path, O_WRONLY) {
                        Ok(fd) => fd,
                        Err(e) => {
                            let _ = vfs_close(data_fd);
                            return Err(e);
                        }
                    };
                    return Ok((
                        TcpHandle {
                            id: conn_id,
                            data_fd,
                            ctl_fd,
                            nonblocking: false,
                            peek_buf: Mutex::new(alloc::vec::Vec::new()),
                        },
                        remote_ip,
                        remote_port,
                    ));
                }
                Ok(_) | Err(Errno::EAGAIN) => {
                    if self.nonblocking {
                        return Err(Errno::EAGAIN);
                    }
                    if deadline_ns != 0 && syscall::monotonic_ns() >= deadline_ns {
                        return Err(Errno::ETIMEDOUT);
                    }
                    wait_fd_or_sleep(
                        self.accept_fd,
                        poll_flags::POLLIN,
                        deadline_ns,
                        ACCEPT_POLL_NS,
                    )?;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// The local port this listener is bound to.
    pub fn local_port(&self) -> u16 {
        self.port
    }

    /// Switch to nonblocking mode.
    pub fn set_nonblocking(&mut self, nonblocking: bool) {
        self.nonblocking = nonblocking;
    }

    /// Returns `true` if this listener is in nonblocking mode.
    pub fn is_nonblocking(&self) -> bool {
        self.nonblocking
    }

    /// Stop accepting connections and release VFS file descriptors.
    pub fn close(self) {
        let _ = vfs_write(self.ctl_fd, b"close");
        let _ = vfs_close(self.accept_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

/// Parse an accept-response line: `"<conn_id> <a>.<b>.<c>.<d> <port>"`.
fn parse_accept_response(text: &str) -> SysResult<(u32, [u8; 4], u16)> {
    let mut parts = text.split_whitespace();
    let conn_id: u32 = parts
        .next()
        .ok_or(Errno::EINVAL)?
        .parse()
        .map_err(|_| Errno::EINVAL)?;
    let ip_str = parts.next().ok_or(Errno::EINVAL)?;
    let port: u16 = parts
        .next()
        .ok_or(Errno::EINVAL)?
        .parse()
        .map_err(|_| Errno::EINVAL)?;
    let ip = parse_ipv4_bytes(ip_str)?;
    Ok((conn_id, ip, port))
}

// UdpHandle

/// An open UDP socket backed by the `/net/udp/<id>/` VFS subtree.
///
/// Obtained via [`udp_bind`].
///
/// ## Wire format
///
/// Each **write** to `data_fd`: `[4: dest_ipv4][2: dest_port_le][4: len_le][payload]`
///
/// Each **read** from `data_fd`: `[4: src_ipv4][2: src_port_le][4: len_le][payload]`
pub struct UdpHandle {
    /// Socket ID assigned by netd.
    pub id: u32,
    /// Local port this socket is bound to.
    pub local_port: u16,
    /// Read-write data channel: `/net/udp/<id>/data`.
    pub data_fd: Fd,
    /// Write-only control channel: `/net/udp/<id>/ctl`.
    pub ctl_fd: Fd,
    /// Default remote endpoint set by `connect()`, if any.
    connected_remote: Option<([u8; 4], u16)>,
    /// Whether the data fd is in nonblocking mode.
    nonblocking: bool,
}

impl UdpHandle {
    /// Send a datagram to `dest_addr:dest_port`.
    ///
    /// Returns the number of payload bytes sent.
    pub fn send_to(&self, data: &[u8], dest_addr: [u8; 4], dest_port: u16) -> SysResult<usize> {
        let mut pkt = alloc::vec![0u8; 4 + 2 + 4 + data.len()];
        pkt[..4].copy_from_slice(&dest_addr);
        pkt[4..6].copy_from_slice(&dest_port.to_le_bytes());
        pkt[6..10].copy_from_slice(&(data.len() as u32).to_le_bytes());
        pkt[10..].copy_from_slice(data);
        vfs_write(self.data_fd, &pkt)?;
        Ok(data.len())
    }

    /// Send a datagram to the connected remote address.
    ///
    /// Returns `Err(ENOTCONN)` if no remote address has been set.
    pub fn send(&self, data: &[u8]) -> SysResult<usize> {
        let (addr, port) = self.connected_remote.ok_or(Errno::ENOTCONN)?;
        self.send_to(data, addr, port)
    }

    /// Receive a datagram.
    ///
    /// Returns `(bytes_written_to_buf, src_ipv4, src_port)`.
    /// Returns `Err(EAGAIN)` when nonblocking and no datagram is available.
    pub fn recv_from(&self, buf: &mut [u8]) -> SysResult<(usize, [u8; 4], u16)> {
        let header_len = 4 + 2 + 4;
        let mut raw = alloc::vec![0u8; header_len + buf.len()];
        let n = vfs_read(self.data_fd, &mut raw)?;
        if n < header_len {
            return Err(Errno::EIO);
        }
        let src_ip: [u8; 4] = raw[..4].try_into().unwrap();
        let src_port = u16::from_le_bytes([raw[4], raw[5]]);
        let payload_len = u32::from_le_bytes([raw[6], raw[7], raw[8], raw[9]]) as usize;
        let copy_len = payload_len.min(buf.len()).min(n.saturating_sub(header_len));
        buf[..copy_len].copy_from_slice(&raw[header_len..header_len + copy_len]);
        Ok((copy_len, src_ip, src_port))
    }

    /// Receive a datagram, discarding the source address.
    ///
    /// Returns `Err(ENOTCONN)` if not connected.
    pub fn recv(&self, buf: &mut [u8]) -> SysResult<usize> {
        if self.connected_remote.is_none() {
            return Err(Errno::ENOTCONN);
        }
        let (n, ..) = self.recv_from(buf)?;
        Ok(n)
    }

    /// Set the default remote address for `send()` / `recv()`.
    pub fn connect(&mut self, addr: [u8; 4], port: u16) -> SysResult<()> {
        use alloc::format;
        let cmd = format!(
            "connect {}.{}.{}.{} {}",
            addr[0], addr[1], addr[2], addr[3], port
        );
        vfs_write(self.ctl_fd, cmd.as_bytes())?;
        self.connected_remote = Some((addr, port));
        Ok(())
    }

    /// The local port this socket is bound to.
    pub fn local_port(&self) -> u16 {
        self.local_port
    }

    /// Switch to nonblocking mode.
    pub fn set_nonblocking(&mut self, nonblocking: bool) {
        self.nonblocking = nonblocking;
    }

    /// Returns `true` if in nonblocking mode.
    pub fn is_nonblocking(&self) -> bool {
        self.nonblocking
    }

    /// Close the UDP socket and release VFS file descriptors.
    pub fn close(self) {
        let _ = vfs_write(self.ctl_fd, b"close");
        let _ = vfs_close(self.data_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

// connection state

/// Parsed connection state from `/net/tcp/<id>/status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcpState {
    Created,
    Connecting,
    Connected,
    Closed,
    Other,
}

fn read_tcp_state(id: u32) -> TcpState {
    use alloc::format;
    let path = format!("/net/tcp/{id}/status");
    let Ok(fd) = vfs_open(&path, O_RDONLY) else {
        return TcpState::Other;
    };
    let mut buf = [0u8; 256];
    let n = vfs_read(fd, &mut buf).unwrap_or(0);
    let _ = vfs_close(fd);

    let text = core::str::from_utf8(&buf[..n]).unwrap_or("");
    for line in text.lines() {
        if let Some(state) = line.strip_prefix("state: ") {
            return match state.trim() {
                "created" => TcpState::Created,
                "bound" | "syn-sent" | "syn-received" => TcpState::Connecting,
                "connected" | "established" | "fin-wait-1" | "fin-wait-2" | "close-wait" => {
                    TcpState::Connected
                }
                "closed" | "time-wait" | "closing" | "last-ack" => TcpState::Closed,
                _ => TcpState::Other,
            };
        }
    }
    TcpState::Other
}

// public API

/// Open a TCP connection to `addr` (dotted-decimal IPv4) on `port`.
///
/// Returns `Err(EAFNOSUPPORT)` for IPv6 addresses.
/// Returns `Err(ETIMEDOUT)` if `deadline_ns` (nanoseconds from boot, `0` =
/// no deadline) passes before the connection succeeds.
pub fn tcp_connect(addr: &str, port: u16, deadline_ns: u64) -> SysResult<TcpHandle> {
    use alloc::format;

    reject_ipv6(addr)?;

    // Step 1: allocate socket
    let new_fd = vfs_open("/net/tcp/new", O_RDONLY)?;
    let mut id_buf = [0u8; 32];
    let n = vfs_read(new_fd, &mut id_buf)?;
    vfs_close(new_fd)?;
    let id_str = core::str::from_utf8(&id_buf[..n])
        .map_err(|_| Errno::EINVAL)?
        .trim();
    let id: u32 = id_str.parse().map_err(|_| Errno::EINVAL)?;

    // Step 2: open ctl fd
    let ctl_path = format!("/net/tcp/{id}/ctl");
    let ctl_fd = vfs_open(&ctl_path, O_WRONLY)?;

    // Step 3: send connect command
    let cmd = format!("connect {addr} {port}");
    if let Err(e) = vfs_write(ctl_fd, cmd.as_bytes()) {
        let _ = vfs_close(ctl_fd);
        return Err(e);
    }

    // Step 4: open data fd
    let data_path = format!("/net/tcp/{id}/data");
    let data_fd = match vfs_open(&data_path, O_RDWR | O_NONBLOCK) {
        Ok(fd) => fd,
        Err(e) => {
            let _ = vfs_write(ctl_fd, b"close");
            let _ = vfs_close(ctl_fd);
            return Err(e);
        }
    };

    // Step 5: poll until connected or closed/timed-out
    loop {
        match read_tcp_state(id) {
            TcpState::Connected => break,
            TcpState::Closed => {
                let _ = vfs_close(data_fd);
                let _ = vfs_write(ctl_fd, b"close");
                let _ = vfs_close(ctl_fd);
                return Err(Errno::ECONNREFUSED);
            }
            _ => {}
        }
        if deadline_ns != 0 && syscall::monotonic_ns() >= deadline_ns {
            let _ = vfs_close(data_fd);
            let _ = vfs_write(ctl_fd, b"close");
            let _ = vfs_close(ctl_fd);
            return Err(Errno::ETIMEDOUT);
        }
        wait_fd_or_sleep(data_fd, poll_flags::POLLOUT, deadline_ns, CONNECT_POLL_NS)?;
    }

    Ok(TcpHandle {
        id,
        data_fd,
        ctl_fd,
        nonblocking: false,
        peek_buf: Mutex::new(alloc::vec::Vec::new()),
    })
}

/// Bind a TCP listener on `addr:port`, with up to `backlog` queued connections.
///
/// Use `"0.0.0.0"` to bind on all interfaces.
/// Returns `Err(EAFNOSUPPORT)` if `addr` is an IPv6 literal.
pub fn tcp_bind(addr: &str, port: u16, backlog: u16) -> SysResult<TcpListenerHandle> {
    use alloc::format;

    reject_ipv6(addr)?;

    // Step 1: allocate socket
    let new_fd = vfs_open("/net/tcp/new", O_RDONLY)?;
    let mut id_buf = [0u8; 32];
    let n = vfs_read(new_fd, &mut id_buf)?;
    vfs_close(new_fd)?;
    let id_str = core::str::from_utf8(&id_buf[..n])
        .map_err(|_| Errno::EINVAL)?
        .trim();
    let id: u32 = id_str.parse().map_err(|_| Errno::EINVAL)?;

    // Step 2: open ctl fd
    let ctl_path = format!("/net/tcp/{id}/ctl");
    let ctl_fd = vfs_open(&ctl_path, O_WRONLY)?;

    // Step 3: listen
    let backlog = backlog.max(1);
    let cmd = format!("listen {port} {backlog}");
    if let Err(e) = vfs_write(ctl_fd, cmd.as_bytes()) {
        let _ = vfs_close(ctl_fd);
        return Err(e);
    }

    // Step 4: open accept fd
    let accept_path = format!("/net/tcp/{id}/accept");
    let accept_fd = match vfs_open(&accept_path, O_RDONLY | O_NONBLOCK) {
        Ok(fd) => fd,
        Err(e) => {
            let _ = vfs_write(ctl_fd, b"close");
            let _ = vfs_close(ctl_fd);
            return Err(e);
        }
    };

    Ok(TcpListenerHandle {
        id,
        port,
        ctl_fd,
        accept_fd,
        nonblocking: false,
    })
}

/// Bind a UDP socket to `addr:port`.
///
/// Use port `0` to let netd assign an ephemeral port.
/// Returns `Err(EAFNOSUPPORT)` if `addr` contains a colon (IPv6).
pub fn udp_bind(addr: &str, port: u16) -> SysResult<UdpHandle> {
    use alloc::format;

    reject_ipv6(addr)?;

    // Step 1: allocate socket
    let new_fd = vfs_open("/net/udp/new", O_RDONLY)?;
    let mut id_buf = [0u8; 32];
    let n = vfs_read(new_fd, &mut id_buf)?;
    vfs_close(new_fd)?;
    let id_str = core::str::from_utf8(&id_buf[..n])
        .map_err(|_| Errno::EINVAL)?
        .trim();
    let id: u32 = id_str.parse().map_err(|_| Errno::EINVAL)?;

    // Step 2: open ctl fd
    let ctl_path = format!("/net/udp/{id}/ctl");
    let ctl_fd = vfs_open(&ctl_path, O_WRONLY)?;

    // Step 3: bind to port
    let cmd = format!("bind {port}");
    if let Err(e) = vfs_write(ctl_fd, cmd.as_bytes()) {
        let _ = vfs_close(ctl_fd);
        return Err(e);
    }

    // Step 4: open data fd
    let data_path = format!("/net/udp/{id}/data");
    let data_fd = match vfs_open(&data_path, O_RDWR | O_NONBLOCK) {
        Ok(fd) => fd,
        Err(e) => {
            let _ = vfs_write(ctl_fd, b"close");
            let _ = vfs_close(ctl_fd);
            return Err(e);
        }
    };

    Ok(UdpHandle {
        id,
        local_port: port,
        data_fd,
        ctl_fd,
        connected_remote: None,
        nonblocking: false,
    })
}

/// Resolve a hostname to an IPv4 address via `/net/dns/lookup`.
///
/// IPv4 literals are parsed locally without a VFS round-trip.
/// Returns `Err(EAFNOSUPPORT)` for IPv6 literals.
/// Returns `Err(ETIMEDOUT)` when `deadline_ns` (nanoseconds from boot,
/// `0` = no deadline) passes before resolution completes.
pub fn resolve_hostname(hostname: &str, deadline_ns: u64) -> SysResult<[u8; 4]> {
    // Fast path: parse IPv4 literals locally.
    if let Some(ip) = try_parse_ipv4(hostname) {
        return Ok(ip);
    }

    // Reject IPv6 literals.
    if hostname.contains(':') || hostname.starts_with('[') {
        return Err(Errno::EAFNOSUPPORT);
    }

    // Perform DNS lookup via netd VFS.
    let lookup_fd = vfs_open("/net/dns/lookup", O_RDWR)?;

    if let Err(e) = vfs_write(lookup_fd, hostname.as_bytes()) {
        let _ = vfs_close(lookup_fd);
        return Err(e);
    }

    let mut buf = [0u8; 32];
    loop {
        match vfs_read(lookup_fd, &mut buf) {
            Ok(n) if n > 0 => {
                let _ = vfs_close(lookup_fd);
                let text = core::str::from_utf8(&buf[..n])
                    .map_err(|_| Errno::EINVAL)?
                    .trim();
                return parse_ipv4_bytes(text);
            }
            Ok(_) | Err(Errno::EAGAIN) => {
                if deadline_ns != 0 && syscall::monotonic_ns() >= deadline_ns {
                    let _ = vfs_close(lookup_fd);
                    return Err(Errno::ETIMEDOUT);
                }
                wait_fd_or_sleep(lookup_fd, poll_flags::POLLIN, deadline_ns, CONNECT_POLL_NS)?;
            }
            Err(e) => {
                let _ = vfs_close(lookup_fd);
                return Err(e);
            }
        }
    }
}

// Address helpers

/// Try to parse a dotted-decimal IPv4 address.  Returns `None` if the
/// string is not a valid IPv4 literal (e.g. a hostname or IPv6 address).
pub fn try_parse_ipv4(s: &str) -> Option<[u8; 4]> {
    let s = s.trim();
    let mut iter = s.split('.');
    let a: u8 = iter.next()?.parse().ok()?;
    let b: u8 = iter.next()?.parse().ok()?;
    let c: u8 = iter.next()?.parse().ok()?;
    let d_rest = iter.next()?;
    if iter.next().is_some() {
        return None; // too many components
    }
    let d: u8 = d_rest.parse().ok()?;
    Some([a, b, c, d])
}

/// Parse a dotted-decimal IPv4 address, returning `Err(EINVAL)` on failure.
pub fn parse_ipv4_bytes(s: &str) -> SysResult<[u8; 4]> {
    try_parse_ipv4(s).ok_or(Errno::EINVAL)
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_ipv6_plain_ipv6() {
        assert_eq!(reject_ipv6("::1"), Err(Errno::EAFNOSUPPORT));
        assert_eq!(reject_ipv6("2001:db8::1"), Err(Errno::EAFNOSUPPORT));
        assert_eq!(reject_ipv6("fe80::1%eth0"), Err(Errno::EAFNOSUPPORT));
    }

    #[test]
    fn test_reject_ipv6_allows_ipv4() {
        assert_eq!(reject_ipv6("127.0.0.1"), Ok(()));
        assert_eq!(reject_ipv6("0.0.0.0"), Ok(()));
        assert_eq!(reject_ipv6("192.168.1.1"), Ok(()));
    }

    #[test]
    fn test_reject_ipv6_allows_hostname() {
        assert_eq!(reject_ipv6("example.com"), Ok(()));
        assert_eq!(reject_ipv6("localhost"), Ok(()));
    }

    #[test]
    fn test_try_parse_ipv4_valid() {
        assert_eq!(try_parse_ipv4("127.0.0.1"), Some([127, 0, 0, 1]));
        assert_eq!(try_parse_ipv4("0.0.0.0"), Some([0, 0, 0, 0]));
        assert_eq!(
            try_parse_ipv4("255.255.255.255"),
            Some([255, 255, 255, 255])
        );
        assert_eq!(try_parse_ipv4("192.168.1.50"), Some([192, 168, 1, 50]));
        assert_eq!(try_parse_ipv4("  10.0.0.1  "), Some([10, 0, 0, 1]));
    }

    #[test]
    fn test_try_parse_ipv4_invalid() {
        assert_eq!(try_parse_ipv4("localhost"), None);
        assert_eq!(try_parse_ipv4("::1"), None);
        assert_eq!(try_parse_ipv4("1.2.3"), None);
        assert_eq!(try_parse_ipv4("1.2.3.4.5"), None);
        assert_eq!(try_parse_ipv4(""), None);
        assert_eq!(try_parse_ipv4("not-an-ip"), None);
    }

    #[test]
    fn test_try_parse_ipv4_overflow() {
        assert_eq!(try_parse_ipv4("256.0.0.0"), None);
        assert_eq!(try_parse_ipv4("0.0.0.256"), None);
    }

    #[test]
    fn test_parse_ipv4_bytes_valid() {
        assert_eq!(parse_ipv4_bytes("10.0.0.1"), Ok([10, 0, 0, 1]));
    }

    #[test]
    fn test_parse_ipv4_bytes_invalid() {
        assert_eq!(parse_ipv4_bytes("not-an-ip"), Err(Errno::EINVAL));
    }

    #[test]
    fn test_parse_accept_response_valid() {
        let (id, ip, port) = parse_accept_response("42 192.168.1.5 8080").unwrap();
        assert_eq!(id, 42);
        assert_eq!(ip, [192, 168, 1, 5]);
        assert_eq!(port, 8080);
    }

    #[test]
    fn test_parse_accept_response_loopback() {
        let (id, ip, port) = parse_accept_response("1 127.0.0.1 12345").unwrap();
        assert_eq!(id, 1);
        assert_eq!(ip, [127, 0, 0, 1]);
        assert_eq!(port, 12345);
    }

    #[test]
    fn test_parse_accept_response_invalid() {
        assert!(parse_accept_response("").is_err());
        assert!(parse_accept_response("42 not-an-ip 80").is_err());
        assert!(parse_accept_response("not-a-number 1.2.3.4 80").is_err());
        assert!(parse_accept_response("42 1.2.3.4").is_err());
    }
}
