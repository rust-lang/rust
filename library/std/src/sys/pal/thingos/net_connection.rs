use super::each_addr;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, SocketAddrV4, ToSocketAddrs};
use crate::string::String;
use crate::sync::{Arc, Mutex};
use crate::time::Duration;
use crate::{fmt, thread, vec};

use crate::sys::pal::raw_syscall6;

const SYS_SLEEP_NS: u32 = 0x1200;
const SYS_TIME_MONOTONIC: u32 = 0x1202;
const SYS_VFS_OPEN: u32 = 0x4000;
const SYS_VFS_CLOSE: u32 = 0x4001;
const SYS_VFS_READ: u32 = 0x4002;
const SYS_VFS_WRITE: u32 = 0x4003;
const SYS_VFS_READV: u32 = 0x4021;
const SYS_VFS_WRITEV: u32 = 0x4022;
const SYS_FS_DUP: u32 = 0x400C;

// ── Scatter-gather I/O vector (matches abi::syscall::IoVec / POSIX struct iovec) ──
#[repr(C)]
struct KernelIoVec {
    base: usize,
    len: usize,
}

const O_RDONLY: u32 = 0x0000;
const O_WRONLY: u32 = 0x0001;
const O_RDWR: u32 = 0x0002;
const O_NONBLOCK: u32 = 0x0800;

const EAGAIN: i32 = 11;
const EINVAL: i32 = 22;
const EPIPE: i32 = 32;
const EAFNOSUPPORT: i32 = 97;
const ETIMEDOUT: i32 = 110;
const ECONNREFUSED: i32 = 111;
const ENOTSUP: i32 = 95;
const ENOTCONN: i32 = 107;

const CONNECT_POLL_NS: u64 = 5_000_000;
const IO_POLL_NS: u64 = 1_000_000;

/// Default TCP listen backlog passed to netd.
const LISTEN_BACKLOG: u16 = 128;
/// Buffer large enough for a single accept-response line: `"<conn_id> <ip> <port>\n"`.
const ACCEPT_RESPONSE_BUF_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TcpState {
    Created,
    Bound,
    Connected,
    Closed,
    Other,
}

#[derive(Debug, Clone, Copy)]
struct StatusInfo {
    state: TcpState,
    local: Option<SocketAddr>,
    remote: Option<SocketAddr>,
}

impl Default for StatusInfo {
    fn default() -> Self {
        Self { state: TcpState::Other, local: None, remote: None }
    }
}

pub struct TcpStream {
    id: u32,
    data_fd: i32,
    ctl_fd: i32,
    peer_addr: SocketAddr,
    read_timeout: Arc<Mutex<Option<Duration>>>,
    write_timeout: Arc<Mutex<Option<Duration>>>,
    nonblocking: Arc<Mutex<bool>>,
    /// Userspace peek buffer: bytes read from VFS but not yet consumed by `read()`.
    peek_buf: Arc<Mutex<vec::Vec<u8>>>,
}

impl TcpStream {
    pub fn connect<A: ToSocketAddrs>(addr: A) -> crate::io::Result<TcpStream> {
        each_addr(addr, |addr| Self::connect_inner(addr, None))
    }

    pub fn connect_timeout(addr: &SocketAddr, timeout: Duration) -> crate::io::Result<TcpStream> {
        if timeout.is_zero() {
            return Err(io::const_error!(
                crate::io::ErrorKind::InvalidInput,
                "cannot set a 0 duration timeout"
            ));
        }
        Self::connect_inner(addr, Some(timeout))
    }

    fn connect_inner(addr: &SocketAddr, timeout: Option<Duration>) -> crate::io::Result<TcpStream> {
        let SocketAddr::V4(peer_v4) = addr else {
            return Err(crate::io::Error::from_raw_os_error(EAFNOSUPPORT));
        };

        let id = allocate_tcp_socket()?;
        let ctl_path = socket_path(id, "ctl");
        let data_path = socket_path(id, "data");

        let ctl_fd = match vfs_open(&ctl_path, O_WRONLY) {
            Ok(fd) => fd,
            Err(err) => return Err(err),
        };
        let data_fd = match vfs_open(&data_path, O_RDWR | O_NONBLOCK) {
            Ok(fd) => fd,
            Err(err) => {
                let _ = vfs_close(ctl_fd);
                return Err(err);
            }
        };

        let stream = TcpStream {
            id,
            data_fd,
            ctl_fd,
            peer_addr: SocketAddr::V4(*peer_v4),
            read_timeout: Arc::new(Mutex::new(None)),
            write_timeout: Arc::new(Mutex::new(None)),
            nonblocking: Arc::new(Mutex::new(false)),
            peek_buf: Arc::new(Mutex::new(vec::Vec::new())),
        };

        let peer_ip = peer_v4.ip().octets();
        let cmd = format!(
            "connect {}.{}.{}.{} {}",
            peer_ip[0],
            peer_ip[1],
            peer_ip[2],
            peer_ip[3],
            peer_v4.port()
        );
        if let Err(err) = vfs_write(stream.ctl_fd, cmd.as_bytes()) {
            return Err(err);
        }

        if let Err(err) = stream.wait_for_connected(timeout) {
            let _ = vfs_write(stream.ctl_fd, b"close");
            return Err(err);
        }

        Ok(stream)
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> crate::io::Result<()> {
        *self.read_timeout.lock().unwrap() = t;
        Ok(())
    }

    pub fn set_write_timeout(&self, t: Option<Duration>) -> crate::io::Result<()> {
        *self.write_timeout.lock().unwrap() = t;
        Ok(())
    }

    pub fn read_timeout(&self) -> crate::io::Result<Option<Duration>> {
        Ok(*self.read_timeout.lock().unwrap())
    }

    pub fn write_timeout(&self) -> crate::io::Result<Option<Duration>> {
        Ok(*self.write_timeout.lock().unwrap())
    }

    pub fn peek(&self, buf: &mut [u8]) -> crate::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut peek_buf = self.peek_buf.lock().unwrap();
        if peek_buf.is_empty() {
            // Fill the peek buffer with a fresh read from the VFS.
            let mut tmp = vec::Vec::new();
            tmp.resize(buf.len(), 0u8);
            let nonblocking = *self.nonblocking.lock().unwrap();
            let timeout = *self.read_timeout.lock().unwrap();
            let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));
            loop {
                match vfs_read(self.data_fd, &mut tmp) {
                    Ok(0) => return Ok(0),
                    Ok(n) => {
                        peek_buf.extend_from_slice(&tmp[..n]);
                        break;
                    }
                    Err(err) if err.raw_os_error() == Some(EAGAIN) && nonblocking => {
                        return Err(crate::io::Error::from_raw_os_error(EAGAIN));
                    }
                    Err(err) if err.raw_os_error() == Some(EAGAIN) => {
                        if deadline_expired(deadline) {
                            return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                        }
                        sleep_ns(IO_POLL_NS);
                        thread::yield_now();
                    }
                    Err(err) => return Err(err),
                }
            }
        }
        // Copy without consuming — peek does not advance the read position.
        let copy_len = buf.len().min(peek_buf.len());
        buf[..copy_len].copy_from_slice(&peek_buf[..copy_len]);
        Ok(copy_len)
    }

    pub fn read(&self, buf: &mut [u8]) -> crate::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // Drain any bytes staged by a prior peek() call.
        {
            let mut peek_buf = self.peek_buf.lock().unwrap();
            if !peek_buf.is_empty() {
                let copy_len = buf.len().min(peek_buf.len());
                buf[..copy_len].copy_from_slice(&peek_buf[..copy_len]);
                let _ = peek_buf.drain(..copy_len);
                return Ok(copy_len);
            }
        }
        let nonblocking = *self.nonblocking.lock().unwrap();
        let timeout = *self.read_timeout.lock().unwrap();
        let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));

        loop {
            match vfs_read(self.data_fd, buf) {
                Ok(n) => return Ok(n),
                Err(err) if err.raw_os_error() == Some(EAGAIN) && nonblocking => {
                    return Err(crate::io::Error::from_raw_os_error(EAGAIN));
                }
                Err(err) if err.raw_os_error() == Some(EAGAIN) => {
                    if deadline_expired(deadline) {
                        return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> crate::io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> crate::io::Result<usize> {
        if bufs.is_empty() {
            return Ok(0);
        }
        // If there is peeked data staged, drain it into the first non-empty buffer
        // before going to the kernel — preserving correct read ordering.
        {
            let mut peek_buf = self.peek_buf.lock().unwrap();
            if !peek_buf.is_empty() {
                for b in bufs.iter_mut() {
                    if !b.is_empty() {
                        let copy_len = b.len().min(peek_buf.len());
                        b[..copy_len].copy_from_slice(&peek_buf[..copy_len]);
                        let _ = peek_buf.drain(..copy_len);
                        return Ok(copy_len);
                    }
                }
                return Ok(0);
            }
        }
        let iovecs: vec::Vec<KernelIoVec> = bufs
            .iter()
            .map(|b| KernelIoVec { base: b.as_ptr() as usize, len: b.len() })
            .collect();
        let nonblocking = *self.nonblocking.lock().unwrap();
        let timeout = *self.read_timeout.lock().unwrap();
        let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));
        loop {
            let ret = unsafe {
                raw_syscall6(
                    SYS_VFS_READV,
                    self.data_fd as usize,
                    iovecs.as_ptr() as usize,
                    iovecs.len(),
                    0,
                    0,
                    0,
                )
            };
            match decode_ret(ret) {
                Ok(n) => return Ok(n),
                Err(ref err) if err.raw_os_error() == Some(EAGAIN) && nonblocking => {
                    return Err(crate::io::Error::from_raw_os_error(EAGAIN));
                }
                Err(ref err) if err.raw_os_error() == Some(EAGAIN) => {
                    if deadline_expired(deadline) {
                        return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn write(&self, buf: &[u8]) -> crate::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        let nonblocking = *self.nonblocking.lock().unwrap();
        let timeout = *self.write_timeout.lock().unwrap();
        let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));

        loop {
            match vfs_write(self.data_fd, buf) {
                Ok(n) if n > 0 => return Ok(n),
                Ok(_) if nonblocking => return Err(crate::io::Error::from_raw_os_error(EAGAIN)),
                Ok(_) => {
                    if self.status_snapshot().state == TcpState::Closed {
                        return Err(crate::io::Error::from_raw_os_error(EPIPE));
                    }
                    if deadline_expired(deadline) {
                        return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> crate::io::Result<usize> {
        if bufs.is_empty() {
            return Ok(0);
        }
        let iovecs: vec::Vec<KernelIoVec> = bufs
            .iter()
            .map(|b| KernelIoVec { base: b.as_ptr() as usize, len: b.len() })
            .collect();
        let nonblocking = *self.nonblocking.lock().unwrap();
        let timeout = *self.write_timeout.lock().unwrap();
        let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));
        loop {
            let ret = unsafe {
                raw_syscall6(
                    SYS_VFS_WRITEV,
                    self.data_fd as usize,
                    iovecs.as_ptr() as usize,
                    iovecs.len(),
                    0,
                    0,
                    0,
                )
            };
            match decode_ret(ret) {
                Ok(n) if n > 0 => return Ok(n),
                Ok(_) if nonblocking => return Err(crate::io::Error::from_raw_os_error(EAGAIN)),
                Ok(_) => {
                    if self.status_snapshot().state == TcpState::Closed {
                        return Err(crate::io::Error::from_raw_os_error(EPIPE));
                    }
                    if deadline_expired(deadline) {
                        return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn peer_addr(&self) -> crate::io::Result<SocketAddr> {
        Ok(self.peer_addr)
    }

    pub fn socket_addr(&self) -> crate::io::Result<SocketAddr> {
        if let Some(local) = self.status_snapshot().local {
            return Ok(local);
        }
        Err(crate::io::Error::from_raw_os_error(EINVAL))
    }

    pub fn shutdown(&self, how: Shutdown) -> crate::io::Result<()> {
        match how {
            Shutdown::Both => vfs_write(self.ctl_fd, b"close").map(|_| ()),
            Shutdown::Read | Shutdown::Write => Err(crate::io::Error::from_raw_os_error(ENOTSUP)),
        }
    }

    pub fn duplicate(&self) -> crate::io::Result<TcpStream> {
        let new_data_fd = {
            let ret =
                unsafe { raw_syscall6(SYS_FS_DUP, self.data_fd as usize, 0, 0, 0, 0, 0) };
            decode_ret(ret).map(|fd| fd as i32)?
        };
        let new_ctl_fd = {
            let ret =
                unsafe { raw_syscall6(SYS_FS_DUP, self.ctl_fd as usize, 0, 0, 0, 0, 0) };
            match decode_ret(ret).map(|fd| fd as i32) {
                Ok(fd) => fd,
                Err(e) => {
                    let _ = vfs_close(new_data_fd);
                    return Err(e);
                }
            }
        };
        Ok(TcpStream {
            id: self.id,
            data_fd: new_data_fd,
            ctl_fd: new_ctl_fd,
            peer_addr: self.peer_addr,
            read_timeout: Arc::new(Mutex::new(*self.read_timeout.lock().unwrap())),
            write_timeout: Arc::new(Mutex::new(*self.write_timeout.lock().unwrap())),
            nonblocking: Arc::new(Mutex::new(*self.nonblocking.lock().unwrap())),
            peek_buf: Arc::new(Mutex::new(vec::Vec::new())),
        })
    }

    pub fn set_linger(&self, _: Option<Duration>) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn linger(&self) -> crate::io::Result<Option<Duration>> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_nodelay(&self, _: bool) -> crate::io::Result<()> {
        Ok(())
    }

    pub fn nodelay(&self) -> crate::io::Result<bool> {
        Ok(true)
    }

    pub fn set_ttl(&self, _: u32) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn ttl(&self) -> crate::io::Result<u32> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn take_error(&self) -> crate::io::Result<Option<crate::io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> crate::io::Result<()> {
        *self.nonblocking.lock().unwrap() = nonblocking;
        Ok(())
    }

    fn wait_for_connected(&self, timeout: Option<Duration>) -> crate::io::Result<()> {
        let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));
        loop {
            let status = self.status_snapshot();
            match status.state {
                TcpState::Connected => return Ok(()),
                TcpState::Closed => return Err(crate::io::Error::from_raw_os_error(ECONNREFUSED)),
                TcpState::Created | TcpState::Bound | TcpState::Other => {
                    if deadline_expired(deadline) {
                        return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                    }
                    sleep_ns(CONNECT_POLL_NS);
                }
            }
        }
    }

    fn status_snapshot(&self) -> StatusInfo {
        read_status(self.id).unwrap_or_default()
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        let _ = vfs_write(self.ctl_fd, b"close");
        let _ = vfs_close(self.data_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = self.status_snapshot();
        f.debug_struct("TcpStream")
            .field("id", &self.id)
            .field("peer", &status.remote.unwrap_or(self.peer_addr))
            .field("local", &status.local)
            .finish()
    }
}

pub struct TcpListener {
    id: u32,
    ctl_fd: i32,
    accept_fd: i32,
    nonblocking: Arc<Mutex<bool>>,
}

impl TcpListener {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> crate::io::Result<TcpListener> {
        each_addr(addr, |a| Self::bind_inner(a))
    }

    fn bind_inner(addr: &SocketAddr) -> crate::io::Result<TcpListener> {
        let SocketAddr::V4(local_v4) = addr else {
            return Err(crate::io::Error::from_raw_os_error(EAFNOSUPPORT));
        };

        let id = allocate_tcp_socket()?;
        let ctl_path = socket_path(id, "ctl");
        let ctl_fd = vfs_open(&ctl_path, O_WRONLY)?;

        let port = local_v4.port();
        let cmd = format!("listen {port} {LISTEN_BACKLOG}");
        if let Err(e) = vfs_write(ctl_fd, cmd.as_bytes()) {
            let _ = vfs_close(ctl_fd);
            return Err(e);
        }

        let accept_path = socket_path(id, "accept");
        let accept_fd = match vfs_open(&accept_path, O_RDONLY | O_NONBLOCK) {
            Ok(fd) => fd,
            Err(e) => {
                let _ = vfs_write(ctl_fd, b"close");
                let _ = vfs_close(ctl_fd);
                return Err(e);
            }
        };

        Ok(TcpListener { id, ctl_fd, accept_fd, nonblocking: Arc::new(Mutex::new(false)) })
    }

    pub fn socket_addr(&self) -> crate::io::Result<SocketAddr> {
        if let Some(local) = read_status(self.id).ok().and_then(|s| s.local) {
            return Ok(local);
        }
        Err(crate::io::Error::from_raw_os_error(EINVAL))
    }

    pub fn accept(&self) -> crate::io::Result<(TcpStream, SocketAddr)> {
        let nonblocking = *self.nonblocking.lock().unwrap();
        loop {
            let mut buf = [0u8; ACCEPT_RESPONSE_BUF_SIZE];
            match vfs_read(self.accept_fd, &mut buf) {
                Ok(n) if n > 0 => {
                    let text = crate::str::from_utf8(&buf[..n])
                        .map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?
                        .trim();
                    let (conn_id, remote_ip, remote_port) = parse_accept_line(text)?;
                    let peer_addr = SocketAddr::V4(SocketAddrV4::new(
                        Ipv4Addr::new(remote_ip[0], remote_ip[1], remote_ip[2], remote_ip[3]),
                        remote_port,
                    ));
                    let data_path = socket_path(conn_id, "data");
                    let conn_ctl_path = socket_path(conn_id, "ctl");
                    let data_fd = vfs_open(&data_path, O_RDWR | O_NONBLOCK)?;
                    let conn_ctl_fd = match vfs_open(&conn_ctl_path, O_WRONLY) {
                        Ok(fd) => fd,
                        Err(e) => {
                            let _ = vfs_close(data_fd);
                            return Err(e);
                        }
                    };
                    // The fd is always opened O_NONBLOCK so that vfs_read never
                    // blocks the kernel thread.  The stream's `nonblocking` field
                    // is the PAL-level flag that controls whether the read/write
                    // loops spin-wait (false) or return EAGAIN immediately (true).
                    // Newly accepted streams start in blocking (spin-wait) mode,
                    // matching the POSIX default for accepted sockets.
                    let stream = TcpStream {
                        id: conn_id,
                        data_fd,
                        ctl_fd: conn_ctl_fd,
                        peer_addr,
                        read_timeout: Arc::new(Mutex::new(None)),
                        write_timeout: Arc::new(Mutex::new(None)),
                        nonblocking: Arc::new(Mutex::new(false)),
                        peek_buf: Arc::new(Mutex::new(vec::Vec::new())),
                    };
                    return Ok((stream, peer_addr));
                }
                Ok(_) => {
                    if nonblocking {
                        return Err(crate::io::Error::from_raw_os_error(EAGAIN));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(ref e) if e.raw_os_error() == Some(EAGAIN) => {
                    if nonblocking {
                        return Err(crate::io::Error::from_raw_os_error(EAGAIN));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub fn duplicate(&self) -> crate::io::Result<TcpListener> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_ttl(&self, _: u32) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn ttl(&self) -> crate::io::Result<u32> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_only_v6(&self, _: bool) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn only_v6(&self) -> crate::io::Result<bool> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn take_error(&self) -> crate::io::Result<Option<crate::io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> crate::io::Result<()> {
        *self.nonblocking.lock().unwrap() = nonblocking;
        Ok(())
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        let _ = vfs_write(self.ctl_fd, b"close");
        let _ = vfs_close(self.accept_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpListener").field("id", &self.id).finish()
    }
}

pub struct UdpSocket {
    id: u32,
    data_fd: i32,
    ctl_fd: i32,
    local_addr: SocketAddr,
    read_timeout: Arc<Mutex<Option<Duration>>>,
    write_timeout: Arc<Mutex<Option<Duration>>>,
    nonblocking: Arc<Mutex<bool>>,
    connected_remote: Arc<Mutex<Option<SocketAddr>>>,
}

impl UdpSocket {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> crate::io::Result<UdpSocket> {
        each_addr(addr, |a| Self::bind_inner(a))
    }

    fn bind_inner(addr: &SocketAddr) -> crate::io::Result<UdpSocket> {
        let SocketAddr::V4(local_v4) = addr else {
            return Err(crate::io::Error::from_raw_os_error(EAFNOSUPPORT));
        };

        let id = allocate_udp_socket()?;
        let ctl_path = udp_socket_path(id, "ctl");
        let ctl_fd = vfs_open(&ctl_path, O_WRONLY)?;

        let port = local_v4.port();
        let cmd = format!("bind {port}");
        if let Err(e) = vfs_write(ctl_fd, cmd.as_bytes()) {
            let _ = vfs_close(ctl_fd);
            return Err(e);
        }

        let data_path = udp_socket_path(id, "data");
        let data_fd = match vfs_open(&data_path, O_RDWR | O_NONBLOCK) {
            Ok(fd) => fd,
            Err(e) => {
                let _ = vfs_write(ctl_fd, b"close");
                let _ = vfs_close(ctl_fd);
                return Err(e);
            }
        };

        Ok(UdpSocket {
            id,
            data_fd,
            ctl_fd,
            local_addr: SocketAddr::V4(*local_v4),
            read_timeout: Arc::new(Mutex::new(None)),
            write_timeout: Arc::new(Mutex::new(None)),
            nonblocking: Arc::new(Mutex::new(false)),
            connected_remote: Arc::new(Mutex::new(None)),
        })
    }

    pub fn peer_addr(&self) -> crate::io::Result<SocketAddr> {
        self.connected_remote
            .lock()
            .unwrap()
            .ok_or_else(|| crate::io::Error::from_raw_os_error(ENOTCONN))
    }

    pub fn socket_addr(&self) -> crate::io::Result<SocketAddr> {
        Ok(self.local_addr)
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> crate::io::Result<(usize, SocketAddr)> {
        let nonblocking = *self.nonblocking.lock().unwrap();
        let timeout = *self.read_timeout.lock().unwrap();
        let deadline = timeout.map(|dur| monotonic_ns().saturating_add(duration_to_ns(dur)));
        // Wire format on read: [4: src_ipv4][2: src_port_le][4: len_le][payload]
        let header_len: usize = 10;
        let mut raw = vec![0u8; header_len + buf.len()];
        loop {
            match vfs_read(self.data_fd, &mut raw) {
                Ok(n) if n >= header_len => {
                    let src_ip: [u8; 4] = raw[..4].try_into().unwrap();
                    let src_port = u16::from_le_bytes([raw[4], raw[5]]);
                    let payload_len =
                        u32::from_le_bytes([raw[6], raw[7], raw[8], raw[9]]) as usize;
                    let copy_len = payload_len.min(buf.len()).min(n - header_len);
                    buf[..copy_len].copy_from_slice(&raw[header_len..header_len + copy_len]);
                    let src_addr =
                        SocketAddr::new(IpAddr::V4(Ipv4Addr::from(src_ip)), src_port);
                    return Ok((copy_len, src_addr));
                }
                Ok(_) => return Err(crate::io::Error::from_raw_os_error(EINVAL)),
                Err(ref e) if e.raw_os_error() == Some(EAGAIN) && nonblocking => {
                    return Err(crate::io::Error::from_raw_os_error(EAGAIN));
                }
                Err(ref e) if e.raw_os_error() == Some(EAGAIN) => {
                    if deadline_expired(deadline) {
                        return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                    }
                    sleep_ns(IO_POLL_NS);
                    thread::yield_now();
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub fn peek_from(&self, _: &mut [u8]) -> crate::io::Result<(usize, SocketAddr)> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn send_to(&self, data: &[u8], addr: &SocketAddr) -> crate::io::Result<usize> {
        let SocketAddr::V4(dest_v4) = addr else {
            return Err(crate::io::Error::from_raw_os_error(EAFNOSUPPORT));
        };
        let dest_ip = dest_v4.ip().octets();
        let dest_port = dest_v4.port();
        // Wire format on write: [4: dest_ipv4][2: dest_port_le][4: len_le][payload]
        let header_len: usize = 10;
        let mut pkt = vec![0u8; header_len + data.len()];
        pkt[..4].copy_from_slice(&dest_ip);
        pkt[4..6].copy_from_slice(&dest_port.to_le_bytes());
        pkt[6..10].copy_from_slice(&(data.len() as u32).to_le_bytes());
        pkt[10..].copy_from_slice(data);
        vfs_write(self.data_fd, &pkt)?;
        Ok(data.len())
    }

    pub fn duplicate(&self) -> crate::io::Result<UdpSocket> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> crate::io::Result<()> {
        *self.read_timeout.lock().unwrap() = t;
        Ok(())
    }

    pub fn set_write_timeout(&self, t: Option<Duration>) -> crate::io::Result<()> {
        *self.write_timeout.lock().unwrap() = t;
        Ok(())
    }

    pub fn read_timeout(&self) -> crate::io::Result<Option<Duration>> {
        Ok(*self.read_timeout.lock().unwrap())
    }

    pub fn write_timeout(&self) -> crate::io::Result<Option<Duration>> {
        Ok(*self.write_timeout.lock().unwrap())
    }

    pub fn set_broadcast(&self, _: bool) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn broadcast(&self) -> crate::io::Result<bool> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn multicast_loop_v4(&self) -> crate::io::Result<bool> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn multicast_ttl_v4(&self) -> crate::io::Result<u32> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn multicast_loop_v6(&self) -> crate::io::Result<bool> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn set_ttl(&self, _: u32) -> crate::io::Result<()> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn ttl(&self) -> crate::io::Result<u32> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn take_error(&self) -> crate::io::Result<Option<crate::io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> crate::io::Result<()> {
        *self.nonblocking.lock().unwrap() = nonblocking;
        Ok(())
    }

    pub fn recv(&self, buf: &mut [u8]) -> crate::io::Result<usize> {
        if self.connected_remote.lock().unwrap().is_none() {
            return Err(crate::io::Error::from_raw_os_error(ENOTCONN));
        }
        let (n, _) = self.recv_from(buf)?;
        Ok(n)
    }

    pub fn peek(&self, _: &mut [u8]) -> crate::io::Result<usize> {
        Err(crate::io::Error::from_raw_os_error(ENOTSUP))
    }

    pub fn send(&self, data: &[u8]) -> crate::io::Result<usize> {
        let remote = *self.connected_remote.lock().unwrap();
        let Some(addr) = remote else {
            return Err(crate::io::Error::from_raw_os_error(ENOTCONN));
        };
        self.send_to(data, &addr)
    }

    pub fn connect<A: ToSocketAddrs>(&self, addr: A) -> crate::io::Result<()> {
        each_addr(addr, |a| {
            let SocketAddr::V4(v4) = a else {
                return Err(crate::io::Error::from_raw_os_error(EAFNOSUPPORT));
            };
            let ip = v4.ip().octets();
            let port = v4.port();
            let cmd = format!("connect {}.{}.{}.{} {}", ip[0], ip[1], ip[2], ip[3], port);
            vfs_write(self.ctl_fd, cmd.as_bytes())?;
            *self.connected_remote.lock().unwrap() = Some(*a);
            Ok(())
        })
    }
}

impl Drop for UdpSocket {
    fn drop(&mut self) {
        let _ = vfs_write(self.ctl_fd, b"close");
        let _ = vfs_close(self.data_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UdpSocket")
            .field("id", &self.id)
            .field("local", &self.local_addr)
            .field("remote", &*self.connected_remote.lock().unwrap())
            .finish()
    }
}

pub type LookupHost = vec::IntoIter<SocketAddr>;

pub fn lookup_host(host: &str, port: u16) -> crate::io::Result<LookupHost> {
    // Fast path: IPv4 literal.
    if let Ok(ip) = host.parse::<Ipv4Addr>() {
        return Ok(vec![SocketAddr::new(IpAddr::V4(ip), port)].into_iter());
    }

    // `host` is the pure hostname (port is a separate argument), so a colon
    // or leading '[' can only come from an IPv6 literal, which the thingos
    // network stack does not support.
    if host.contains(':') || host.starts_with('[') {
        return Err(crate::io::Error::from_raw_os_error(EAFNOSUPPORT));
    }

    // Resolve hostname via netd's /net/dns/lookup VFS interface.
    // Write the hostname, then poll-read until we get a dotted-decimal reply.
    let lookup_fd = vfs_open("/net/dns/lookup", O_RDWR)?;

    if let Err(e) = vfs_write(lookup_fd, host.as_bytes()) {
        let _ = vfs_close(lookup_fd);
        return Err(e);
    }

    // 5-second hard deadline for DNS resolution.
    const DNS_TIMEOUT_NS: u64 = 5_000_000_000;
    let deadline = monotonic_ns().saturating_add(DNS_TIMEOUT_NS);

    // Buffer large enough for a dotted-decimal IPv4 address (max 15 chars,
    // e.g. "255.255.255.255") plus a possible trailing newline.
    const DNS_REPLY_BUF: usize = 32;
    let mut buf = [0u8; DNS_REPLY_BUF];
    loop {
        match vfs_read(lookup_fd, &mut buf) {
            Ok(n) if n > 0 => {
                let _ = vfs_close(lookup_fd);
                let text = crate::str::from_utf8(&buf[..n])
                    .map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?
                    .trim();
                let ip: Ipv4Addr =
                    text.parse().map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?;
                return Ok(vec![SocketAddr::new(IpAddr::V4(ip), port)].into_iter());
            }
            // Ok(0) means the result is not ready yet (netd is still resolving);
            // treat it the same as EAGAIN and retry after a short sleep.
            Ok(_) => {
                if monotonic_ns() >= deadline {
                    let _ = vfs_close(lookup_fd);
                    return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                }
                sleep_ns(CONNECT_POLL_NS);
            }
            Err(ref e) if e.raw_os_error() == Some(EAGAIN) => {
                if monotonic_ns() >= deadline {
                    let _ = vfs_close(lookup_fd);
                    return Err(crate::io::Error::from_raw_os_error(ETIMEDOUT));
                }
                sleep_ns(CONNECT_POLL_NS);
            }
            Err(e) => {
                let _ = vfs_close(lookup_fd);
                return Err(e);
            }
        }
    }
}

fn allocate_tcp_socket() -> crate::io::Result<u32> {
    let fd = vfs_open("/net/tcp/new", O_RDONLY)?;
    let mut buf = [0u8; 32];
    let n = vfs_read(fd, &mut buf)?;
    vfs_close(fd)?;
    let text =
        crate::str::from_utf8(&buf[..n]).map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?.trim();
    text.parse::<u32>().map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))
}

fn socket_path(id: u32, subpath: &str) -> String {
    format!("/net/tcp/{id}/{subpath}")
}

fn allocate_udp_socket() -> crate::io::Result<u32> {
    let fd = vfs_open("/net/udp/new", O_RDONLY)?;
    let mut buf = [0u8; 32];
    let n = vfs_read(fd, &mut buf)?;
    vfs_close(fd)?;
    let text =
        crate::str::from_utf8(&buf[..n]).map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?.trim();
    text.parse::<u32>().map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))
}

fn udp_socket_path(id: u32, subpath: &str) -> String {
    format!("/net/udp/{id}/{subpath}")
}

fn read_status(id: u32) -> crate::io::Result<StatusInfo> {
    let path = socket_path(id, "status");
    let fd = vfs_open(&path, O_RDONLY)?;
    let mut buf = [0u8; 256];
    let n = vfs_read(fd, &mut buf)?;
    vfs_close(fd)?;

    let text =
        crate::str::from_utf8(&buf[..n]).map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?;

    let mut out = StatusInfo::default();
    for line in text.lines() {
        if let Some(state) = line.strip_prefix("state: ") {
            out.state = match state.trim() {
                "created" => TcpState::Created,
                "bound" | "syn-sent" | "syn-received" => TcpState::Bound,
                "connected" | "established" | "fin-wait-1" | "fin-wait-2" | "close-wait" => {
                    TcpState::Connected
                }
                "closed" | "time-wait" | "closing" | "last-ack" => TcpState::Closed,
                _ => TcpState::Other,
            };
        } else if let Some(local) = line.strip_prefix("local: ") {
            out.local = parse_socket_addr(local.trim());
        } else if let Some(remote) = line.strip_prefix("remote: ") {
            out.remote = parse_socket_addr(remote.trim());
        }
    }
    Ok(out)
}

fn parse_socket_addr(text: &str) -> Option<SocketAddr> {
    let (ip, port) = text.rsplit_once(':')?;
    let port = port.parse::<u16>().ok()?;
    let ip = ip.parse::<Ipv4Addr>().ok()?;
    Some(SocketAddr::V4(SocketAddrV4::new(ip, port)))
}

/// Parse an accept-response line: `"<conn_id> <a>.<b>.<c>.<d> <port>"`.
fn parse_accept_line(text: &str) -> crate::io::Result<(u32, [u8; 4], u16)> {
    let mut parts = text.split_whitespace();
    let conn_id: u32 = parts
        .next()
        .ok_or_else(|| crate::io::Error::from_raw_os_error(EINVAL))?
        .parse()
        .map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?;
    let ip_str = parts.next().ok_or_else(|| crate::io::Error::from_raw_os_error(EINVAL))?;
    let port: u16 = parts
        .next()
        .ok_or_else(|| crate::io::Error::from_raw_os_error(EINVAL))?
        .parse()
        .map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?;
    let ip: Ipv4Addr =
        ip_str.parse().map_err(|_| crate::io::Error::from_raw_os_error(EINVAL))?;
    Ok((conn_id, ip.octets(), port))
}

fn deadline_expired(deadline: Option<u64>) -> bool {
    match deadline {
        Some(ns) => monotonic_ns() >= ns,
        None => false,
    }
}

fn duration_to_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn monotonic_ns() -> u64 {
    let ret = unsafe { raw_syscall6(SYS_TIME_MONOTONIC, 0, 0, 0, 0, 0, 0) };
    if ret < 0 { 0 } else { ret as u64 }
}

fn sleep_ns(ns: u64) {
    let _ = unsafe { raw_syscall6(SYS_SLEEP_NS, ns as usize, 0, 0, 0, 0, 0) };
}

fn vfs_open(path: &str, flags: u32) -> crate::io::Result<i32> {
    let ret = unsafe {
        raw_syscall6(SYS_VFS_OPEN, path.as_ptr() as usize, path.len(), flags as usize, 0, 0, 0)
    };
    decode_ret(ret).map(|fd| fd as i32)
}

fn vfs_close(fd: i32) -> crate::io::Result<()> {
    let ret = unsafe { raw_syscall6(SYS_VFS_CLOSE, fd as usize, 0, 0, 0, 0, 0) };
    decode_ret(ret).map(|_| ())
}

fn vfs_read(fd: i32, buf: &mut [u8]) -> crate::io::Result<usize> {
    let ret = unsafe {
        raw_syscall6(SYS_VFS_READ, fd as usize, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0)
    };
    decode_ret(ret)
}

fn vfs_write(fd: i32, buf: &[u8]) -> crate::io::Result<usize> {
    let ret = unsafe {
        raw_syscall6(SYS_VFS_WRITE, fd as usize, buf.as_ptr() as usize, buf.len(), 0, 0, 0)
    };
    decode_ret(ret)
}

fn decode_ret(ret: isize) -> crate::io::Result<usize> {
    if ret < 0 && ret >= -4096 {
        Err(crate::io::Error::from_raw_os_error((-ret) as i32))
    } else {
        Ok(ret as usize)
    }
}
