use core::sync::atomic::{Atomic, AtomicBool, AtomicU32, AtomicUsize, Ordering};

use super::*;
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4, SocketAddrV6};
use crate::os::xous::services;
use crate::sync::Arc;
use crate::time::Duration;

macro_rules! unimpl {
    () => {
        return Err(io::const_error!(
            io::ErrorKind::Unsupported,
            "this function is not yet implemented",
        ));
    };
}

enum ReadOrPeek {
    Read,
    Peek,
}

#[derive(Clone)]
pub struct TcpStream {
    fd: u16,
    local_port: u16,
    remote_port: u16,
    peer_addr: SocketAddr,
    // milliseconds
    read_timeout: Arc<Atomic<u32>>,
    // milliseconds
    write_timeout: Arc<Atomic<u32>>,
    handle_count: Arc<Atomic<usize>>,
    nonblocking: Arc<Atomic<bool>>,
}

fn sockaddr_to_buf(duration: Duration, addr: &SocketAddr, buf: &mut [u8]) {
    // Construct the request.
    let port_bytes = addr.port().to_le_bytes();
    buf[0] = port_bytes[0];
    buf[1] = port_bytes[1];
    for (dest, src) in buf[2..].iter_mut().zip((duration.as_millis() as u64).to_le_bytes()) {
        *dest = src;
    }
    match addr.ip() {
        IpAddr::V4(addr) => {
            buf[10] = 4;
            for (dest, src) in buf[11..].iter_mut().zip(addr.octets()) {
                *dest = src;
            }
        }
        IpAddr::V6(addr) => {
            buf[10] = 6;
            for (dest, src) in buf[11..].iter_mut().zip(addr.octets()) {
                *dest = src;
            }
        }
    }
}

impl TcpStream {
    pub(crate) fn from_listener(
        fd: u16,
        local_port: u16,
        remote_port: u16,
        peer_addr: SocketAddr,
    ) -> TcpStream {
        TcpStream {
            fd,
            local_port,
            remote_port,
            peer_addr,
            read_timeout: Arc::new(AtomicU32::new(0)),
            write_timeout: Arc::new(AtomicU32::new(0)),
            handle_count: Arc::new(AtomicUsize::new(1)),
            nonblocking: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn connect(socketaddr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        Self::connect_timeout(socketaddr?, Duration::ZERO)
    }

    pub fn connect_timeout(addr: &SocketAddr, duration: Duration) -> io::Result<TcpStream> {
        let mut connect_request = ConnectRequest { raw: [0u8; 4096] };

        // Construct the request.
        sockaddr_to_buf(duration, &addr, &mut connect_request.raw);

        let Ok((_, valid)) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdTcpConnect.into(),
            &mut connect_request.raw,
            0,
            4096,
        ) else {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, "invalid response"));
        };

        // The first four bytes should be zero upon success, and will be nonzero
        // for an error.
        let response = connect_request.raw;
        if response[0] != 0 || valid == 0 {
            // errcode is a u8 but stuck in a u16 where the upper byte is invalid. Mask & decode accordingly.
            let errcode = response[0];
            if errcode == NetError::SocketInUse as u8 {
                return Err(io::const_error!(io::ErrorKind::ResourceBusy, "socket in use"));
            } else if errcode == NetError::Unaddressable as u8 {
                return Err(io::const_error!(io::ErrorKind::AddrNotAvailable, "invalid address"));
            } else {
                return Err(io::const_error!(
                    io::ErrorKind::InvalidInput,
                    "unable to connect or internal error",
                ));
            }
        }
        let fd = u16::from_le_bytes([response[2], response[3]]);
        let local_port = u16::from_le_bytes([response[4], response[5]]);
        let remote_port = u16::from_le_bytes([response[6], response[7]]);
        // println!(
        //     "Connected with local port of {}, remote port of {}, file handle of {}",
        //     local_port, remote_port, fd
        // );
        Ok(TcpStream {
            fd,
            local_port,
            remote_port,
            peer_addr: *addr,
            read_timeout: Arc::new(AtomicU32::new(0)),
            write_timeout: Arc::new(AtomicU32::new(0)),
            handle_count: Arc::new(AtomicUsize::new(1)),
            nonblocking: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        if let Some(to) = timeout {
            if to.is_zero() {
                return Err(io::Error::ZERO_TIMEOUT);
            }
        }
        self.read_timeout.store(
            timeout.map(|t| t.as_millis().min(u32::MAX as u128) as u32).unwrap_or_default(),
            Ordering::Relaxed,
        );
        Ok(())
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        if let Some(to) = timeout {
            if to.is_zero() {
                return Err(io::Error::ZERO_TIMEOUT);
            }
        }
        self.write_timeout.store(
            timeout.map(|t| t.as_millis().min(u32::MAX as u128) as u32).unwrap_or_default(),
            Ordering::Relaxed,
        );
        Ok(())
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        match self.read_timeout.load(Ordering::Relaxed) {
            0 => Ok(None),
            t => Ok(Some(Duration::from_millis(t as u64))),
        }
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        match self.write_timeout.load(Ordering::Relaxed) {
            0 => Ok(None),
            t => Ok(Some(Duration::from_millis(t as u64))),
        }
    }

    fn read_or_peek(&self, buf: &mut [u8], op: ReadOrPeek) -> io::Result<usize> {
        let mut receive_request = ReceiveData { raw: [0u8; 4096] };
        let data_to_read = buf.len().min(receive_request.raw.len());

        let opcode = match op {
            ReadOrPeek::Read => {
                services::NetLendMut::StdTcpRx(self.fd, self.nonblocking.load(Ordering::Relaxed))
            }
            ReadOrPeek::Peek => {
                services::NetLendMut::StdTcpPeek(self.fd, self.nonblocking.load(Ordering::Relaxed))
            }
        };

        let Ok((offset, length)) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            opcode.into(),
            &mut receive_request.raw,
            // Reuse the `offset` as the read timeout
            self.read_timeout.load(Ordering::Relaxed) as usize,
            data_to_read,
        ) else {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "library failure: wrong message type or messaging error",
            ));
        };

        if offset != 0 {
            for (dest, src) in buf.iter_mut().zip(receive_request.raw[..length].iter()) {
                *dest = *src;
            }
            Ok(length)
        } else {
            let result = receive_request.raw;
            if result[0] != 0 {
                if result[1] == 8 {
                    // timed out
                    return Err(io::const_error!(io::ErrorKind::TimedOut, "timeout"));
                }
                if result[1] == 9 {
                    // would block
                    return Err(io::const_error!(io::ErrorKind::WouldBlock, "would block"));
                }
            }
            Err(io::const_error!(io::ErrorKind::Other, "recv_slice failure"))
        }
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_or_peek(buf, ReadOrPeek::Peek)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_or_peek(buf, ReadOrPeek::Read)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|b| self.read(b), bufs)
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut send_request = SendData { raw: [0u8; 4096] };
        for (dest, src) in send_request.raw.iter_mut().zip(buf) {
            *dest = *src;
        }
        let buf_len = send_request.raw.len().min(buf.len());

        let (_offset, _valid) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdTcpTx(self.fd).into(),
            &mut send_request.raw,
            // Reuse the offset as the timeout
            self.write_timeout.load(Ordering::Relaxed) as usize,
            buf_len,
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "internal error")))?;

        if send_request.raw[0] != 0 {
            if send_request.raw[4] == 8 {
                // timed out
                return Err(io::const_error!(
                    io::ErrorKind::BrokenPipe,
                    "timeout or connection closed",
                ));
            } else if send_request.raw[4] == 9 {
                // would block
                return Err(io::const_error!(io::ErrorKind::WouldBlock, "would block"));
            } else {
                return Err(io::const_error!(io::ErrorKind::InvalidInput, "error when sending"));
            }
        }
        Ok(u32::from_le_bytes([
            send_request.raw[4],
            send_request.raw[5],
            send_request.raw[6],
            send_request.raw[7],
        ]) as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.peer_addr)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        let mut get_addr = GetAddress { raw: [0u8; 4096] };

        let Ok((_offset, _valid)) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdGetAddress(self.fd).into(),
            &mut get_addr.raw,
            0,
            0,
        ) else {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, "internal error"));
        };
        let mut i = get_addr.raw.iter();
        match *i.next().unwrap() {
            4 => Ok(SocketAddr::V4(SocketAddrV4::new(
                Ipv4Addr::new(
                    *i.next().unwrap(),
                    *i.next().unwrap(),
                    *i.next().unwrap(),
                    *i.next().unwrap(),
                ),
                self.local_port,
            ))),
            6 => {
                let mut new_addr = [0u8; 16];
                for (src, octet) in i.zip(new_addr.iter_mut()) {
                    *octet = *src;
                }
                Ok(SocketAddr::V6(SocketAddrV6::new(new_addr.into(), self.local_port, 0, 0)))
            }
            _ => Err(io::const_error!(io::ErrorKind::InvalidInput, "internal error")),
        }
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdTcpStreamShutdown(self.fd, how).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|_| ())
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        self.handle_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.clone())
    }

    pub fn set_linger(&self, _: Option<Duration>) -> io::Result<()> {
        unimpl!();
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        unimpl!();
    }

    pub fn set_nodelay(&self, enabled: bool) -> io::Result<()> {
        crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdSetNodelay(self.fd, enabled).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|_| ())
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        Ok(crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdGetNodelay(self.fd).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|res| res[0] != 0)?)
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        if ttl > 255 {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, "TTL must be less than 256"));
        }
        crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdSetTtlTcp(self.fd, ttl).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|_| ())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Ok(crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdGetTtlTcp(self.fd).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|res| res[0] as _)?)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        // this call doesn't have a meaning on our platform, but we can at least not panic if it's used.
        Ok(None)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.nonblocking.store(nonblocking, Ordering::Relaxed);
        Ok(())
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TCP connection to {:?} port {} to local port {}",
            self.peer_addr, self.remote_port, self.local_port
        )
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        if self.handle_count.fetch_sub(1, Ordering::Relaxed) == 1 {
            // only drop if we're the last clone
            crate::os::xous::ffi::blocking_scalar(
                services::net_server(),
                services::NetBlockingScalar::StdTcpClose(self.fd).into(),
            )
            .unwrap();
        }
    }
}
