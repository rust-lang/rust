use core::convert::TryInto;
use core::sync::atomic::{Atomic, AtomicUsize, Ordering};

use super::*;
use crate::cell::Cell;
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use crate::os::xous::services;
use crate::sync::Arc;
use crate::time::Duration;
use crate::{fmt, io};

macro_rules! unimpl {
    () => {
        return Err(io::const_error!(
            io::ErrorKind::Unsupported,
            "this function is not yet implemented",
        ));
    };
}

#[derive(Clone)]
pub struct UdpSocket {
    fd: u16,
    local: SocketAddr,
    remote: Cell<Option<SocketAddr>>,
    // in milliseconds. The setting applies only to `recv` calls after the timeout is set.
    read_timeout: Cell<u64>,
    // in milliseconds. The setting applies only to `send` calls after the timeout is set.
    write_timeout: Cell<u64>,
    handle_count: Arc<Atomic<usize>>,
    nonblocking: Cell<bool>,
}

impl UdpSocket {
    pub fn bind(socketaddr: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        let addr = socketaddr?;
        // Construct the request
        let mut connect_request = ConnectRequest { raw: [0u8; 4096] };

        // Serialize the StdUdpBind structure. This is done "manually" because we don't want to
        // make an auto-serdes (like bincode or rkyv) crate a dependency of Xous.
        let port_bytes = addr.port().to_le_bytes();
        connect_request.raw[0] = port_bytes[0];
        connect_request.raw[1] = port_bytes[1];
        match addr.ip() {
            IpAddr::V4(addr) => {
                connect_request.raw[2] = 4;
                for (dest, src) in connect_request.raw[3..].iter_mut().zip(addr.octets()) {
                    *dest = src;
                }
            }
            IpAddr::V6(addr) => {
                connect_request.raw[2] = 6;
                for (dest, src) in connect_request.raw[3..].iter_mut().zip(addr.octets()) {
                    *dest = src;
                }
            }
        }

        let response = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdUdpBind.into(),
            &mut connect_request.raw,
            0,
            4096,
        );

        if let Ok((_, valid)) = response {
            // The first four bytes should be zero upon success, and will be nonzero
            // for an error.
            let response = connect_request.raw;
            if response[0] != 0 || valid == 0 {
                let errcode = response[1];
                if errcode == NetError::SocketInUse as u8 {
                    return Err(io::const_error!(io::ErrorKind::ResourceBusy, "socket in use"));
                } else if errcode == NetError::Invalid as u8 {
                    return Err(io::const_error!(
                        io::ErrorKind::InvalidInput,
                        "port can't be 0 or invalid address",
                    ));
                } else if errcode == NetError::LibraryError as u8 {
                    return Err(io::const_error!(io::ErrorKind::Other, "library error"));
                } else {
                    return Err(io::const_error!(
                        io::ErrorKind::Other,
                        "unable to connect or internal error",
                    ));
                }
            }
            let fd = response[1] as u16;
            return Ok(UdpSocket {
                fd,
                local: *addr,
                remote: Cell::new(None),
                read_timeout: Cell::new(0),
                write_timeout: Cell::new(0),
                handle_count: Arc::new(AtomicUsize::new(1)),
                nonblocking: Cell::new(false),
            });
        }
        Err(io::const_error!(io::ErrorKind::InvalidInput, "invalid response"))
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        match self.remote.get() {
            Some(dest) => Ok(dest),
            None => Err(io::const_error!(io::ErrorKind::NotConnected, "no peer specified")),
        }
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }

    fn recv_inner(&self, buf: &mut [u8], do_peek: bool) -> io::Result<(usize, SocketAddr)> {
        let mut receive_request = ReceiveData { raw: [0u8; 4096] };

        if self.nonblocking.get() {
            // nonblocking
            receive_request.raw[0] = 0;
        } else {
            // blocking
            receive_request.raw[0] = 1;
            for (&s, d) in self
                .read_timeout
                .get()
                .to_le_bytes()
                .iter()
                .zip(receive_request.raw[1..9].iter_mut())
            {
                *d = s;
            }
        }
        if let Ok((_offset, _valid)) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdUdpRx(self.fd).into(),
            &mut receive_request.raw,
            if do_peek { 1 } else { 0 },
            0,
        ) {
            if receive_request.raw[0] != 0 {
                // error case
                if receive_request.raw[1] == NetError::TimedOut as u8 {
                    return Err(io::const_error!(io::ErrorKind::TimedOut, "recv timed out"));
                } else if receive_request.raw[1] == NetError::WouldBlock as u8 {
                    return Err(io::const_error!(io::ErrorKind::WouldBlock, "recv would block"));
                } else if receive_request.raw[1] == NetError::LibraryError as u8 {
                    return Err(io::const_error!(io::ErrorKind::Other, "library error"));
                } else {
                    return Err(io::const_error!(io::ErrorKind::Other, "library error"));
                }
            } else {
                let rr = &receive_request.raw;
                let rxlen = u16::from_le_bytes(rr[1..3].try_into().unwrap());
                let port = u16::from_le_bytes(rr[20..22].try_into().unwrap());
                let addr = if rr[3] == 4 {
                    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(rr[4], rr[5], rr[6], rr[7])), port)
                } else if rr[3] == 6 {
                    SocketAddr::new(
                        IpAddr::V6(Ipv6Addr::new(
                            u16::from_be_bytes(rr[4..6].try_into().unwrap()),
                            u16::from_be_bytes(rr[6..8].try_into().unwrap()),
                            u16::from_be_bytes(rr[8..10].try_into().unwrap()),
                            u16::from_be_bytes(rr[10..12].try_into().unwrap()),
                            u16::from_be_bytes(rr[12..14].try_into().unwrap()),
                            u16::from_be_bytes(rr[14..16].try_into().unwrap()),
                            u16::from_be_bytes(rr[16..18].try_into().unwrap()),
                            u16::from_be_bytes(rr[18..20].try_into().unwrap()),
                        )),
                        port,
                    )
                } else {
                    return Err(io::const_error!(io::ErrorKind::Other, "library error"));
                };
                for (&s, d) in rr[22..22 + rxlen as usize].iter().zip(buf.iter_mut()) {
                    *d = s;
                }
                Ok((rxlen as usize, addr))
            }
        } else {
            Err(io::const_error!(io::ErrorKind::InvalidInput, "unable to recv"))
        }
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_inner(buf, false)
    }

    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_from(buf).map(|(len, _addr)| len)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_inner(buf, true)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.peek_from(buf).map(|(len, _addr)| len)
    }

    pub fn connect(&self, maybe_addr: io::Result<&SocketAddr>) -> io::Result<()> {
        let addr = maybe_addr?;
        self.remote.set(Some(*addr));
        Ok(())
    }

    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        if let Some(addr) = self.remote.get() {
            self.send_to(buf, &addr)
        } else {
            Err(io::const_error!(io::ErrorKind::NotConnected, "No remote specified"))
        }
    }

    pub fn send_to(&self, buf: &[u8], addr: &SocketAddr) -> io::Result<usize> {
        let mut tx_req = SendData { raw: [0u8; 4096] };

        // Construct the request.
        let port_bytes = addr.port().to_le_bytes();
        tx_req.raw[0] = port_bytes[0];
        tx_req.raw[1] = port_bytes[1];
        match addr.ip() {
            IpAddr::V4(addr) => {
                tx_req.raw[2] = 4;
                for (dest, src) in tx_req.raw[3..].iter_mut().zip(addr.octets()) {
                    *dest = src;
                }
            }
            IpAddr::V6(addr) => {
                tx_req.raw[2] = 6;
                for (dest, src) in tx_req.raw[3..].iter_mut().zip(addr.octets()) {
                    *dest = src;
                }
            }
        }
        let len = buf.len() as u16;
        let len_bytes = len.to_le_bytes();
        tx_req.raw[19] = len_bytes[0];
        tx_req.raw[20] = len_bytes[1];
        for (&s, d) in buf.iter().zip(tx_req.raw[21..].iter_mut()) {
            *d = s;
        }

        // let buf = unsafe {
        //     xous::MemoryRange::new(
        //         &mut tx_req as *mut SendData as usize,
        //         size_of::<SendData>(),
        //     )
        //     .unwrap()
        // };

        // write time-outs are implemented on the caller side. Basically, if the Net crate server
        // is too busy to take the call immediately: retry, until the timeout is reached.
        let now = crate::time::Instant::now();
        let write_timeout = if self.nonblocking.get() {
            // nonblocking
            core::time::Duration::ZERO
        } else {
            // blocking
            if self.write_timeout.get() == 0 {
                // forever
                core::time::Duration::from_millis(u64::MAX)
            } else {
                // or this amount of time
                core::time::Duration::from_millis(self.write_timeout.get())
            }
        };
        loop {
            let response = crate::os::xous::ffi::try_lend_mut(
                services::net_server(),
                services::NetLendMut::StdUdpTx(self.fd).into(),
                &mut tx_req.raw,
                0,
                4096,
            );
            match response {
                Ok((_, valid)) => {
                    let response = &tx_req.raw;
                    if response[0] != 0 || valid == 0 {
                        let errcode = response[1];
                        if errcode == NetError::SocketInUse as u8 {
                            return Err(io::const_error!(
                                io::ErrorKind::ResourceBusy,
                                "socket in use",
                            ));
                        } else if errcode == NetError::Invalid as u8 {
                            return Err(io::const_error!(
                                io::ErrorKind::InvalidInput,
                                "socket not valid",
                            ));
                        } else if errcode == NetError::LibraryError as u8 {
                            return Err(io::const_error!(io::ErrorKind::Other, "library error"));
                        } else {
                            return Err(io::const_error!(
                                io::ErrorKind::Other,
                                "unable to connect",
                            ));
                        }
                    } else {
                        // no error
                        return Ok(len as usize);
                    }
                }
                Err(crate::os::xous::ffi::Error::ServerQueueFull) => {
                    if now.elapsed() >= write_timeout {
                        return Err(io::const_error!(io::ErrorKind::WouldBlock, "write timed out"));
                    } else {
                        // question: do we want to do something a bit more gentle than immediately retrying?
                        crate::thread::yield_now();
                    }
                }
                _ => return Err(io::const_error!(io::ErrorKind::Other, "library error")),
            }
        }
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.handle_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.clone())
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        if let Some(d) = timeout {
            if d.is_zero() {
                return Err(io::Error::ZERO_TIMEOUT);
            }
        }
        self.read_timeout
            .set(timeout.map(|t| t.as_millis().min(u64::MAX as u128) as u64).unwrap_or_default());
        Ok(())
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        if let Some(d) = timeout {
            if d.is_zero() {
                return Err(io::Error::ZERO_TIMEOUT);
            }
        }
        self.write_timeout
            .set(timeout.map(|t| t.as_millis().min(u64::MAX as u128) as u64).unwrap_or_default());
        Ok(())
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        match self.read_timeout.get() {
            0 => Ok(None),
            t => Ok(Some(Duration::from_millis(t as u64))),
        }
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        match self.write_timeout.get() {
            0 => Ok(None),
            t => Ok(Some(Duration::from_millis(t as u64))),
        }
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        if ttl > 255 {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, "TTL must be less than 256"));
        }
        crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdSetTtlUdp(self.fd, ttl).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|_| ())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Ok(crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdGetTtlUdp(self.fd).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, "unexpected return value")))
        .map(|res| res[0] as _)?)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        // this call doesn't have a meaning on our platform, but we can at least not panic if it's used.
        Ok(None)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.nonblocking.set(nonblocking);
        Ok(())
    }

    // ------------- smoltcp base stack does not have multicast or broadcast support ---------------
    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        unimpl!();
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        unimpl!();
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        unimpl!();
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        unimpl!();
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        unimpl!();
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        unimpl!();
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        unimpl!();
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        unimpl!();
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        unimpl!();
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        unimpl!();
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        unimpl!();
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        unimpl!();
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UDP listening on {:?} to {:?}", self.local, self.remote.get())
    }
}

impl Drop for UdpSocket {
    fn drop(&mut self) {
        if self.handle_count.fetch_sub(1, Ordering::Relaxed) == 1 {
            // only drop if we're the last clone
            crate::os::xous::ffi::blocking_scalar(
                services::net_server(),
                services::NetBlockingScalar::StdUdpClose(self.fd).into(),
            )
            .unwrap();
        }
    }
}
