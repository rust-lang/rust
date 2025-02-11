use core::convert::TryInto;
use core::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};

use super::*;
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use crate::os::xous::services;
use crate::sync::Arc;
use crate::{fmt, io};

macro_rules! unimpl {
    () => {
        return Err(io::const_error!(
            io::ErrorKind::Unsupported,
            &"This function is not yet implemented",
        ));
    };
}

#[derive(Clone)]
pub struct TcpListener {
    fd: Arc<AtomicU16>,
    local: SocketAddr,
    handle_count: Arc<AtomicUsize>,
    nonblocking: Arc<AtomicBool>,
}

impl TcpListener {
    pub fn bind(socketaddr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        let mut addr = *socketaddr?;

        let fd = TcpListener::bind_inner(&mut addr)?;
        return Ok(TcpListener {
            fd: Arc::new(AtomicU16::new(fd)),
            local: addr,
            handle_count: Arc::new(AtomicUsize::new(1)),
            nonblocking: Arc::new(AtomicBool::new(false)),
        });
    }

    /// This returns the raw fd of a Listener, so that it can also be used by the
    /// accept routine to replenish the Listener object after its handle has been converted into
    /// a TcpStream object.
    fn bind_inner(addr: &mut SocketAddr) -> io::Result<u16> {
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

        let Ok((_, valid)) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdTcpListen.into(),
            &mut connect_request.raw,
            0,
            4096,
        ) else {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, &"Invalid response"));
        };

        // The first four bytes should be zero upon success, and will be nonzero
        // for an error.
        let response = connect_request.raw;
        if response[0] != 0 || valid == 0 {
            let errcode = response[1];
            if errcode == NetError::SocketInUse as u8 {
                return Err(io::const_error!(io::ErrorKind::ResourceBusy, &"Socket in use"));
            } else if errcode == NetError::Invalid as u8 {
                return Err(io::const_error!(io::ErrorKind::AddrNotAvailable, &"Invalid address"));
            } else if errcode == NetError::LibraryError as u8 {
                return Err(io::const_error!(io::ErrorKind::Other, &"Library error"));
            } else {
                return Err(io::const_error!(
                    io::ErrorKind::Other,
                    &"Unable to connect or internal error"
                ));
            }
        }
        let fd = response[1] as usize;
        if addr.port() == 0 {
            // oddly enough, this is a valid port and it means "give me something valid, up to you what that is"
            let assigned_port = u16::from_le_bytes(response[2..4].try_into().unwrap());
            addr.set_port(assigned_port);
        }
        // println!("TcpListening with file handle of {}\r\n", fd);
        Ok(fd.try_into().unwrap())
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let mut receive_request = ReceiveData { raw: [0u8; 4096] };

        if self.nonblocking.load(Ordering::Relaxed) {
            // nonblocking
            receive_request.raw[0] = 0;
        } else {
            // blocking
            receive_request.raw[0] = 1;
        }

        if let Ok((_offset, _valid)) = crate::os::xous::ffi::lend_mut(
            services::net_server(),
            services::NetLendMut::StdTcpAccept(self.fd.load(Ordering::Relaxed)).into(),
            &mut receive_request.raw,
            0,
            0,
        ) {
            if receive_request.raw[0] != 0 {
                // error case
                if receive_request.raw[1] == NetError::TimedOut as u8 {
                    return Err(io::const_error!(io::ErrorKind::TimedOut, &"accept timed out",));
                } else if receive_request.raw[1] == NetError::WouldBlock as u8 {
                    return Err(
                        io::const_error!(io::ErrorKind::WouldBlock, &"accept would block",),
                    );
                } else if receive_request.raw[1] == NetError::LibraryError as u8 {
                    return Err(io::const_error!(io::ErrorKind::Other, &"Library error"));
                } else {
                    return Err(io::const_error!(io::ErrorKind::Other, &"library error",));
                }
            } else {
                // accept successful
                let rr = &receive_request.raw;
                let stream_fd = u16::from_le_bytes(rr[1..3].try_into().unwrap());
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
                    return Err(io::const_error!(io::ErrorKind::Other, &"library error",));
                };

                // replenish the listener
                let mut local_copy = self.local.clone(); // port is non-0 by this time, but the method signature needs a mut
                let new_fd = TcpListener::bind_inner(&mut local_copy)?;
                self.fd.store(new_fd, Ordering::Relaxed);

                // now return a stream converted from the old stream's fd
                Ok((TcpStream::from_listener(stream_fd, self.local.port(), port, addr), addr))
            }
        } else {
            Err(io::const_error!(io::ErrorKind::InvalidInput, &"Unable to accept"))
        }
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        self.handle_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.clone())
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        if ttl > 255 {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, "TTL must be less than 256"));
        }
        crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdSetTtlTcp(self.fd.load(Ordering::Relaxed), ttl).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, &"Unexpected return value")))
        .map(|_| ())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Ok(crate::os::xous::ffi::blocking_scalar(
            services::net_server(),
            services::NetBlockingScalar::StdGetTtlTcp(self.fd.load(Ordering::Relaxed)).into(),
        )
        .or(Err(io::const_error!(io::ErrorKind::InvalidInput, &"Unexpected return value")))
        .map(|res| res[0] as _)?)
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        unimpl!();
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        unimpl!();
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

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TCP listening on {:?}", self.local)
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        if self.handle_count.fetch_sub(1, Ordering::Relaxed) == 1 {
            // only drop if we're the last clone
            crate::os::xous::ffi::blocking_scalar(
                services::net_server(),
                crate::os::xous::services::NetBlockingScalar::StdTcpClose(
                    self.fd.load(Ordering::Relaxed),
                )
                .into(),
            )
            .unwrap();
        }
    }
}
