//! Leaving Most of this unimplemented since TCP was mostly implemented to get testing working.
//! In the future, should probably desing networking around SIMPLE_NETWOR_PROTOCOL

use super::{tcp4, uefi_service_binding};
use crate::cell::RefCell;
use crate::sys::uefi::common;
use crate::time::Duration;
use crate::{
    io::{self, IoSlice, IoSliceMut},
    net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4},
    sys::unsupported,
};
use r_efi::protocols;

pub(crate) struct TcpProtocol {
    protocol: TcpProtocolInner,
    read_timeout: RefCell<Option<u64>>,
    write_timeout: RefCell<Option<u64>>,
}

enum TcpProtocolInner {
    V4(tcp4::Tcp4Protocol),
}

impl TcpProtocol {
    fn new(
        protocol: TcpProtocolInner,
        read_timeout: Option<u64>,
        write_timeout: Option<u64>,
    ) -> Self {
        Self {
            protocol,
            read_timeout: RefCell::new(read_timeout),
            write_timeout: RefCell::new(write_timeout),
        }
    }

    pub(crate) fn bind(addr: &SocketAddr) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(x) => {
                let t = create_tcp4_all_handles(
                    false,
                    x,
                    &Ipv4Addr::new(255, 255, 255, 0),
                    &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
                )?;
                Ok(Self::new(TcpProtocolInner::from(t), None, None))
            }
            SocketAddr::V6(_x) => unsupported(),
        }
    }

    pub(crate) fn connect(addr: &SocketAddr) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(addr) => {
                let tcp4_protocol = create_tcp4_all_handles(
                    true,
                    addr,
                    &Ipv4Addr::new(255, 255, 255, 0),
                    &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
                )?;
                tcp4_protocol.connect(None)?;
                Ok(Self::new(TcpProtocolInner::from(tcp4_protocol), None, None))
            }
            SocketAddr::V6(_) => unsupported(),
        }
    }

    pub(crate) fn connect_timeout(addr: &SocketAddr, timeout: u64) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(addr) => {
                let tcp4_protocol = create_tcp4_all_handles(
                    true,
                    addr,
                    &Ipv4Addr::new(255, 255, 255, 0),
                    &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
                )?;
                tcp4_protocol.connect(Some(timeout))?;
                Ok(Self::new(TcpProtocolInner::from(tcp4_protocol), None, None))
            }
            SocketAddr::V6(_) => unsupported(),
        }
    }

    pub(crate) fn accept(&self) -> io::Result<(TcpProtocol, SocketAddr)> {
        let stream = match &self.protocol {
            TcpProtocolInner::V4(x) => TcpProtocol::new(
                TcpProtocolInner::from(x.accept()?),
                *self.read_timeout.borrow(),
                *self.write_timeout.borrow(),
            ),
        };
        let socket_addr = stream.peer_addr()?;
        Ok((stream, SocketAddr::from(socket_addr)))
    }

    #[inline]
    pub(crate) fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => x.receive(buf, *self.read_timeout.borrow()),
        }
    }

    #[inline]
    pub(crate) fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => x.receive_vectored(bufs, *self.read_timeout.borrow()),
        }
    }

    #[inline]
    pub(crate) fn write(&self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() == 0 {
            // Writing a zero-length buffer (even for a connection closed by client) seems succeed
            // in Linux. Thus doing the same here.
            return Ok(0);
        }
        match &self.protocol {
            TcpProtocolInner::V4(x) => x.transmit(buf, *self.write_timeout.borrow()),
        }
    }

    #[inline]
    pub(crate) fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => x.transmit_vectored(bufs, *self.write_timeout.borrow()),
        }
    }

    pub(crate) fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        match how {
            Shutdown::Read => unsupported(),
            Shutdown::Write => unsupported(),
            Shutdown::Both => match &self.protocol {
                TcpProtocolInner::V4(x) => x.close(false),
            },
        }
    }

    #[inline]
    pub(crate) fn peer_addr(&self) -> io::Result<SocketAddr> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => Ok(x.remote_socket()?.into()),
        }
    }

    #[inline]
    pub(crate) fn local_addr(&self) -> io::Result<SocketAddr> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => Ok(x.station_socket()?.into()),
        }
    }

    pub(crate) fn nodelay(&self) -> io::Result<bool> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => {
                let config_data = x.get_config_data()?;
                let b = unsafe { (*config_data.control_option).enable_nagle };
                Ok(bool::from(b))
            }
        }
    }

    pub(crate) fn ttl(&self) -> io::Result<u32> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => {
                let config_data = x.get_config_data()?;
                Ok(u32::from(config_data.time_to_live))
            }
        }
    }

    pub(crate) fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        match &self.protocol {
            TcpProtocolInner::V4(x) => {
                let mut config_data = x.get_config_data()?;
                config_data.time_to_live = ttl as u8;
                x.reset()?;
                x.configure(&mut config_data)
            }
        }
    }

    pub(crate) fn read_timeout(&self) -> io::Result<Option<Duration>> {
        match self.read_timeout.try_borrow() {
            Ok(x) => match *x {
                Some(timeout) => Ok(Some(Duration::from_nanos(timeout * 100))),
                None => Ok(None),
            },
            Err(_) => Err(io::const_io_error!(io::ErrorKind::ResourceBusy, "read timeout is busy")),
        }
    }

    pub(crate) fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        let timeout = match timeout {
            None => None,
            Some(x) => Some(u64::try_from(x.as_nanos() / 100).map_err(|_| {
                io::const_io_error!(io::ErrorKind::InvalidInput, "Timeout too long")
            })?),
        };
        *self.read_timeout.borrow_mut() = timeout;
        Ok(())
    }

    pub(crate) fn write_timeout(&self) -> io::Result<Option<Duration>> {
        match self.write_timeout.try_borrow() {
            Ok(x) => match *x {
                Some(timeout) => Ok(Some(Duration::from_nanos(timeout * 100))),
                None => Ok(None),
            },
            Err(_) => Err(io::const_io_error!(io::ErrorKind::ResourceBusy, "read timeout is busy")),
        }
    }

    pub(crate) fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        let timeout = match timeout {
            None => None,
            Some(x) => Some(u64::try_from(x.as_nanos() / 100).map_err(|_| {
                io::const_io_error!(io::ErrorKind::InvalidInput, "Timeout too long")
            })?),
        };
        *self.write_timeout.borrow_mut() = timeout;
        Ok(())
    }
}

impl From<tcp4::Tcp4Protocol> for TcpProtocolInner {
    #[inline]
    fn from(t: tcp4::Tcp4Protocol) -> Self {
        TcpProtocolInner::V4(t)
    }
}

fn create_tcp4_all_handles(
    active_flag: bool,
    station_addr: &crate::net::SocketAddrV4,
    subnet_mask: &crate::net::Ipv4Addr,
    remote_addr: &crate::net::SocketAddrV4,
) -> io::Result<tcp4::Tcp4Protocol> {
    let handles = common::locate_handles(protocols::tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;

    // Try all handles
    for handle in handles {
        let service_binding = uefi_service_binding::ServiceBinding::new(
            protocols::tcp4::SERVICE_BINDING_PROTOCOL_GUID,
            handle,
        );
        let tcp4_protocol = match tcp4::Tcp4Protocol::create(service_binding) {
            Ok(x) => x,
            Err(_) => {
                continue;
            }
        };

        match tcp4_protocol.default_config(
            true,
            active_flag,
            station_addr,
            subnet_mask,
            remote_addr,
        ) {
            Ok(()) => return Ok(tcp4_protocol),
            Err(_) => {
                continue;
            }
        }
    }

    Err(io::const_io_error!(io::ErrorKind::Other, "failed to open any EFI_TCP4_PROTOCOL"))
}
