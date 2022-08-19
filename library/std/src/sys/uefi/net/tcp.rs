//! Leaving Most of this unimplemented since TCP was mostly implemented to get testing working.
//! In the future, should probably desing networking around SIMPLE_NETWOR_PROTOCOL

use super::{tcp4, uefi_service_binding};
use crate::sys::uefi::common;
use crate::{
    io::{self, IoSlice, IoSliceMut},
    net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4},
    sys::unsupported,
};
use r_efi::protocols;

pub enum TcpProtocol {
    V4(tcp4::Tcp4Protocol),
}

impl TcpProtocol {
    pub fn bind(addr: &SocketAddr) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(x) => {
                let handles =
                    common::locate_handles(protocols::tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;

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
                        false,
                        x,
                        &Ipv4Addr::new(255, 255, 255, 0),
                        &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
                    ) {
                        Ok(()) => return Ok(Self::from(tcp4_protocol)),
                        Err(_) => {
                            continue;
                        }
                    }
                }

                Err(io::error::const_io_error!(
                    io::ErrorKind::Other,
                    "Failed to open any EFI_TCP6_PROTOCOL"
                ))
            }
            SocketAddr::V6(_x) => {
                todo!();
            }
        }
    }

    // FIXME: Not reall tested properly yet
    pub fn connect(addr: &SocketAddr) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(addr) => {
                let handles =
                    common::locate_handles(protocols::tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;

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
                        true,
                        &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
                        &Ipv4Addr::new(255, 255, 255, 0),
                        addr,
                    ) {
                        Ok(()) => match tcp4_protocol.connect() {
                            Ok(_) => return Ok(Self::from(tcp4_protocol)),
                            Err(e) => {
                                eprintln!("Connect Error: {e:?}");
                                continue;
                            }
                        },
                        Err(e) => {
                            eprintln!("Configure Error: {e:?}");
                            continue;
                        }
                    }
                }

                Err(io::error::const_io_error!(
                    io::ErrorKind::Other,
                    "Failed to open any EFI_TCP6_PROTOCOL"
                ))
            }
            SocketAddr::V6(_) => unsupported(),
        }
    }

    pub fn accept(&self) -> io::Result<(TcpProtocol, SocketAddr)> {
        let stream = match self {
            TcpProtocol::V4(x) => TcpProtocol::from(x.accept()?),
        };
        let socket_addr = stream.peer_addr()?;
        Ok((stream, SocketAddr::from(socket_addr)))
    }

    #[inline]
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.receive(buf),
        }
    }

    #[inline]
    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.receive_vectored(bufs),
        }
    }

    #[inline]
    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.transmit(buf),
        }
    }

    #[inline]
    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.transmit_vectored(bufs),
        }
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        match how {
            Shutdown::Read => unsupported(),
            Shutdown::Write => unsupported(),
            Shutdown::Both => match self {
                TcpProtocol::V4(x) => x.close(false),
            },
        }
    }

    #[inline]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        match self {
            TcpProtocol::V4(x) => Ok(x.remote_socket()?.into()),
        }
    }

    #[inline]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        match self {
            TcpProtocol::V4(x) => Ok(x.station_socket()?.into()),
        }
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        match self {
            TcpProtocol::V4(x) => {
                let config_data = x.get_config_data()?;
                let b = unsafe { (*config_data.control_option).enable_nagle };
                Ok(bool::from(b))
            }
        }
    }

    pub fn ttl(&self) -> io::Result<u32> {
        match self {
            TcpProtocol::V4(x) => {
                let config_data = x.get_config_data()?;
                Ok(u32::from(config_data.time_to_live))
            }
        }
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        match self {
            TcpProtocol::V4(x) => {
                let mut config_data = x.get_config_data()?;
                config_data.time_to_live = ttl as u8;
                x.reset()?;
                x.configure(&mut config_data)
            }
        }
    }
}

impl From<tcp4::Tcp4Protocol> for TcpProtocol {
    #[inline]
    fn from(t: tcp4::Tcp4Protocol) -> Self {
        Self::V4(t)
    }
}
