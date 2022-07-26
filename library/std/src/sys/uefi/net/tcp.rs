use super::{tcp4, uefi_service_binding};
use crate::{
    io::{self, IoSlice, IoSliceMut},
    net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4},
    os::uefi,
    sync::Arc,
    sys::unsupported,
};
use r_efi::protocols;

pub enum TcpProtocol {
    V4(Arc<tcp4::Tcp4Protocol>),
}

impl TcpProtocol {
    pub fn bind(addr: &SocketAddr) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(x) => {
                let handles =
                    uefi::env::locate_handles(protocols::tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;

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

                    match tcp4_protocol.config(
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

                Err(io::Error::new(io::ErrorKind::Other, "Failed to open any EFI_TCP6_PROTOCOL"))
            }
            SocketAddr::V6(_x) => {
                todo!();
            }
        }
    }

    pub fn accept(&self) -> io::Result<(TcpProtocol, SocketAddr)> {
        let stream = match self {
            TcpProtocol::V4(x) => TcpProtocol::from(x.accept()?),
        };

        // FIXME: Return Actual SocketAddr
        let socket_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0);
        Ok((stream, SocketAddr::from(socket_addr)))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.receive(buf),
        }
    }

    // FIXME: Maybe can implment using Fragment Tables
    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.receive_vectored(bufs),
        }
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        match self {
            TcpProtocol::V4(x) => x.transmit(buf),
        }
    }

    // FIXME: Maybe can implment using Fragment Tables
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
}

impl From<tcp4::Tcp4Protocol> for TcpProtocol {
    fn from(t: tcp4::Tcp4Protocol) -> Self {
        Self::V4(Arc::new(t))
    }
}
