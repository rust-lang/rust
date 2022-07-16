use super::tcp4 as uefi_tcp4;
// use super::tcp6 as uefi_tcp6;
use super::uefi_service_binding;
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, SocketAddrV4, SocketAddrV6};
use crate::os::uefi;
use crate::os::uefi::raw::protocols::{tcp4, tcp6};
use crate::sys::unsupported;
use crate::time::Duration;

pub struct TcpStream {
    inner: uefi_tcp4::Tcp4Protocol,
}

impl TcpStream {
    fn new(inner: uefi_tcp4::Tcp4Protocol) -> Self {
        Self { inner }
    }

    pub fn connect(_: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        todo!()
    }

    pub fn connect_timeout(_: &SocketAddr, _: Duration) -> io::Result<TcpStream> {
        todo!()
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unimplemented!()
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unimplemented!()
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        unimplemented!()
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        unimplemented!()
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        unimplemented!()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.receive(buf)
    }

    // FIXME: Maybe can implment using Fragment Tables
    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.transmit(buf)
    }

    // FIXME: Maybe can implment using Fragment Tables
    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Ok(SocketAddr::from(self.inner.remote_socket()?))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(SocketAddr::from(self.inner.station_socket()?))
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        match how {
            Shutdown::Read => unsupported(),
            Shutdown::Write => unsupported(),
            Shutdown::Both => self.inner.close(false),
        }
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        unimplemented!()
    }

    // Seems to be similar to abort_on_close option in `EFI_TCP6_PROTOCOL->Close()`
    pub fn set_linger(&self, _: Option<Duration>) -> io::Result<()> {
        todo!()
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        todo!()
    }

    // Seems to be similar to `EFI_TCP6_OPTION->EnableNagle`
    pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
        todo!()
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        todo!()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unimplemented!()
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unimplemented!()
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        todo!()
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct TcpListener {
    inner: uefi_tcp4::Tcp4Protocol,
}

impl TcpListener {
    fn new(inner: uefi_tcp4::Tcp4Protocol) -> Self {
        Self { inner }
    }

    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        let addr = addr?;
        match addr {
            SocketAddr::V4(x) => {
                let handles = uefi::env::locate_handles(tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;

                // Try all handles
                for handle in handles {
                    let service_binding = uefi_service_binding::ServiceBinding::new(
                        tcp4::SERVICE_BINDING_PROTOCOL_GUID,
                        handle,
                    );
                    let tcp4_protocol = match uefi_tcp4::Tcp4Protocol::create(service_binding) {
                        Ok(x) => x,
                        Err(e) => {
                            println!("Error creating Protocol from Service Binding: {:?}", e);
                            continue;
                        }
                    };

                    // Not sure about Station/Remote address yet
                    match tcp4_protocol.config(
                        true,
                        false,
                        x,
                        &Ipv4Addr::new(255, 255, 255, 0),
                        &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
                    ) {
                        Ok(()) => return Ok(TcpListener::new(tcp4_protocol)),
                        Err(e) => {
                            println!("Error during Protocol Config: {:?}", e);
                            continue;
                        }
                    }
                }

                Err(io::Error::new(io::ErrorKind::Other, "Failed to open any EFI_TCP6_PROTOCOL"))
            }
            SocketAddr::V6(x) => {
                todo!();
                // let handles = uefi::env::locate_handles(tcp6::SERVICE_BINDING_PROTOCOL_GUID)?;

                // // Try all handles
                // for handle in handles {
                //     let service_binding = uefi_service_binding::ServiceBinding::new(
                //         tcp6::SERVICE_BINDING_PROTOCOL_GUID,
                //         handle,
                //     );
                //     let tcp6_protocol = match uefi_tcp6::Tcp6Protocol::create(service_binding) {
                //         Ok(x) => x,
                //         Err(e) => {
                //             println!("Error creating Protocol from Service Binding: {:?}", e);
                //             continue;
                //         }
                //     };

                //     // Not sure about Station/Remote address yet
                //     match tcp6_protocol.config(
                //         false,
                //         &SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, x.port(), 0, 0),
                //         &SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, x.port(), 0, 0),
                //     ) {
                //         Ok(()) => return Ok(TcpListener::new(tcp6_protocol)),
                //         Err(e) => {
                //             println!("Error during Protocol Config: {:?}", e);
                //             continue;
                //         }
                //     }
                // }

                // Err(io::Error::new(io::ErrorKind::Other, "Failed to open any EFI_TCP6_PROTOCOL"))
            }
        }
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(SocketAddr::from(self.inner.station_socket()?))
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let new_protocol = self.inner.accept()?;
        // FIXME: Removing this causes remote_socket method to freeze for some reason
        // println!("Here");
        // let socket_addr = new_protocol.remote_socket()?;
        let socket_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0);
        let stream = TcpStream::new(new_protocol);
        Ok((stream, SocketAddr::from(socket_addr)))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        unimplemented!()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unimplemented!()
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        unimplemented!()
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Ok(false)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unimplemented!()
    }

    // Internally TCP6 Protocol is nonblocking
    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        todo!()
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct UdpSocket(!);

impl UdpSocket {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        unsupported()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.0
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.0
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        self.0
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.0
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        self.0
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        self.0
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        self.0
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        self.0
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        self.0
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        self.0
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        self.0
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        self.0
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn ttl(&self) -> io::Result<u32> {
        self.0
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn connect(&self, _: io::Result<&SocketAddr>) -> io::Result<()> {
        self.0
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

pub struct LookupHost(!);

impl LookupHost {
    pub fn port(&self) -> u16 {
        self.0
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        self.0
    }
}

impl TryFrom<&str> for LookupHost {
    type Error = io::Error;

    fn try_from(_v: &str) -> io::Result<LookupHost> {
        unsupported()
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from(_v: (&'a str, u16)) -> io::Result<LookupHost> {
        unsupported()
    }
}

#[allow(nonstandard_style)]
pub mod netc {
    pub const AF_INET: u8 = 0;
    pub const AF_INET6: u8 = 1;
    pub type sa_family_t = u8;

    #[derive(Copy, Clone)]
    pub struct in_addr {
        pub s_addr: u32,
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr_in {
        pub sin_family: sa_family_t,
        pub sin_port: u16,
        pub sin_addr: in_addr,
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr_in6 {
        pub sin6_family: sa_family_t,
        pub sin6_port: u16,
        pub sin6_addr: in6_addr,
        pub sin6_flowinfo: u32,
        pub sin6_scope_id: u32,
    }

    #[derive(Copy, Clone)]
    pub struct in6_addr {
        pub s6_addr: [u8; 16],
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr {}

    pub type socklen_t = usize;
}
