pub use imp::net as imp;

pub mod traits {
    pub use super::{Net as sys_Net, LookupAddr as sys_LookupAddr, AddrV4 as sys_AddrV4, AddrV6 as sys_AddrV6, SocketAddrV4 as sys_SocketAddrV4, SocketAddrV6 as sys_SocketAddrV6, Socket as sys_Socket, TcpStream as sys_TcpStream, TcpListener as sys_TcpListener, UdpSocket as sys_UdpSocket};
}

pub mod prelude {
    pub use super::imp::Net;
    pub use super::traits::*;
    pub use super::Shutdown;

    pub mod ipaddr {
        pub use net::IpAddr::*;
    }

    pub mod socketaddr {
        pub use net::SocketAddr::*;
    }

    pub type SocketAddr = super::SocketAddr<Net>;
    pub type IpAddr = super::IpAddr<Net>;
    pub type SocketAddrV4 = <Net as sys_Net>::SocketAddrV4;
    pub type SocketAddrV6 = <Net as sys_Net>::SocketAddrV6;
    pub type AddrV4 = <SocketAddrV4 as sys_SocketAddrV4>::Addr;
    pub type AddrV6 = <SocketAddrV6 as sys_SocketAddrV6>::Addr;
    pub type LookupHost = <Net as sys_Net>::LookupHost;
    pub type LookupAddr = <Net as sys_Net>::LookupAddr;
    pub type Socket = <Net as sys_Net>::Socket;
    pub type TcpStream = <Net as sys_Net>::TcpStream;
    pub type TcpListener = <Net as sys_Net>::TcpListener;
    pub type UdpSocket = <Net as sys_Net>::UdpSocket;
}

use error::prelude::*;
use inner::prelude::*;
use io;
use core::result;
use core::fmt;
use core::iter;
use core::hash;
use core::str;
use core::time::Duration;

pub enum Shutdown {
    Read,
    Write,
    Both,
}

pub trait LookupHost<N: Net + ?Sized>: iter::Iterator<Item=Result<SocketAddr<N>>> { }

pub trait LookupAddr {
    fn as_str(&self) -> result::Result<&str, str::Utf8Error> {
        str::from_utf8(self.as_bytes())
    }

    fn as_bytes(&self) -> &[u8];
}

pub enum SocketAddr<N: Net + ?Sized> {
    V4(N::SocketAddrV4),
    V6(N::SocketAddrV6),
}

pub enum IpAddr<N: Net + ?Sized> {
    V4(<N::SocketAddrV4 as SocketAddrV4>::Addr),
    V6(<N::SocketAddrV6 as SocketAddrV6>::Addr),
}

impl<N: Net + ?Sized> Copy for IpAddr<N> { }
impl<N: Net + ?Sized> Copy for SocketAddr<N> { }
impl<N: Net + ?Sized> Clone for IpAddr<N> { fn clone(&self) -> Self { *self } }
impl<N: Net + ?Sized> Clone for SocketAddr<N> { fn clone(&self) -> Self { *self } }

pub trait AddrV4: Copy + Clone + Sized + PartialOrd + Ord + PartialEq + Eq + hash::Hash {
    fn new(a: u8, b: u8, c: u8, d: u8) -> Self;

    fn octets(&self) -> [u8; 4];
}

pub trait AddrV6: Copy + Clone + Sized + PartialOrd + Ord + PartialEq + Eq + hash::Hash {
    fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Self;

    fn segments(&self) -> [u16; 8];
}

pub trait SocketAddrV4: Copy + Clone + Sized + PartialEq + Eq + hash::Hash {
    type Addr: AddrV4;

    fn new(ip: Self::Addr, port: u16) -> Self;

    fn addr(&self) -> &Self::Addr;
    fn port(&self) -> u16;
}

pub trait SocketAddrV6: Copy + Clone + Sized + PartialEq + Eq + hash::Hash  {
    type Addr: AddrV6;

    fn new(ip: Self::Addr, port: u16, flowinfo: u32, scope_id: u32) -> Self;

    fn addr(&self) -> &Self::Addr;
    fn port(&self) -> u16;
    fn flowinfo(&self) -> u32;
    fn scope_id(&self) -> u32;
}

pub trait Net {
    type SocketAddrV4: SocketAddrV4;
    type SocketAddrV6: SocketAddrV6;

    type LookupHost: LookupHost<Self>;

    type Socket: fmt::Debug;
    type TcpStream: TcpStream<Self>;
    type TcpListener: TcpListener<Self>;
    type UdpSocket: UdpSocket<Self>;
    type LookupAddr: LookupAddr;

    fn lookup_host(host: &str) -> Result<Self::LookupHost> where Self::LookupHost: Sized;
    fn lookup_addr(addr: &IpAddr<Self>) -> Result<Self::LookupAddr> where Self::LookupAddr: Sized;

    fn connect_tcp(addr: &SocketAddr<Self>) -> Result<Self::TcpStream> where Self::TcpStream: Sized;
    fn bind_tcp(addr: &SocketAddr<Self>) -> Result<Self::TcpListener> where Self::TcpListener: Sized;
    fn bind_udp(addr: &SocketAddr<Self>) -> Result<Self::UdpSocket> where Self::UdpSocket: Sized;
}

pub trait Socket<N: Net + ?Sized>: FromInner<N::Socket> + AsInner<N::Socket> + IntoInner<N::Socket> {
    fn socket(&self) -> &N::Socket;
    fn into_socket(self) -> N::Socket where Self: Sized;

    fn socket_addr(&self) -> Result<SocketAddr<N>>;

    fn set_read_timeout(&self, dur: Option<Duration>) -> Result<()>;
    fn set_write_timeout(&self, dur: Option<Duration>) -> Result<()>;
    fn read_timeout(&self) -> Result<Option<Duration>>;
    fn write_timeout(&self) -> Result<Option<Duration>>;

    fn duplicate(&self) -> Result<Self> where Self: Sized;
}

pub trait TcpStream<N: Net + ?Sized>: Socket<N> + io::Read + io::Write {
    fn peer_addr(&self) -> Result<SocketAddr<N>>;
    fn shutdown(&self, how: Shutdown) -> Result<()>;
}

pub trait TcpListener<N: Net + ?Sized>: Socket<N> {
    fn accept(&self) -> Result<(N::TcpStream, SocketAddr<N>)>;
}

pub trait UdpSocket<N: Net + ?Sized>: Socket<N> {
    fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr<N>)>;
    fn send_to(&self, buf: &[u8], dst: &SocketAddr<N>) -> Result<usize>;
}
