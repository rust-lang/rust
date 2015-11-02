pub use sys::imp::net::{
    LookupAddr, LookupHost,
    SocketAddrV4, SocketAddrV6,
    IpAddrV4, IpAddrV6,
    Socket, TcpStream, TcpListener, UdpSocket,
    lookup_host, lookup_addr,
    connect_tcp, bind_tcp, bind_udp,
};

pub enum SocketAddr {
    V4(SocketAddrV4),
    V6(SocketAddrV6),
}

pub enum IpAddr {
    V4(IpAddrV4),
    V6(IpAddrV6),
}

impl Copy for IpAddr { }
impl Copy for SocketAddr { }
impl Clone for IpAddr { fn clone(&self) -> Self { *self } }
impl Clone for SocketAddr { fn clone(&self) -> Self { *self } }
