use sys::net::{SocketAddrV4, SocketAddrV6, IpAddrV4, IpAddrV6};

#[derive(Copy, Clone)]
pub enum SocketAddr {
    V4(SocketAddrV4),
    V6(SocketAddrV6),
}

#[derive(Copy, Clone)]
pub enum IpAddr {
    V4(IpAddrV4),
    V6(IpAddrV6),
}
