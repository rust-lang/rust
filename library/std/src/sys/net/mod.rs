/// This module contains the implementations of `TcpStream`, `TcpListener` and
/// `UdpSocket` as well as related functionality like DNS resolving.
mod connection;
pub(crate) use connection::*;

mod hostname;
pub(crate) use hostname::hostname;
