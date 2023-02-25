//! Networking primitives for IP communication.
//!
//! This module provides types for IP and socket addresses.
//!
//! # Organization
//!
//! * [`IpAddr`] represents IP addresses of either IPv4 or IPv6; [`Ipv4Addr`] and
//!   [`Ipv6Addr`] are respectively IPv4 and IPv6 addresses
//! * [`SocketAddr`] represents socket addresses of either IPv4 or IPv6; [`SocketAddrV4`]
//!   and [`SocketAddrV6`] are respectively IPv4 and IPv6 socket addresses

#![unstable(feature = "ip_in_core", issue = "108443")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::ip_addr::{IpAddr, Ipv4Addr, Ipv6Addr, Ipv6MulticastScope};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::parser::AddrParseError;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::socket_addr::{SocketAddr, SocketAddrV4, SocketAddrV6};

mod display_buffer;
mod ip_addr;
mod parser;
mod socket_addr;
