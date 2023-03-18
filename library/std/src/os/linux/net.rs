//! Linux-specific networking functionality.

#![stable(feature = "unix_socket_abstract", since = "CURRENT_RUSTC_VERSION")]

#[stable(feature = "unix_socket_abstract", since = "CURRENT_RUSTC_VERSION")]
pub use crate::os::net::linux_ext::addr::SocketAddrExt;

#[unstable(feature = "tcp_quickack", issue = "96256")]
pub use crate::os::net::linux_ext::tcp::TcpStreamExt;
