//! Android-specific networking functionality.

#![stable(feature = "unix_socket_abstract", since = "1.70.0")]

#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
pub use crate::os::net::linux_ext::addr::SocketAddrExt;
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub use crate::os::net::linux_ext::socket::UnixSocketExt;
#[unstable(feature = "tcp_quickack", issue = "96256")]
pub use crate::os::net::linux_ext::tcp::TcpStreamExt;
