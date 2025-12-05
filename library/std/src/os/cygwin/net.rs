//! Cygwin-specific networking functionality.
//!
//! There are some limitations of Unix domain sockets on Cygwin:
//! * The syscalls `accept` and `connect` need
//! [handshake](https://inbox.sourceware.org/cygwin/Z_UERXFI1g-1v3p2@calimero.vinschen.de/T/#t).
//! * Cannot bind to abstract addr.
//! * Unbounded unix socket has an abstract local addr.
//! * Doesn't support recvmsg with control data.

#![stable(feature = "unix_socket_abstract", since = "1.70.0")]

#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
pub use crate::os::net::linux_ext::addr::SocketAddrExt;
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub use crate::os::net::linux_ext::socket::UnixSocketExt;
#[stable(feature = "tcp_quickack", since = "1.89.0")]
pub use crate::os::net::linux_ext::tcp::TcpStreamExt;
