//! WASI-specific networking functionality

#![unstable(feature = "wasi_ext", issue = "71213")]

use crate::os::fd::AsRawFd;
use crate::sys::err2io;
use crate::{io, net};

/// WASI-specific extensions to [`std::net::TcpListener`].
///
/// [`std::net::TcpListener`]: crate::net::TcpListener
pub trait TcpListenerExt {
    /// Accept a socket.
    ///
    /// This corresponds to the `sock_accept` syscall.
    fn sock_accept(&self, flags: u16) -> io::Result<u32>;
}

impl TcpListenerExt for net::TcpListener {
    fn sock_accept(&self, flags: u16) -> io::Result<u32> {
        unsafe { wasi::sock_accept(self.as_raw_fd() as wasi::Fd, flags).map_err(err2io) }
    }
}
