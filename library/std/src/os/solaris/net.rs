//! solaris-specific networking functionality.

#![unstable(feature = "unix_socket_exclbind", issue = "123481")]

use crate::io;
use crate::os::unix::net;
use crate::sealed::Sealed;
use crate::sys_common::AsInner;

/// solaris-specific functionality for `AF_UNIX` sockets [`UnixDatagram`]
/// and [`UnixStream`].
///
/// [`UnixDatagram`]: net::UnixDatagram
/// [`UnixStream`]: net::UnixStream
#[unstable(feature = "unix_socket_exclbind", issue = "123481")]
pub trait UnixSocketExt: Sealed {
    /// Enables exclusive binding on the socket.
    ///
    /// If true and if the socket had been set with `SO_REUSEADDR`,
    /// it neutralises its effect.
    /// See [`man 3 tcp`](https://docs.oracle.com/cd/E88353_01/html/E37843/setsockopt-3c.html)
    #[unstable(feature = "unix_socket_exclbind", issue = "123481")]
    fn so_exclbind(&self, excl: bool) -> io::Result<()>;

    /// Get the bind exclusivity bind state of the socket.
    #[unstable(feature = "unix_socket_exclbind", issue = "123481")]
    fn exclbind(&self) -> io::Result<bool>;
}

#[unstable(feature = "unix_socket_exclbind", issue = "123481")]
impl UnixSocketExt for net::UnixDatagram {
    fn exclbind(&self) -> io::Result<bool> {
        self.as_inner().exclbind()
    }

    fn so_exclbind(&self, excl: bool) -> io::Result<()> {
        self.as_inner().set_exclbind(excl)
    }
}

#[unstable(feature = "unix_socket_exclbind", issue = "123481")]
impl UnixSocketExt for net::UnixStream {
    fn exclbind(&self) -> io::Result<bool> {
        self.as_inner().exclbind()
    }

    fn so_exclbind(&self, excl: bool) -> io::Result<()> {
        self.as_inner().set_exclbind(excl)
    }
}
