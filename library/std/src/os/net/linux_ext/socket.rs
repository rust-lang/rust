//! Linux and Android-specific socket functionality.

use crate::io;
use crate::os::unix::net;
use crate::sealed::Sealed;
use crate::sys_common::AsInner;

/// Linux-specific functionality for `AF_UNIX` sockets [`UnixDatagram`]
/// and [`UnixStream`].
///
/// [`UnixDatagram`]: net::UnixDatagram
/// [`UnixStream`]: net::UnixStream
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub trait UnixSocketExt: Sealed {
    /// Query the current setting of socket option `SO_PASSCRED`.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    fn passcred(&self) -> io::Result<bool>;

    /// Enable or disable socket option `SO_PASSCRED`.
    ///
    /// This option enables the credentials of the sending process to be
    /// received as a control message in [`AncillaryData`].
    ///
    /// [`AncillaryData`]: net::AncillaryData
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// #[cfg(target_os = "linux")]
    /// use std::os::linux::net::UnixSocketExt;
    /// #[cfg(target_os = "android")]
    /// use std::os::android::net::UnixSocketExt;
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixDatagram::unbound()?;
    ///     sock.set_passcred(true).expect("set_passcred failed");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    fn set_passcred(&self, passcred: bool) -> io::Result<()>;
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl UnixSocketExt for net::UnixDatagram {
    fn passcred(&self) -> io::Result<bool> {
        self.as_inner().passcred()
    }

    fn set_passcred(&self, passcred: bool) -> io::Result<()> {
        self.as_inner().set_passcred(passcred)
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl UnixSocketExt for net::UnixStream {
    fn passcred(&self) -> io::Result<bool> {
        self.as_inner().passcred()
    }

    fn set_passcred(&self, passcred: bool) -> io::Result<()> {
        self.as_inner().set_passcred(passcred)
    }
}
