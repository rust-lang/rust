#![allow(unused_variables, dead_code)]
use crate::path::Path;
use crate::sys::cvt;
use crate::{fmt, io, mem};

pub(super) fn sockaddr_un(path: &Path) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    Err(crate::io::const_io_error!(
        crate::io::ErrorKind::Unsupported,
        "unix sockets are not supported on this platform",
    ))
}

fn sun_path_offset(addr: &libc::sockaddr_un) -> usize {
    // Work with an actual instance of the type since using a null pointer is UB
    let base = (addr as *const libc::sockaddr_un).addr();
    let path = (&addr.sun_path as *const libc::c_char).addr();
    path - base
}

/// An address associated with a Unix socket.
///
/// Not currently supported on this platform
#[derive(Clone)]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct SocketAddr {
    pub(super) addr: libc::sockaddr_un,
    pub(super) len: libc::socklen_t,
}

impl SocketAddr {
    pub(super) fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int,
    {
        unsafe {
            let mut addr: libc::sockaddr_un = mem::zeroed();
            let mut len = mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
            cvt(f(&mut addr as *mut _ as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }

    pub(super) fn from_parts(
        addr: libc::sockaddr_un,
        mut len: libc::socklen_t,
    ) -> io::Result<SocketAddr> {
        if len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            len = sun_path_offset(&addr) as libc::socklen_t; // i.e., zero-length address
        } else if addr.sun_family != libc::AF_UNIX as libc::sa_family_t {
            return Err(io::const_io_error!(
                io::ErrorKind::InvalidInput,
                "file descriptor did not correspond to a Unix socket",
            ));
        }

        Ok(SocketAddr { addr, len })
    }

    /// Constructs a `SockAddr` with the family `AF_UNIX` and the provided path.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket_creation", since = "1.61.0")]
    pub fn from_pathname<P>(path: P) -> io::Result<SocketAddr>
    where
        P: AsRef<Path>,
    {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns `true` if the address is unnamed.
    ///
    /// Not currently supported on this platform
    #[must_use]
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn is_unnamed(&self) -> bool {
        true
    }

    /// Returns the contents of this address if it is a `pathname` address.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    #[must_use]
    pub fn as_pathname(&self) -> Option<&Path> {
        None
    }

    /// Returns the contents of this address if it is an abstract namespace
    /// without the leading null byte.
    ///
    /// Not currently supported on this platform
    #[doc(cfg(any(target_os = "android", target_os = "linux")))]
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    #[unstable(feature = "unix_socket_abstract", issue = "85410")]
    pub fn as_abstract_namespace(&self) -> Option<&[u8]> {
        None
    }

    /// Creates an abstract domain socket address from a namespace
    ///
    /// Not currently supported on this platform
    #[doc(cfg(any(target_os = "android", target_os = "linux")))]
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    #[unstable(feature = "unix_socket_abstract", issue = "85410")]
    pub fn from_abstract_namespace(_namespace: &[u8]) -> io::Result<SocketAddr> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ));
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "(unnamed)")
    }
}
