use crate::ffi::OsStr;
#[cfg(any(doc, target_os = "android", target_os = "linux"))]
use crate::os::net::linux_ext;
use crate::os::unix::ffi::OsStrExt;
use crate::path::Path;
use crate::sealed::Sealed;
use crate::sys::cvt;
use crate::{fmt, io, mem, ptr};

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(not(unix))]
#[allow(non_camel_case_types)]
mod libc {
    pub use core::ffi::c_int;
    pub type socklen_t = u32;
    pub struct sockaddr;
    #[derive(Clone)]
    pub struct sockaddr_un {
        pub sun_path: [u8; 1],
    }
}

const SUN_PATH_OFFSET: usize = mem::offset_of!(libc::sockaddr_un, sun_path);

pub(super) fn sockaddr_un(path: &Path) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    // SAFETY: All zeros is a valid representation for `sockaddr_un`.
    let mut addr: libc::sockaddr_un = unsafe { mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;

    let bytes = path.as_os_str().as_bytes();

    if bytes.contains(&0) {
        return Err(io::const_io_error!(
            io::ErrorKind::InvalidInput,
            "paths must not contain interior null bytes",
        ));
    }

    if bytes.len() >= addr.sun_path.len() {
        return Err(io::const_io_error!(
            io::ErrorKind::InvalidInput,
            "path must be shorter than SUN_LEN",
        ));
    }
    // SAFETY: `bytes` and `addr.sun_path` are not overlapping and
    // both point to valid memory.
    // NOTE: We zeroed the memory above, so the path is already null
    // terminated.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), addr.sun_path.as_mut_ptr().cast(), bytes.len())
    };

    let mut len = SUN_PATH_OFFSET + bytes.len();
    match bytes.get(0) {
        Some(&0) | None => {}
        Some(_) => len += 1,
    }
    Ok((addr, len as libc::socklen_t))
}

enum AddressKind<'a> {
    Unnamed,
    Pathname(&'a Path),
    Abstract(&'a [u8]),
}

/// An address associated with a Unix socket.
///
/// # Examples
///
/// ```
/// use std::os::unix::net::UnixListener;
///
/// let socket = match UnixListener::bind("/tmp/sock") {
///     Ok(sock) => sock,
///     Err(e) => {
///         println!("Couldn't bind: {e:?}");
///         return
///     }
/// };
/// let addr = socket.local_addr().expect("Couldn't get local address");
/// ```
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
            cvt(f((&raw mut addr) as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }

    pub(super) fn from_parts(
        addr: libc::sockaddr_un,
        mut len: libc::socklen_t,
    ) -> io::Result<SocketAddr> {
        if cfg!(target_os = "openbsd") {
            // on OpenBSD, getsockname(2) returns the actual size of the socket address,
            // and not the len of the content. Figure out the length for ourselves.
            // https://marc.info/?l=openbsd-bugs&m=170105481926736&w=2
            let sun_path: &[u8] =
                unsafe { mem::transmute::<&[libc::c_char], &[u8]>(&addr.sun_path) };
            len = core::slice::memchr::memchr(0, sun_path)
                .map_or(len, |new_len| (new_len + SUN_PATH_OFFSET) as libc::socklen_t);
        }

        if len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            len = SUN_PATH_OFFSET as libc::socklen_t; // i.e., zero-length address
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
    /// # Errors
    ///
    /// Returns an error if the path is longer than `SUN_LEN` or if it contains
    /// NULL bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::os::unix::net::SocketAddr;
    /// use std::path::Path;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let address = SocketAddr::from_pathname("/path/to/socket")?;
    /// assert_eq!(address.as_pathname(), Some(Path::new("/path/to/socket")));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Creating a `SocketAddr` with a NULL byte results in an error.
    ///
    /// ```
    /// use std::os::unix::net::SocketAddr;
    ///
    /// assert!(SocketAddr::from_pathname("/path/with/\0/bytes").is_err());
    /// ```
    #[stable(feature = "unix_socket_creation", since = "1.61.0")]
    pub fn from_pathname<P>(path: P) -> io::Result<SocketAddr>
    where
        P: AsRef<Path>,
    {
        sockaddr_un(path.as_ref()).map(|(addr, len)| SocketAddr { addr, len })
    }

    /// Returns `true` if the address is unnamed.
    ///
    /// # Examples
    ///
    /// A named address:
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixListener::bind("/tmp/sock")?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     assert_eq!(addr.is_unnamed(), false);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// An unnamed address:
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixDatagram::unbound()?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     assert_eq!(addr.is_unnamed(), true);
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn is_unnamed(&self) -> bool {
        matches!(self.address(), AddressKind::Unnamed)
    }

    /// Returns the contents of this address if it is a `pathname` address.
    ///
    /// # Examples
    ///
    /// With a pathname:
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    /// use std::path::Path;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixListener::bind("/tmp/sock")?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     assert_eq!(addr.as_pathname(), Some(Path::new("/tmp/sock")));
    ///     Ok(())
    /// }
    /// ```
    ///
    /// Without a pathname:
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixDatagram::unbound()?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     assert_eq!(addr.as_pathname(), None);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    #[must_use]
    pub fn as_pathname(&self) -> Option<&Path> {
        if let AddressKind::Pathname(path) = self.address() { Some(path) } else { None }
    }

    fn address(&self) -> AddressKind<'_> {
        let len = self.len as usize - SUN_PATH_OFFSET;
        let path = unsafe { mem::transmute::<&[libc::c_char], &[u8]>(&self.addr.sun_path) };

        // macOS seems to return a len of 16 and a zeroed sun_path for unnamed addresses
        if len == 0
            || (cfg!(not(any(target_os = "linux", target_os = "android")))
                && self.addr.sun_path[0] == 0)
        {
            AddressKind::Unnamed
        } else if self.addr.sun_path[0] == 0 {
            AddressKind::Abstract(&path[1..len])
        } else {
            AddressKind::Pathname(OsStr::from_bytes(&path[..len - 1]).as_ref())
        }
    }
}

#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
impl Sealed for SocketAddr {}

#[doc(cfg(any(target_os = "android", target_os = "linux")))]
#[cfg(any(doc, target_os = "android", target_os = "linux"))]
#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
impl linux_ext::addr::SocketAddrExt for SocketAddr {
    fn as_abstract_name(&self) -> Option<&[u8]> {
        if let AddressKind::Abstract(name) = self.address() { Some(name) } else { None }
    }

    fn from_abstract_name<N>(name: N) -> crate::io::Result<Self>
    where
        N: AsRef<[u8]>,
    {
        let name = name.as_ref();
        unsafe {
            let mut addr: libc::sockaddr_un = mem::zeroed();
            addr.sun_family = libc::AF_UNIX as libc::sa_family_t;

            if name.len() + 1 > addr.sun_path.len() {
                return Err(io::const_io_error!(
                    io::ErrorKind::InvalidInput,
                    "abstract socket name must be shorter than SUN_LEN",
                ));
            }

            crate::ptr::copy_nonoverlapping(
                name.as_ptr(),
                addr.sun_path.as_mut_ptr().add(1) as *mut u8,
                name.len(),
            );
            let len = (SUN_PATH_OFFSET + 1 + name.len()) as libc::socklen_t;
            SocketAddr::from_parts(addr, len)
        }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.address() {
            AddressKind::Unnamed => write!(fmt, "(unnamed)"),
            AddressKind::Abstract(name) => write!(fmt, "\"{}\" (abstract)", name.escape_ascii()),
            AddressKind::Pathname(path) => write!(fmt, "{path:?} (pathname)"),
        }
    }
}
