use crate::bstr::ByteStr;
use crate::ffi::OsStr;
#[cfg(any(doc, target_os = "android", target_os = "linux", target_os = "cygwin"))]
use crate::os::net::linux_ext;
use crate::os::unix::ffi::OsStrExt;
use crate::path::Path;
use crate::sys::cvt;
use crate::{fmt, io, mem};

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(not(unix))]
#[allow(non_camel_case_types)]
mod libc {
    pub use core::ffi::c_int;
    pub type sa_family_t = u8;
    pub type socklen_t = u32;
    pub struct sockaddr;
    #[derive(Clone)]
    pub struct sockaddr_un {
        pub sun_family: sa_family_t,
        pub sun_path: [u8; 1],
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "macos", target_os = "dragonfly"))] {
        // MacOS and DragonFly utilize `SOCK_MAXADDRLEN` to define
        // the maximum size that the `sockaddr_un` struct could be.
        // SOCK_MAXADDRLEN = 255 on these platforms, and it's based on
        // sizeof(sa_len) (a u8) + sizeof(sun_family) (a u8) + 253 bytes
        // (excluding nul). We add +1 for nul byte just in case.
        // See https://github.com/rust-lang/rust/issues/160684
        pub(crate) const SOCK_MAX_SIZE: usize = libc::SOCK_MAXADDRLEN as usize + 1;
    } else if #[cfg(any(target_os = "netbsd"))] {
        // NetBSD uses `UCHAR_MAX` (essentially a `u8::MAX`) + 1 as defined by `sockaddr_big` here:
        // https://github.com/IIJ-NetBSD/netbsd-src/blob/master/sys/sys/socket.h#L272-L287
        pub(crate) const SOCK_MAX_SIZE: usize = u8::MAX as usize + 1;
    } else {
        pub(crate) const SOCK_MAX_SIZE: usize = size_of::<libc::sockaddr_un>();
    }
}

// Offset to `libc::sockaddr_un.sun_family`
pub(crate) const SUN_FAMILY_OFFSET: usize = mem::offset_of!(libc::sockaddr_un, sun_family);
// Offset to `libc::sockaddr_un.sun_path`
pub(crate) const SUN_PATH_OFFSET: usize = mem::offset_of!(libc::sockaddr_un, sun_path);
// This represents the maximum number of characters + 1 allowed to be
// stored in the socket path (e.g. 254 for NetBSD because it allows 253
// valid characters for its path, 104 for FreeBSD because it allows up to
// 103 characters for its path)
pub(crate) const SUN_PATH_MAX_LEN: usize = SOCK_MAX_SIZE - SUN_PATH_OFFSET;

enum AddressKind<'a> {
    Unnamed,
    Pathname(&'a Path),
    Abstract(&'a ByteStr),
}

/// An address associated with a Unix socket.
///
/// # Examples
///
#[cfg_attr(target_family = "unix", doc = "```")]
#[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
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
    /// Size of the socket address, `sun_family` and `sun_path`
    /// fields from `libc::sockaddr_un` included
    pub(super) len: libc::socklen_t,
    /// Stack allocated buffer that contains full size of what `libc::sockaddr_un`
    /// could be (as `sun_path` field defined for `sockaddr_un` does not represent
    /// the maximum path length of a Unix Domain socket name)
    pub(super) addr: [u8; SOCK_MAX_SIZE],
}

impl SocketAddr {
    pub(super) fn default() -> SocketAddr {
        let mut addr: [u8; SOCK_MAX_SIZE] = [0; SOCK_MAX_SIZE];
        let sun_family = (libc::AF_UNIX as libc::sa_family_t).to_ne_bytes();
        addr[SUN_FAMILY_OFFSET..SUN_FAMILY_OFFSET + size_of::<libc::sa_family_t>()]
            .copy_from_slice(&sun_family);

        SocketAddr { len: SUN_PATH_OFFSET as libc::socklen_t, addr }
    }

    pub(super) fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(*mut [u8; SOCK_MAX_SIZE], *mut libc::socklen_t) -> libc::c_int,
    {
        let mut addr: [u8; SOCK_MAX_SIZE] = [0; SOCK_MAX_SIZE];
        let mut len = size_of::<libc::sockaddr_un>() as libc::socklen_t;
        cvt(f((&raw mut addr) as *mut _, &mut len))?;
        SocketAddr::from_parts(addr, len)
    }

    pub(super) fn from_path(path: &Path) -> io::Result<SocketAddr> {
        let mut sockaddr = SocketAddr::default();

        let bytes = path.as_os_str().as_bytes();

        if bytes.contains(&0) {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "paths must not contain interior null bytes",
            ));
        }

        if bytes.len() >= SUN_PATH_MAX_LEN {
            const LEN_EXCEEDED_MSG: &'static str =
                cfg_select! {
                    any(target_os = "macos", target_os = "dragonfly") => {
                        "path must be shorter than SOCK_MAXADDRLEN - 1"
                    }
                    target_os = "netbsd" => "path must be shorter than UCHAR_MAX - 1",
                    _ => "path must be shorter than SUN_LEN",
                };
            return Err(io::const_error!(io::ErrorKind::InvalidInput, LEN_EXCEEDED_MSG));
        }

        sockaddr.set_path(bytes);

        let mut len = SUN_PATH_OFFSET + bytes.len();
        #[cfg(any(
            target_os = "dragonfly",
            target_os = "macos",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "freebsd"
        ))]
        {
            const _: () = assert!((SUN_PATH_MAX_LEN as usize) <= (u8::MAX as usize));
            // For these platforms, `sockaddr_un` has a `sun_len` (u8) field that should be initialized with a value
            // (nul byte excluding)
            sockaddr.addr[mem::offset_of!(libc::sockaddr_un, sun_len)] = len as u8;
        }

        match bytes.get(0) {
            Some(&0) | None => {}
            Some(_) => {
                // on QNX7.1 and QNX8 the `len` value returned by the SUN_LEN
                // macro in its libc does not include the null byte in the count so
                // don't add it here to match what a C program passes to bind(2) and
                // similar functions
                if cfg!(not(any(target_os = "qnx", target_env = "nto71"))) {
                    len += 1
                }
            }
        }

        // Even though len here is a `usize` and `libc::socklen_t` is a `u32`
        // our len value should be limited to whatever value a `u32` can hold
        sockaddr.set_len(len as libc::socklen_t);

        Ok(sockaddr)
    }

    pub(super) fn from_parts(
        addr: [u8; SOCK_MAX_SIZE],
        mut len: libc::socklen_t,
    ) -> io::Result<SocketAddr> {
        if cfg!(target_os = "openbsd") {
            // on OpenBSD, getsockname(2) returns the actual size of the socket address,
            // and not the len of the content. Figure out the length for ourselves.
            // https://marc.info/?l=openbsd-bugs&m=170105481926736&w=2
            let sun_path = &addr[SUN_PATH_OFFSET..];
            len = core::slice::memchr::memchr(0, sun_path)
                .map_or(len, |new_len| (new_len + SUN_PATH_OFFSET) as libc::socklen_t);
        }

        if len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            len = SUN_PATH_OFFSET as libc::socklen_t; // i.e., zero-length address
        } else if SocketAddr::sun_family_from_addr(&addr) != libc::AF_UNIX as libc::sa_family_t {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "file descriptor did not correspond to a Unix socket",
            ));
        }

        Ok(SocketAddr { len, addr })
    }

    fn sun_family_from_addr(addr: &[u8; SOCK_MAX_SIZE]) -> libc::sa_family_t {
        let sun_family_array = addr[SUN_FAMILY_OFFSET..SUN_FAMILY_OFFSET + size_of::<libc::sa_family_t>()].try_into().expect("Slice should have exactly the same number of bytes extracted as the size of libc::sa_family_t");
        libc::sa_family_t::from_ne_bytes(sun_family_array)
    }

    fn set_path(&mut self, path_bytes: &[u8]) {
        self.addr[SUN_PATH_OFFSET..SUN_PATH_OFFSET + path_bytes.len()].copy_from_slice(path_bytes);
    }

    fn set_len(&mut self, len: libc::socklen_t) {
        self.len = len;
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
    #[cfg_attr(target_family = "unix", doc = "```")]
    #[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
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
    #[cfg_attr(target_family = "unix", doc = "```")]
    #[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
    /// use std::os::unix::net::SocketAddr;
    ///
    /// assert!(SocketAddr::from_pathname("/path/with/\0/bytes").is_err());
    /// ```
    #[stable(feature = "unix_socket_creation", since = "1.61.0")]
    pub fn from_pathname<P>(path: P) -> io::Result<SocketAddr>
    where
        P: AsRef<Path>,
    {
        SocketAddr::from_path(path.as_ref())
    }

    /// Returns `true` if the address is unnamed.
    ///
    /// # Examples
    ///
    /// A named address:
    ///
    #[cfg_attr(target_family = "unix", doc = "```no_run")]
    #[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
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
    #[cfg_attr(target_family = "unix", doc = "```")]
    #[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
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
    #[cfg_attr(target_family = "unix", doc = "```no_run")]
    #[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
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
    #[cfg_attr(target_family = "unix", doc = "```")]
    #[cfg_attr(not(target_family = "unix"), doc = "```ignore (needs unix)")]
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
        let path = &self.addr[SUN_PATH_OFFSET..];

        // macOS seems to return a len of 16 and a zeroed sun_path for unnamed addresses
        if len == 0
            || (cfg!(not(any(target_os = "linux", target_os = "android", target_os = "cygwin")))
                && path[0] == 0)
        {
            AddressKind::Unnamed
        } else if path[0] == 0 {
            AddressKind::Abstract(ByteStr::from_bytes(&path[1..len]))
        } else {
            // linux adds a trailing NUL and counts it in the length, freebsd, netbsd
            // and qnx do not, and a caller may bind(2) without one either. unix(7)
            // gives the portable rule: strnlen(sun_path, len - offsetof(sun_path))
            let end = core::slice::memchr::memchr(0, &path[..len]).unwrap_or(len);
            AddressKind::Pathname(OsStr::from_bytes(&path[..end]).as_ref())
        }
    }
}

#[doc(cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin")))]
#[cfg(any(doc, target_os = "android", target_os = "linux", target_os = "cygwin"))]
#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
impl linux_ext::addr::SocketAddrExt for SocketAddr {
    fn as_abstract_name(&self) -> Option<&[u8]> {
        if let AddressKind::Abstract(name) = self.address() { Some(name.as_bytes()) } else { None }
    }

    fn from_abstract_name<N>(name: N) -> io::Result<Self>
    where
        N: AsRef<[u8]>,
    {
        let name = name.as_ref();
        let mut sockaddr = SocketAddr::default();

        if name.len() + 1 > SUN_PATH_MAX_LEN {
            const LEN_EXCEEDED_MSG: &'static str =
                cfg_select! {
                    any(target_os = "macos", target_os = "dragonfly") => {
                        "path must be shorter than SOCK_MAXADDRLEN - 2"
                    }
                    target_os = "netbsd" => "path must be shorter than UCHAR_MAX - 2",
                    _ => "path must be shorter than SUN_LEN - 1",
                };

            return Err(io::const_error!(io::ErrorKind::InvalidInput, LEN_EXCEEDED_MSG,));
        }

        sockaddr.addr[SUN_PATH_OFFSET + 1..SUN_PATH_OFFSET + 1 + name.len()].copy_from_slice(name);
        sockaddr.len = (SUN_PATH_OFFSET + 1 + name.len()) as libc::socklen_t;
        Ok(sockaddr)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.address() {
            AddressKind::Unnamed => write!(fmt, "(unnamed)"),
            AddressKind::Abstract(name) => write!(fmt, "{name:?} (abstract)"),
            AddressKind::Pathname(path) => write!(fmt, "{path:?} (pathname)"),
        }
    }
}
