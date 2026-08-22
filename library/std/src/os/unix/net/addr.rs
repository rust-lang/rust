use crate::bstr::ByteStr;
use crate::ffi::OsStr;
#[cfg(any(doc, target_os = "android", target_os = "linux", target_os = "cygwin"))]
use crate::os::net::linux_ext;
use crate::os::unix::ffi::OsStrExt;
use crate::path::Path;
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

// Offset to `libc::sockaddr_un.sun_path`
const SUN_PATH_OFFSET: usize = mem::offset_of!(libc::sockaddr_un, sun_path);
// Max size of `libc::sockaddr_un.sun_path`
const SUN_PATH_MAX_LEN: usize = SOCK_MAX_SIZE - SUN_PATH_OFFSET;

pub(super) fn sockaddr_un(path: &Path) -> io::Result<([u8; SOCK_MAX_SIZE], libc::socklen_t)> {
    // SAFETY: All zeros is a valid representation for `sockaddr_un`.
    let mut addr: [u8; SOCK_MAX_SIZE] = [0; SOCK_MAX_SIZE];

    let sun_family_bytes = (libc::AF_UNIX as libc::sa_family_t).to_ne_bytes();
    // SAFETY: `sun_family_bytes` and `addr.sun_family` are not overlapping and
    // both point to valid memory.
    // NOTE: On Linux, the struct definition for `libc::sockaddr_un` has two
    // fields a u16 `sun_family`, and then the flexible array sized `sun_path`.
    // On BSD, however, the struct definition for `libc::sockaddr_un` has three
    // different fields a u8 `sun_len`, a u8 `sun_family`, and then the flexible
    // array `sun_path`. Because `sun_family` could be treated as a u8 or u16, we
    // write the bytes of `libc::AF_UNIX` into the appropriate `sun_family` location
    // in native endian order.
    unsafe {
        ptr::copy_nonoverlapping(
            sun_family_bytes.as_ptr(),
            addr.as_mut_ptr().byte_add(mem::offset_of!(libc::sockaddr_un, sun_family)),
            sun_family_bytes.len(),
        )
    };

    let bytes = path.as_os_str().as_bytes();

    if bytes.contains(&0) {
        return Err(io::const_error!(
            io::ErrorKind::InvalidInput,
            "paths must not contain interior null bytes",
        ));
    }

    if bytes.len() >= SUN_PATH_MAX_LEN {
        cfg_if::cfg_if! {
            if #[cfg(any(target_os = "macos", target_os = "dragonfly"))] {
                return Err(io::const_error!(
                    io::ErrorKind::InvalidInput,
                    "path must be shorter than SOCK_MAXADDRLEN - 1",
                ));
            } else if #[cfg(any(target_os = "netbsd"))] {
                return Err(io::const_error!(
                    io::ErrorKind::InvalidInput,
                    "path must be shorter than UCHAR_MAX - 1",
                ));
            } else {
                return Err(io::const_error!(
                    io::ErrorKind::InvalidInput,
                    "path must be shorter than SUN_LEN",
                ));
            }
        }
    }
    // SAFETY: `bytes` and `addr.sun_path` are not overlapping and
    // both point to valid memory.
    // NOTE: We zeroed the memory above, so the path is already null
    // terminated.
    unsafe {
        ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            addr.as_mut_ptr().byte_add(SUN_PATH_OFFSET),
            bytes.len(),
        )
    };

    let mut len = SUN_PATH_OFFSET + bytes.len();
    #[cfg(any(
        target_os = "dragonfly",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
    ))]
    {
        // For these platforms, `sockaddr_un` has a `sun_len` (u8) field that should be initialized with a value
        // (nul byte excluding)
        addr[mem::offset_of!(libc::sockaddr_un, sun_len)] = len as u8;
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

    #[cfg(target_os = "freebsd")]
    {
        // For these platforms, `sockaddr_un` has a `sun_len` (u8) field that should be initialized with a value
        // (nul byte including)
        addr[mem::offset_of!(libc::sockaddr_un, sun_len)] = len as u8;
    }

    Ok((addr, len as libc::socklen_t))
}

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
    /// Heap allocated box that contains full size of what `libc::sockaddr_un`
    /// could be (as `sun_path` field defined for `sockaddr_un` does not represent
    /// the maximum path length of a Unix Domain socket name)
    pub(super) addr: [u8; SOCK_MAX_SIZE],
}

impl SocketAddr {
    pub(super) fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int,
    {
        unsafe {
            let mut addr: libc::sockaddr_un = mem::zeroed();
            let mut len = size_of::<libc::sockaddr_un>() as libc::socklen_t;
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
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "file descriptor did not correspond to a Unix socket",
            ));
        }

        let mut sockaddr: [u8; SOCK_MAX_SIZE] = [0; SOCK_MAX_SIZE];
        let addr_ptr = ptr::addr_of!(addr) as *const u8;
        cfg_if::cfg_if! {
            if #[cfg(all(unix, any(target_os = "macos", target_os = "freebsd", target_os = "openbsd", target_os = "netbsd", target_os = "dragonfly")))] {
                // SAFETY: `addr_ptr` and `sockaddr` are not overlapping and
                // both point to valid memory.
                // NOTE: We zeroed the memory above, so the `.sun_path` is already null
                // terminated.
                unsafe {
                    // BSD platforms have a `sun_len` field for their `sockaddr_un` struct
                    // which tells us the size of the socket address
                    ptr::copy_nonoverlapping(addr_ptr, sockaddr.as_mut_ptr(), addr.sun_len as usize)
                };
            } else {
                // SAFETY: `addr` and `sockaddr` are not overlapping and
                // both point to valid memory.
                // NOTE: We zeroed the memory above, so the path is already null
                // terminated.
                unsafe {
                    // The `SOCK_MAX_SIZE` internally is the sizeof(libc::sockaddr_un)
                    ptr::copy_nonoverlapping(addr_ptr, sockaddr.as_mut_ptr(), SOCK_MAX_SIZE)
                };
            }
        };
        Ok(SocketAddr { len, addr: sockaddr })
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
        sockaddr_un(path.as_ref()).map(|(addr, len)| SocketAddr { len, addr })
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
        unsafe {
            let mut addr: libc::sockaddr_un = mem::zeroed();
            addr.sun_family = libc::AF_UNIX as libc::sa_family_t;

            if name.len() + 1 > addr.sun_path.len() {
                return Err(io::const_error!(
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
            AddressKind::Abstract(name) => write!(fmt, "{name:?} (abstract)"),
            AddressKind::Pathname(path) => write!(fmt, "{path:?} (pathname)"),
        }
    }
}
