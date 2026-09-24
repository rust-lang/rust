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

enum AddressKind<'a> {
    Unnamed,
    Pathname(&'a Path),
    Abstract(&'a ByteStr),
}

/// The max socket address size in bytes
const SOCK_MAX_SIZE: usize = cfg_select! {
    any(target_vendor = "apple", target_os = "dragonfly") => {
        // Apple and DragonFly utilize `SOCK_MAXADDRLEN` to define
        // the maximum size that the `sockaddr_un` struct could be.
        // SOCK_MAXADDRLEN = 255 on these platforms, and it's based on
        // sizeof(sa_len) (a u8) + sizeof(sun_family) (a u8) + 253 bytes
        // (excluding nul). We add +1 for nul byte just in case.
        // See https://github.com/rust-lang/rust/issues/160684
        libc::SOCK_MAXADDRLEN as usize + 1
    }
    any(target_os = "netbsd") => {
        // NetBSD uses `UCHAR_MAX` (essentially a `u8::MAX`) + 1 as defined by `sockaddr_big` here:
        // https://github.com/IIJ-NetBSD/netbsd-src/blob/cc2b5d89fa44/sys/sys/socket.h#L272-L287
        libc::c_uchar::MAX as usize + 1
    }
    _ => {
        // All other platforms max socket address size is based on
        // the size of their sockaddr_un struct
        size_of::<libc::sockaddr_un>()
    }
};

// A check to see if it's safe to downcast SOCK_MAX_SIZE to libc::socklen_t
// (a u32)
const _: () = assert!(SOCK_MAX_SIZE <= libc::socklen_t::MAX as usize);

/// Offset to `libc::sockaddr_un.sun_family` in bytes
const SUN_FAMILY_OFFSET: usize = mem::offset_of!(libc::sockaddr_un, sun_family);
/// Offset to `libc::sockaddr_un.sun_path` in bytes
const SUN_PATH_OFFSET: usize = mem::offset_of!(libc::sockaddr_un, sun_path);
/// This represents the maximum number of bytes + 1 allowed to be
/// stored in the socket path (e.g. 254 for NetBSD because it allows 253
/// valid characters for its path, 104 for FreeBSD because it allows up to
/// 103 characters for its path)
pub(crate) const SUN_PATH_MAX_LEN: usize = SOCK_MAX_SIZE - SUN_PATH_OFFSET;
/// Platform-dependent error message when user provides a longer socket address path
/// than what the OS allows for.
const LEN_EXCEEDED_MSG: &'static str =
    cfg_select! {
        any(target_os = "macos", target_os = "dragonfly") => {
            "path must be shorter than SOCK_MAXADDRLEN - 1"
        }
        target_os = "netbsd" => "path must be shorter than UCHAR_MAX - 1",
        _ => "path must be shorter than SUN_LEN",
    };

/// An internal helper struct that provides a `len` field describing the actual filled size of the socket
/// address and a `buf` field that contains an internal platform-agnostic struct of `libc::sockaddr_un`.
///
/// On BSD platforms, `libc::sockaddr_un` is defined as:
/// struct sockaddr_un {
///     sun_len: u8,
///     sun_family: sa_family_t,
///     sun_path: [c_char; 104],
/// }
///
/// On Linux and other platforms, `libc::sockaddr_un` is defined as:
/// struct sockaddr_un {
///     sun_family: sa_family_t,
///     sun_path: [c_char; 108],
/// }
///
/// Note:
/// * `sa_family_t` is a u8 on BSD platforms and a u16 on Linux/other platforms.
/// * The fix-sized array value for `sun_path` field could be different across
/// other platforms (for Linux it's 108, but this may be a different on other platforms).
///
/// Although `sockaddr_un.sun_path` is restricted to 104 characters on BSD platforms,
/// DragonFlyBSD/NetBSD/Apple actually allow `sun_path` to hold up to 253 non-nul bytes.
/// Therefore, this `SockaddrUn` struct aims to hold a buffer that contains the maximum
/// socket address size for each platform.
#[derive(Clone)]
#[repr(C)]
pub(super) struct SockaddrBuf {
    /// Size of the socket address, `sun_family` and `sun_path`
    /// fields from `libc::sockaddr_un` included
    len: libc::socklen_t,
    /// Stack allocated buffer that contains full size of what `libc::sockaddr_un`
    /// could be (as `sun_path` field defined for `sockaddr_un` does not represent
    /// the maximum path length of a Unix Domain socket name)
    buf: [u8; SOCK_MAX_SIZE],
    /// Make it sound to cast the struct to/from a `sockaddr_un`
    align: [libc::sockaddr_un; 0],
}

impl SockaddrBuf {
    /* API for use in Rust */

    /// This returns an empty Unix Domain socket address with a
    /// length value containing the current size of the `sockaddr_un`
    pub(super) fn default() -> SockaddrBuf {
        let mut sockaddr_un: [u8; SOCK_MAX_SIZE] = [0; SOCK_MAX_SIZE];
        let sun_family = (libc::AF_UNIX as libc::sa_family_t).to_ne_bytes();
        sockaddr_un[SUN_FAMILY_OFFSET..SUN_FAMILY_OFFSET + size_of::<libc::sa_family_t>()]
            .copy_from_slice(&sun_family);

        // Size of the socket address is not 0 since we have an initialized
        // `sun_family`/`sun_len`
        let len = SUN_PATH_OFFSET as libc::socklen_t;

        SockaddrBuf { len, buf: sockaddr_un, align: [] }
    }

    /// Extracts the `sun_path` value from the socket address buffer.
    ///
    /// On QNX, NTO, DragonFlyBSD, NetBSD, OpenBSD, FreeBSD, and Apple family
    /// the nul byte is not part of `SockaddrBuf.len`
    ///
    /// On all other platforms (e.g. Linux), the nul byte is a part of `SockaddrBuf.len`
    fn path(&self) -> &[u8] {
        &self.buf[SUN_PATH_OFFSET..self.len as usize]
    }

    /// Sets the socket address path for `sockaddr_un` in `SocketaddrBuf`.
    fn set_path(&mut self, bytes: &[u8]) -> io::Result<()> {
        if core::slice::memchr::memchr(0, bytes).is_some() {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "paths must not contain interior null bytes",
            ));
        }

        if bytes.len() >= SUN_PATH_MAX_LEN {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, LEN_EXCEEDED_MSG));
        }

        self.buf[SUN_PATH_OFFSET..SUN_PATH_OFFSET + bytes.len()].copy_from_slice(bytes);
        self.buf[SUN_PATH_OFFSET + bytes.len()] = 0;

        let mut len = SUN_PATH_OFFSET + bytes.len();
        #[cfg(any(
            target_os = "dragonfly",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "freebsd"
        ))]
        {
            const _: () = assert!(SUN_PATH_MAX_LEN as usize <= u8::MAX as usize);
            // For these platforms, `sockaddr_un` has a `sun_len` (u8) field that should be initialized with a value
            // (nul byte excluding)
            self.buf[mem::offset_of!(libc::sockaddr_un, sun_len)] = len as u8;
        }

        match bytes.get(0) {
            Some(&0) | None => {}
            Some(_) => {
                // on QNX7.1 and QNX8 the `len` value returned by the SUN_LEN
                // macro in its libc does not include the null byte in the count so
                // don't add it here to match what a C program passes to bind(2) and
                // similar functions
                // For BSD-based platforms, they do not count the nul terminating byte
                // in the address length
                if cfg!(not(any(
                    target_os = "qnx",
                    target_env = "nto71",
                    target_os = "dragonfly",
                    target_vendor = "apple",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "freebsd"
                ))) {
                    len += 1
                }
            }
        }

        // Even though len here is a `usize` and `libc::socklen_t` is a `u32`
        // our len value should be limited to whatever value a `u32` can hold
        self.len = len as libc::socklen_t;

        Ok(())
    }

    /// Extracts the `sun_path` value from the abstract socket address buffer (excludes initial
    /// nul byte).
    fn abstract_path(&self) -> &[u8] {
        &self.buf[SUN_PATH_OFFSET + 1..self.len as usize]
    }

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    /// Sets the abstract socket address path for `sockaddr_un` in `SocketaddrBuf`.
    /// Only used on Android, Linux, and Cygwin.
    fn set_abstract_path(&mut self, bytes: &[u8]) -> io::Result<()> {
        // Abstract socket address paths are not nul terminated!
        // (Hence the > instead of >=)
        // See Linux manpage: https://man7.org/linux/man-pages/man7/unix.7.html
        // "The socket's address in this namespace is given by the additional
        // bytes in sun_path that are covered by the specified length of the
        // address structure. (Null bytes in the name have no special
        // significance.)"
        if bytes.len() + 1 > SUN_PATH_MAX_LEN {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "path must be shorter than SUN_LEN - 1",
            ));
        }

        // +1 because the first byte in the sun_path for abstract sockets is a nul byte
        self.buf[SUN_PATH_OFFSET + 1..SUN_PATH_OFFSET + 1 + bytes.len()].copy_from_slice(bytes);
        self.len = (SUN_PATH_OFFSET + 1 + bytes.len()) as libc::socklen_t;

        Ok(())
    }

    /// Extracts the `sun_family` value from the socket address buffer.
    fn sun_family(&self) -> libc::sa_family_t {
        let sun_family_array = self.buf[SUN_FAMILY_OFFSET..SUN_FAMILY_OFFSET + size_of::<libc::sa_family_t>()]
        .try_into()
        .expect("Slice should have exactly the same number of bytes extracted as the size of libc::sa_family_t");
        libc::sa_family_t::from_ne_bytes(sun_family_array)
    }

    /* API for use in sockets in abstract namespace + interop with C */

    /// Types passed to libc calls that read a socket address.
    pub(super) fn as_libc_input(&self) -> (*const libc::sockaddr, libc::socklen_t) {
        (self.buf.as_ptr().cast(), self.len)
    }

    /// Types passed to libc calls that write a socket address. Length is first set to the
    /// max allowed.
    ///
    /// Note that using this API _must_ be followed by a call to `update_from_libc`.
    // Note that returning two raw pointers from a single `&mut self` function call rather than
    // two `&mut self -> *mut T` function calls is required to pass Miri with stacked borrows.
    pub(super) fn as_max_libc_output(&mut self) -> (*mut libc::sockaddr, *mut libc::socklen_t) {
        // Even though len here is a `usize` and `libc::socklen_t` is a `u32`
        // our len value should be limited to whatever value a `u32` can hold
        self.len = SOCK_MAX_SIZE as libc::socklen_t;

        (self.buf.as_mut_ptr().cast(), &mut self.len)
    }

    /// Some platforms encode things differently. Call this after a `libc` call to adjust length
    /// and validate as needed.
    pub(super) fn update_from_libc(&mut self) -> io::Result<()> {
        if cfg!(target_os = "openbsd") {
            // on OpenBSD, getsockname(2) doesn't re-adjust the len field of the socket address,
            // so it doesn't reflect the true len of the content. Figure out the length for ourselves.
            // https://marc.info/?l=openbsd-bugs&m=170105481926736&w=2
            let sun_path = self.path();

            if let Some(new_len) = core::slice::memchr::memchr(0, sun_path) {
                self.len = (new_len + SUN_PATH_OFFSET) as libc::socklen_t;
            }
        }

        if self.len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            self.len = SUN_PATH_OFFSET as libc::socklen_t; // i.e., zero-length address
        } else if self.sun_family() != libc::AF_UNIX as libc::sa_family_t {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "file descriptor did not correspond to a Unix socket",
            ));
        }

        Ok(())
    }
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
    pub(super) sock: SockaddrBuf,
}

impl SocketAddr {
    pub(super) fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(&mut SockaddrBuf) -> libc::c_int,
    {
        let mut sock = SockaddrBuf::default();
        cvt(f(&mut sock))?;
        sock.update_from_libc()?;
        Ok(SocketAddr { sock })
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

    /// Constructs a Unix Domain socket address from a given `Path`
    pub(super) fn from_path(path: &Path) -> io::Result<SocketAddr> {
        let bytes = path.as_os_str().as_bytes();
        let mut sock = SockaddrBuf::default();
        sock.set_path(bytes)?;
        Ok(SocketAddr { sock })
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
        let len = self.sock.len as usize - SUN_PATH_OFFSET;
        let path = self.sock.path();

        // macOS seems to return a len of 16 and a zeroed sun_path for unnamed addresses
        if len == 0
            || (cfg!(not(any(target_os = "linux", target_os = "android", target_os = "cygwin")))
                && path[0] == 0)
        {
            AddressKind::Unnamed
        } else if path[0] == 0 {
            AddressKind::Abstract(ByteStr::from_bytes(self.sock.abstract_path()))
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
        let mut sock = SockaddrBuf::default();
        sock.set_abstract_path(name)?;
        Ok(SocketAddr { sock })
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
