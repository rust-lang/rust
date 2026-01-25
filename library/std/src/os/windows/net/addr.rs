use crate::bstr::ByteStr;
use crate::ffi::OsStr;
use crate::path::Path;
#[cfg(not(doc))]
use crate::sys::c::{AF_UNIX, SOCKADDR, SOCKADDR_UN};
use crate::sys::cvt_nz;
use crate::{fmt, io, mem, ptr};

#[cfg(not(doc))]
pub fn sockaddr_un(path: &Path) -> io::Result<(SOCKADDR_UN, usize)> {
    // SAFETY: All zeros is a valid representation for `sockaddr_un`.
    let mut addr: SOCKADDR_UN = unsafe { mem::zeroed() };
    addr.sun_family = AF_UNIX;

    // path to UTF-8 bytes
    let bytes = path
        .to_str()
        .ok_or(io::const_error!(io::ErrorKind::InvalidInput, "path must be valid UTF-8"))?
        .as_bytes();
    if bytes.len() >= addr.sun_path.len() {
        return Err(io::const_error!(io::ErrorKind::InvalidInput, "path too long"));
    }
    // SAFETY: `bytes` and `addr.sun_path` are not overlapping and
    // both point to valid memory.
    // NOTE: We zeroed the memory above, so the path is already null
    // terminated.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), addr.sun_path.as_mut_ptr().cast(), bytes.len())
    };

    let len = SUN_PATH_OFFSET + bytes.len() + 1;
    Ok((addr, len))
}
#[cfg(not(doc))]
const SUN_PATH_OFFSET: usize = mem::offset_of!(SOCKADDR_UN, sun_path);
pub struct SocketAddr {
    #[cfg(not(doc))]
    pub(super) addr: SOCKADDR_UN,
    pub(super) len: u32, // Use u32 here as same as libc::socklen_t
}
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.address() {
            AddressKind::Unnamed => write!(fmt, "(unnamed)"),
            AddressKind::Abstract(name) => write!(fmt, "{name:?} (abstract)"),
            AddressKind::Pathname(path) => write!(fmt, "{path:?} (pathname)"),
        }
    }
}

impl SocketAddr {
    #[cfg(not(doc))]
    pub(super) fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(*mut SOCKADDR, *mut i32) -> i32,
    {
        unsafe {
            let mut addr: SOCKADDR_UN = mem::zeroed();
            let mut len = mem::size_of::<SOCKADDR_UN>() as i32;
            cvt_nz(f(&raw mut addr as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }
    #[cfg(not(doc))]
    pub(super) fn from_parts(addr: SOCKADDR_UN, len: i32) -> io::Result<SocketAddr> {
        if addr.sun_family != AF_UNIX {
            Err(io::const_error!(io::ErrorKind::InvalidInput, "invalid address family"))
        } else if len < SUN_PATH_OFFSET as _ || len > mem::size_of::<SOCKADDR_UN>() as _ {
            Err(io::const_error!(io::ErrorKind::InvalidInput, "invalid address length"))
        } else {
            Ok(SocketAddr { addr, len: len as _ })
        }
    }

    /// Returns the contents of this address if it is a `pathname` address.
    ///
    /// # Examples
    ///
    /// With a pathname:
    ///
    /// ```no_run
    /// use std::os::windows::net::UnixListener;
    /// use std::path::Path;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixListener::bind("/tmp/sock")?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     assert_eq!(addr.as_pathname(), Some(Path::new("/tmp/sock")));
    ///     Ok(())
    /// }
    /// ```
    pub fn as_pathname(&self) -> Option<&Path> {
        if let AddressKind::Pathname(path) = self.address() { Some(path) } else { None }
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
    /// use std::os::windows::net::SocketAddr;
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
    /// use std::os::windows::net::SocketAddr;
    ///
    /// assert!(SocketAddr::from_pathname("/path/with/\0/bytes").is_err());
    /// ```
    pub fn from_pathname<P>(path: P) -> io::Result<SocketAddr>
    where
        P: AsRef<Path>,
    {
        sockaddr_un(path.as_ref()).map(|(addr, len)| SocketAddr { addr, len: len as _ })
    }
    fn address(&self) -> AddressKind<'_> {
        let len = self.len as usize - SUN_PATH_OFFSET;
        let path = unsafe { mem::transmute::<&[i8], &[u8]>(&self.addr.sun_path) };

        if len == 0 {
            AddressKind::Unnamed
        } else if self.addr.sun_path[0] == 0 {
            AddressKind::Abstract(ByteStr::from_bytes(&path[1..len]))
        } else {
            AddressKind::Pathname(unsafe {
                OsStr::from_encoded_bytes_unchecked(&path[..len - 1]).as_ref()
            })
        }
    }

    /// Returns `true` if the address is unnamed.
    ///
    /// # Examples
    ///
    /// A named address:
    ///
    /// ```no_run
    /// use std::os::windows::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixListener::bind("/tmp/sock")?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     assert_eq!(addr.is_unnamed(), false);
    ///     Ok(())
    /// }
    /// ```
    pub fn is_unnamed(&self) -> bool {
        matches!(self.address(), AddressKind::Unnamed)
    }
}
enum AddressKind<'a> {
    Unnamed,
    Pathname(&'a Path),
    Abstract(&'a ByteStr),
}
