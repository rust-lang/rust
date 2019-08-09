#![stable(feature = "unix_socket", since = "1.10.0")]

//! Unix-specific networking functionality

#[cfg(unix)]
use libc;

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(not(unix))]
mod libc {
    pub use libc::c_int;
    pub type socklen_t = u32;
    pub struct sockaddr;
    #[derive(Clone)]
    pub struct sockaddr_un;
}

use crate::ascii;
use crate::ffi::OsStr;
use crate::fmt;
use crate::io::{self, Initializer, IoSlice, IoSliceMut};
use crate::mem;
use crate::net::{self, Shutdown};
use crate::os::unix::ffi::OsStrExt;
use crate::os::unix::io::{RawFd, AsRawFd, FromRawFd, IntoRawFd};
use crate::path::Path;
use crate::time::Duration;
use crate::sys::{self, cvt};
use crate::sys::net::Socket;
use crate::sys_common::{self, AsInner, FromInner, IntoInner};

#[cfg(any(target_os = "linux", target_os = "android",
          target_os = "dragonfly", target_os = "freebsd",
          target_os = "openbsd", target_os = "netbsd",
          target_os = "haiku"))]
use libc::MSG_NOSIGNAL;
#[cfg(not(any(target_os = "linux", target_os = "android",
              target_os = "dragonfly", target_os = "freebsd",
              target_os = "openbsd", target_os = "netbsd",
              target_os = "haiku")))]
const MSG_NOSIGNAL: libc::c_int = 0x0;

fn sun_path_offset(addr: &libc::sockaddr_un) -> usize {
    // Work with an actual instance of the type since using a null pointer is UB
    let base = addr as *const _ as usize;
    let path = &addr.sun_path as *const _ as usize;
    path - base
}

unsafe fn sockaddr_un(path: &Path) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    let mut addr: libc::sockaddr_un = mem::zeroed();
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;

    let bytes = path.as_os_str().as_bytes();

    if bytes.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                  "paths may not contain interior null bytes"));
    }

    if bytes.len() >= addr.sun_path.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                  "path must be shorter than SUN_LEN"));
    }
    for (dst, src) in addr.sun_path.iter_mut().zip(bytes.iter()) {
        *dst = *src as libc::c_char;
    }
    // null byte for pathname addresses is already there because we zeroed the
    // struct

    let mut len = sun_path_offset(&addr) + bytes.len();
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
///         println!("Couldn't bind: {:?}", e);
///         return
///     }
/// };
/// let addr = socket.local_addr().expect("Couldn't get local address");
/// ```
#[derive(Clone)]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct SocketAddr {
    addr: libc::sockaddr_un,
    len: libc::socklen_t,
}

impl SocketAddr {
    fn new<F>(f: F) -> io::Result<SocketAddr>
        where F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int
    {
        unsafe {
            let mut addr: libc::sockaddr_un = mem::zeroed();
            let mut len = mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
            cvt(f(&mut addr as *mut _ as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }

    fn from_parts(addr: libc::sockaddr_un, mut len: libc::socklen_t) -> io::Result<SocketAddr> {
        if len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            len = sun_path_offset(&addr) as libc::socklen_t;  // i.e., zero-length address
        } else if addr.sun_family != libc::AF_UNIX as libc::sa_family_t {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "file descriptor did not correspond to a Unix socket"));
        }

        Ok(SocketAddr {
            addr,
            len,
        })
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
    /// let socket = UnixListener::bind("/tmp/sock").unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.is_unnamed(), false);
    /// ```
    ///
    /// An unnamed address:
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let socket = UnixDatagram::unbound().unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.is_unnamed(), true);
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn is_unnamed(&self) -> bool {
        if let AddressKind::Unnamed = self.address() {
            true
        } else {
            false
        }
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
    /// let socket = UnixListener::bind("/tmp/sock").unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.as_pathname(), Some(Path::new("/tmp/sock")));
    /// ```
    ///
    /// Without a pathname:
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let socket = UnixDatagram::unbound().unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.as_pathname(), None);
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn as_pathname(&self) -> Option<&Path> {
        if let AddressKind::Pathname(path) = self.address() {
            Some(path)
        } else {
            None
        }
    }

    fn address(&self) -> AddressKind<'_> {
        let len = self.len as usize - sun_path_offset(&self.addr);
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

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.address() {
            AddressKind::Unnamed => write!(fmt, "(unnamed)"),
            AddressKind::Abstract(name) => write!(fmt, "{} (abstract)", AsciiEscaped(name)),
            AddressKind::Pathname(path) => write!(fmt, "{:?} (pathname)", path),
        }
    }
}

struct AsciiEscaped<'a>(&'a [u8]);

impl<'a> fmt::Display for AsciiEscaped<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "\"")?;
        for byte in self.0.iter().cloned().flat_map(ascii::escape_default) {
            write!(fmt, "{}", byte as char)?;
        }
        write!(fmt, "\"")
    }
}

/// A Unix stream socket.
///
/// # Examples
///
/// ```no_run
/// use std::os::unix::net::UnixStream;
/// use std::io::prelude::*;
///
/// let mut stream = UnixStream::connect("/path/to/my/socket").unwrap();
/// stream.write_all(b"hello world").unwrap();
/// let mut response = String::new();
/// stream.read_to_string(&mut response).unwrap();
/// println!("{}", response);
/// ```
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixStream(Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixStream {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixStream");
        builder.field("fd", self.0.as_inner());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        if let Ok(addr) = self.peer_addr() {
            builder.field("peer", &addr);
        }
        builder.finish()
    }
}

impl UnixStream {
    /// Connects to the socket named by `path`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = match UnixStream::connect("/tmp/sock") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't connect: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
        fn inner(path: &Path) -> io::Result<UnixStream> {
            unsafe {
                let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
                let (addr, len) = sockaddr_un(path)?;

                cvt(libc::connect(*inner.as_inner(), &addr as *const _ as *const _, len))?;
                Ok(UnixStream(inner))
            }
        }
        inner(path.as_ref())
    }

    /// Creates an unnamed pair of connected sockets.
    ///
    /// Returns two `UnixStream`s which are connected to each other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let (sock1, sock2) = match UnixStream::pair() {
    ///     Ok((sock1, sock2)) => (sock1, sock2),
    ///     Err(e) => {
    ///         println!("Couldn't create a pair of sockets: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        let (i1, i2) = Socket::new_pair(libc::AF_UNIX, libc::SOCK_STREAM)?;
        Ok((UnixStream(i1), UnixStream(i2)))
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixStream` is a reference to the same stream that this
    /// object references. Both handles will read and write the same stream of
    /// data, and options set on one stream will be propagated to the other
    /// stream.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let sock_copy = socket.try_clone().expect("Couldn't clone socket");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        self.0.duplicate().map(UnixStream)
    }

    /// Returns the socket address of the local half of this connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(*self.0.as_inner(), addr, len) })
    }

    /// Returns the socket address of the remote half of this connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let addr = socket.peer_addr().expect("Couldn't get peer address");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getpeername(*self.0.as_inner(), addr, len) })
    }

    /// Sets the read timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`read`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`Err`]: ../../../../std/result/enum.Result.html#variant.Err
    /// [`read`]: ../../../../std/io/trait.Read.html#tymethod.read
    /// [`Duration`]: ../../../../std/time/struct.Duration.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_read_timeout(Some(Duration::new(1, 0))).expect("Couldn't set read timeout");
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let result = socket.set_read_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_RCVTIMEO)
    }

    /// Sets the write timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`write`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is
    /// passed to this method.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`Err`]: ../../../../std/result/enum.Result.html#variant.Err
    /// [`write`]: ../../../../std/io/trait.Write.html#tymethod.write
    /// [`Duration`]: ../../../../std/time/struct.Duration.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_write_timeout(Some(Duration::new(1, 0))).expect("Couldn't set write timeout");
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::net::UdpSocket;
    /// use std::time::Duration;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").unwrap();
    /// let result = socket.set_write_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_SNDTIMEO)
    }

    /// Returns the read timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_read_timeout(Some(Duration::new(1, 0))).expect("Couldn't set read timeout");
    /// assert_eq!(socket.read_timeout().unwrap(), Some(Duration::new(1, 0)));
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_RCVTIMEO)
    }

    /// Returns the write timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_write_timeout(Some(Duration::new(1, 0))).expect("Couldn't set write timeout");
    /// assert_eq!(socket.write_timeout().unwrap(), Some(Duration::new(1, 0)));
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_SNDTIMEO)
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_nonblocking(true).expect("Couldn't set nonblocking");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// if let Ok(Some(err)) = socket.take_error() {
    ///     println!("Got error: {:?}", err);
    /// }
    /// ```
    ///
    /// # Platform specific
    /// On Redox this always returns `None`.
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Shuts down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O calls on the
    /// specified portions to immediately return with an appropriate value
    /// (see the documentation of [`Shutdown`]).
    ///
    /// [`Shutdown`]: ../../../../std/net/enum.Shutdown.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::net::Shutdown;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.shutdown(Shutdown::Both).expect("shutdown function failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl io::Read for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        io::Read::read(&mut &*self, buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        io::Read::read_vectored(&mut &*self, bufs)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> io::Read for &'a UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl io::Write for UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::Write::write(&mut &*self, buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        io::Write::write_vectored(&mut &*self, bufs)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::Write::flush(&mut &*self)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> io::Write for &'a UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixStream {
    fn as_raw_fd(&self) -> RawFd {
        *self.0.as_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixStream {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixStream {
        UnixStream(Socket::from_inner(fd))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixStream {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::TcpStream {
    fn as_raw_fd(&self) -> RawFd { *self.as_inner().socket().as_inner() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::TcpListener {
    fn as_raw_fd(&self) -> RawFd { *self.as_inner().socket().as_inner() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::UdpSocket {
    fn as_raw_fd(&self) -> RawFd { *self.as_inner().socket().as_inner() }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::TcpStream {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpStream {
        let socket = sys::net::Socket::from_inner(fd);
        net::TcpStream::from_inner(sys_common::net::TcpStream::from_inner(socket))
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::TcpListener {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpListener {
        let socket = sys::net::Socket::from_inner(fd);
        net::TcpListener::from_inner(sys_common::net::TcpListener::from_inner(socket))
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::UdpSocket {
    unsafe fn from_raw_fd(fd: RawFd) -> net::UdpSocket {
        let socket = sys::net::Socket::from_inner(fd);
        net::UdpSocket::from_inner(sys_common::net::UdpSocket::from_inner(socket))
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::TcpStream {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_socket().into_inner()
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::TcpListener {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_socket().into_inner()
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::UdpSocket {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_socket().into_inner()
    }
}

/// A structure representing a Unix domain socket server.
///
/// # Examples
///
/// ```no_run
/// use std::thread;
/// use std::os::unix::net::{UnixStream, UnixListener};
///
/// fn handle_client(stream: UnixStream) {
///     // ...
/// }
///
/// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
///
/// // accept connections and process them, spawning a new thread for each one
/// for stream in listener.incoming() {
///     match stream {
///         Ok(stream) => {
///             /* connection succeeded */
///             thread::spawn(|| handle_client(stream));
///         }
///         Err(err) => {
///             /* connection failed */
///             break;
///         }
///     }
/// }
/// ```
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixListener(Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixListener {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixListener");
        builder.field("fd", self.0.as_inner());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        builder.finish()
    }
}

impl UnixListener {
    /// Creates a new `UnixListener` bound to the specified socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = match UnixListener::bind("/path/to/the/socket") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't connect: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
        fn inner(path: &Path) -> io::Result<UnixListener> {
            unsafe {
                let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
                let (addr, len) = sockaddr_un(path)?;

                cvt(libc::bind(*inner.as_inner(), &addr as *const _ as *const _, len as _))?;
                cvt(libc::listen(*inner.as_inner(), 128))?;

                Ok(UnixListener(inner))
            }
        }
        inner(path.as_ref())
    }

    /// Accepts a new incoming connection to this listener.
    ///
    /// This function will block the calling thread until a new Unix connection
    /// is established. When established, the corresponding [`UnixStream`] and
    /// the remote peer's address will be returned.
    ///
    /// [`UnixStream`]: ../../../../std/os/unix/net/struct.UnixStream.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// match listener.accept() {
    ///     Ok((socket, addr)) => println!("Got a client: {:?}", addr),
    ///     Err(e) => println!("accept function failed: {:?}", e),
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        let mut storage: libc::sockaddr_un = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as libc::socklen_t;
        let sock = self.0.accept(&mut storage as *mut _ as *mut _, &mut len)?;
        let addr = SocketAddr::from_parts(storage, len)?;
        Ok((UnixStream(sock), addr))
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixListener` is a reference to the same socket that this
    /// object references. Both handles can be used to accept incoming
    /// connections and options set on one listener will affect the other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// let listener_copy = listener.try_clone().expect("try_clone failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
    }

    /// Returns the local socket address of this listener.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// let addr = listener.local_addr().expect("Couldn't get local address");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(*self.0.as_inner(), addr, len) })
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// listener.set_nonblocking(true).expect("Couldn't set non blocking");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/tmp/sock").unwrap();
    ///
    /// if let Ok(Some(err)) = listener.take_error() {
    ///     println!("Got error: {:?}", err);
    /// }
    /// ```
    ///
    /// # Platform specific
    /// On Redox this always returns `None`.
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Returns an iterator over incoming connections.
    ///
    /// The iterator will never return [`None`] and will also not yield the
    /// peer's [`SocketAddr`] structure.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`SocketAddr`]: struct.SocketAddr.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::thread;
    /// use std::os::unix::net::{UnixStream, UnixListener};
    ///
    /// fn handle_client(stream: UnixStream) {
    ///     // ...
    /// }
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// for stream in listener.incoming() {
    ///     match stream {
    ///         Ok(stream) => {
    ///             thread::spawn(|| handle_client(stream));
    ///         }
    ///         Err(err) => {
    ///             break;
    ///         }
    ///     }
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixListener {
    fn as_raw_fd(&self) -> RawFd {
        *self.0.as_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixListener {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixListener {
        UnixListener(Socket::from_inner(fd))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixListener {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> IntoIterator for &'a UnixListener {
    type Item = io::Result<UnixStream>;
    type IntoIter = Incoming<'a>;

    fn into_iter(self) -> Incoming<'a> {
        self.incoming()
    }
}

/// An iterator over incoming connections to a [`UnixListener`].
///
/// It will never return [`None`].
///
/// [`None`]: ../../../../std/option/enum.Option.html#variant.None
/// [`UnixListener`]: struct.UnixListener.html
///
/// # Examples
///
/// ```no_run
/// use std::thread;
/// use std::os::unix::net::{UnixStream, UnixListener};
///
/// fn handle_client(stream: UnixStream) {
///     // ...
/// }
///
/// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
///
/// for stream in listener.incoming() {
///     match stream {
///         Ok(stream) => {
///             thread::spawn(|| handle_client(stream));
///         }
///         Err(err) => {
///             break;
///         }
///     }
/// }
/// ```
#[derive(Debug)]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct Incoming<'a> {
    listener: &'a UnixListener,
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<UnixStream>;

    fn next(&mut self) -> Option<io::Result<UnixStream>> {
        Some(self.listener.accept().map(|s| s.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::max_value(), None)
    }
}

/// A Unix datagram socket.
///
/// # Examples
///
/// ```no_run
/// use std::os::unix::net::UnixDatagram;
///
/// let socket = UnixDatagram::bind("/path/to/my/socket").unwrap();
/// socket.send_to(b"hello world", "/path/to/other/socket").unwrap();
/// let mut buf = [0; 100];
/// let (count, address) = socket.recv_from(&mut buf).unwrap();
/// println!("socket {:?} sent {:?}", address, &buf[..count]);
/// ```
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixDatagram(Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixDatagram {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixDatagram");
        builder.field("fd", self.0.as_inner());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        if let Ok(addr) = self.peer_addr() {
            builder.field("peer", &addr);
        }
        builder.finish()
    }
}

impl UnixDatagram {
    /// Creates a Unix datagram socket bound to the given path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = match UnixDatagram::bind("/path/to/the/socket") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't bind: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixDatagram> {
        fn inner(path: &Path) -> io::Result<UnixDatagram> {
            unsafe {
                let socket = UnixDatagram::unbound()?;
                let (addr, len) = sockaddr_un(path)?;

                cvt(libc::bind(*socket.0.as_inner(), &addr as *const _ as *const _, len as _))?;

                Ok(socket)
            }
        }
        inner(path.as_ref())
    }

    /// Creates a Unix Datagram socket which is not bound to any address.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = match UnixDatagram::unbound() {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't unbound: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn unbound() -> io::Result<UnixDatagram> {
        let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_DGRAM)?;
        Ok(UnixDatagram(inner))
    }

    /// Creates an unnamed pair of connected sockets.
    ///
    /// Returns two `UnixDatagrams`s which are connected to each other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let (sock1, sock2) = match UnixDatagram::pair() {
    ///     Ok((sock1, sock2)) => (sock1, sock2),
    ///     Err(e) => {
    ///         println!("Couldn't unbound: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn pair() -> io::Result<(UnixDatagram, UnixDatagram)> {
        let (i1, i2) = Socket::new_pair(libc::AF_UNIX, libc::SOCK_DGRAM)?;
        Ok((UnixDatagram(i1), UnixDatagram(i2)))
    }

    /// Connects the socket to the specified address.
    ///
    /// The [`send`] method may be used to send data to the specified address.
    /// [`recv`] and [`recv_from`] will only receive data from that address.
    ///
    /// [`send`]: #method.send
    /// [`recv`]: #method.recv
    /// [`recv_from`]: #method.recv_from
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// match sock.connect("/path/to/the/socket") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't connect: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn connect<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        fn inner(d: &UnixDatagram, path: &Path) -> io::Result<()> {
            unsafe {
                let (addr, len) = sockaddr_un(path)?;

                cvt(libc::connect(*d.0.as_inner(), &addr as *const _ as *const _, len))?;

                Ok(())
            }
        }
        inner(self, path.as_ref())
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixDatagram` is a reference to the same socket that this
    /// object references. Both handles can be used to accept incoming
    /// connections and options set on one side will affect the other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::bind("/path/to/the/socket").unwrap();
    ///
    /// let sock_copy = sock.try_clone().expect("try_clone failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixDatagram> {
        self.0.duplicate().map(UnixDatagram)
    }

    /// Returns the address of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::bind("/path/to/the/socket").unwrap();
    ///
    /// let addr = sock.local_addr().expect("Couldn't get local address");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(*self.0.as_inner(), addr, len) })
    }

    /// Returns the address of this socket's peer.
    ///
    /// The [`connect`] method will connect the socket to a peer.
    ///
    /// [`connect`]: #method.connect
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.connect("/path/to/the/socket").unwrap();
    ///
    /// let addr = sock.peer_addr().expect("Couldn't get peer address");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getpeername(*self.0.as_inner(), addr, len) })
    }

    /// Receives data from the socket.
    ///
    /// On success, returns the number of bytes read and the address from
    /// whence the data came.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// let mut buf = vec![0; 10];
    /// match sock.recv_from(buf.as_mut_slice()) {
    ///     Ok((size, sender)) => println!("received {} bytes from {:?}", size, sender),
    ///     Err(e) => println!("recv_from function failed: {:?}", e),
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let mut count = 0;
        let addr = SocketAddr::new(|addr, len| {
            unsafe {
                count = libc::recvfrom(*self.0.as_inner(),
                                       buf.as_mut_ptr() as *mut _,
                                       buf.len(),
                                       0,
                                       addr,
                                       len);
                if count > 0 {
                    1
                } else if count == 0 {
                    0
                } else {
                    -1
                }
            }
        })?;

        Ok((count as usize, addr))
    }

    /// Receives data from the socket.
    ///
    /// On success, returns the number of bytes read.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::bind("/path/to/the/socket").unwrap();
    /// let mut buf = vec![0; 10];
    /// sock.recv(buf.as_mut_slice()).expect("recv function failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    /// Sends data on the socket to the specified address.
    ///
    /// On success, returns the number of bytes written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.send_to(b"omelette au fromage", "/some/sock").expect("send_to function failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn send_to<P: AsRef<Path>>(&self, buf: &[u8], path: P) -> io::Result<usize> {
        fn inner(d: &UnixDatagram, buf: &[u8], path: &Path) -> io::Result<usize> {
            unsafe {
                let (addr, len) = sockaddr_un(path)?;

                let count = cvt(libc::sendto(*d.0.as_inner(),
                                             buf.as_ptr() as *const _,
                                             buf.len(),
                                             MSG_NOSIGNAL,
                                             &addr as *const _ as *const _,
                                             len))?;
                Ok(count as usize)
            }
        }
        inner(self, buf, path.as_ref())
    }

    /// Sends data on the socket to the socket's peer.
    ///
    /// The peer address may be set by the `connect` method, and this method
    /// will return an error if the socket has not already been connected.
    ///
    /// On success, returns the number of bytes written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.connect("/some/sock").expect("Couldn't connect");
    /// sock.send(b"omelette au fromage").expect("send_to function failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    /// Sets the read timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`recv`] and [`recv_from`] calls will
    /// block indefinitely. An [`Err`] is returned if the zero [`Duration`]
    /// is passed to this method.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`Err`]: ../../../../std/result/enum.Result.html#variant.Err
    /// [`recv`]: #method.recv
    /// [`recv_from`]: #method.recv_from
    /// [`Duration`]: ../../../../std/time/struct.Duration.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    /// use std::time::Duration;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.set_read_timeout(Some(Duration::new(1, 0))).expect("set_read_timeout function failed");
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::os::unix::net::UnixDatagram;
    /// use std::time::Duration;
    ///
    /// let socket = UnixDatagram::unbound().unwrap();
    /// let result = socket.set_read_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_RCVTIMEO)
    }

    /// Sets the write timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`send`] and [`send_to`] calls will
    /// block indefinitely. An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`send`]: #method.send
    /// [`send_to`]: #method.send_to
    /// [`Duration`]: ../../../../std/time/struct.Duration.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    /// use std::time::Duration;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.set_write_timeout(Some(Duration::new(1, 0)))
    ///     .expect("set_write_timeout function failed");
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::os::unix::net::UnixDatagram;
    /// use std::time::Duration;
    ///
    /// let socket = UnixDatagram::unbound().unwrap();
    /// let result = socket.set_write_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_SNDTIMEO)
    }

    /// Returns the read timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    /// use std::time::Duration;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.set_read_timeout(Some(Duration::new(1, 0))).expect("set_read_timeout function failed");
    /// assert_eq!(sock.read_timeout().unwrap(), Some(Duration::new(1, 0)));
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_RCVTIMEO)
    }

    /// Returns the write timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    /// use std::time::Duration;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.set_write_timeout(Some(Duration::new(1, 0)))
    ///     .expect("set_write_timeout function failed");
    /// assert_eq!(sock.write_timeout().unwrap(), Some(Duration::new(1, 0)));
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_SNDTIMEO)
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.set_nonblocking(true).expect("set_nonblocking function failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// if let Ok(Some(err)) = sock.take_error() {
    ///     println!("Got error: {:?}", err);
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Shut down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O calls on the
    /// specified portions to immediately return with an appropriate value
    /// (see the documentation of [`Shutdown`]).
    ///
    /// [`Shutdown`]: ../../../../std/net/enum.Shutdown.html
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixDatagram;
    /// use std::net::Shutdown;
    ///
    /// let sock = UnixDatagram::unbound().unwrap();
    /// sock.shutdown(Shutdown::Both).expect("shutdown function failed");
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixDatagram {
    fn as_raw_fd(&self) -> RawFd {
        *self.0.as_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixDatagram {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixDatagram {
        UnixDatagram(Socket::from_inner(fd))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixDatagram {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner()
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod test {
    use crate::thread;
    use crate::io::{self, ErrorKind};
    use crate::io::prelude::*;
    use crate::time::Duration;
    use crate::sys_common::io::test::tmpdir;

    use super::*;

    macro_rules! or_panic {
        ($e:expr) => {
            match $e {
                Ok(e) => e,
                Err(e) => panic!("{}", e),
            }
        }
    }

    #[test]
    fn basic() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");
        let msg1 = b"hello";
        let msg2 = b"world!";

        let listener = or_panic!(UnixListener::bind(&socket_path));
        let thread = thread::spawn(move || {
            let mut stream = or_panic!(listener.accept()).0;
            let mut buf = [0; 5];
            or_panic!(stream.read(&mut buf));
            assert_eq!(&msg1[..], &buf[..]);
            or_panic!(stream.write_all(msg2));
        });

        let mut stream = or_panic!(UnixStream::connect(&socket_path));
        assert_eq!(Some(&*socket_path),
                   stream.peer_addr().unwrap().as_pathname());
        or_panic!(stream.write_all(msg1));
        let mut buf = vec![];
        or_panic!(stream.read_to_end(&mut buf));
        assert_eq!(&msg2[..], &buf[..]);
        drop(stream);

        thread.join().unwrap();
    }

    #[test]
    fn vectored() {
        let (mut s1, mut s2) = or_panic!(UnixStream::pair());

        let len = or_panic!(s1.write_vectored(
            &[IoSlice::new(b"hello"), IoSlice::new(b" "), IoSlice::new(b"world!")],
        ));
        assert_eq!(len, 12);

        let mut buf1 = [0; 6];
        let mut buf2 = [0; 7];
        let len = or_panic!(s2.read_vectored(
            &mut [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2)],
        ));
        assert_eq!(len, 12);
        assert_eq!(&buf1, b"hello ");
        assert_eq!(&buf2, b"world!\0");
    }

    #[test]
    fn pair() {
        let msg1 = b"hello";
        let msg2 = b"world!";

        let (mut s1, mut s2) = or_panic!(UnixStream::pair());
        let thread = thread::spawn(move || {
            // s1 must be moved in or the test will hang!
            let mut buf = [0; 5];
            or_panic!(s1.read(&mut buf));
            assert_eq!(&msg1[..], &buf[..]);
            or_panic!(s1.write_all(msg2));
        });

        or_panic!(s2.write_all(msg1));
        let mut buf = vec![];
        or_panic!(s2.read_to_end(&mut buf));
        assert_eq!(&msg2[..], &buf[..]);
        drop(s2);

        thread.join().unwrap();
    }

    #[test]
    fn try_clone() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");
        let msg1 = b"hello";
        let msg2 = b"world";

        let listener = or_panic!(UnixListener::bind(&socket_path));
        let thread = thread::spawn(move || {
            let mut stream = or_panic!(listener.accept()).0;
            or_panic!(stream.write_all(msg1));
            or_panic!(stream.write_all(msg2));
        });

        let mut stream = or_panic!(UnixStream::connect(&socket_path));
        let mut stream2 = or_panic!(stream.try_clone());

        let mut buf = [0; 5];
        or_panic!(stream.read(&mut buf));
        assert_eq!(&msg1[..], &buf[..]);
        or_panic!(stream2.read(&mut buf));
        assert_eq!(&msg2[..], &buf[..]);

        thread.join().unwrap();
    }

    #[test]
    fn iter() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");

        let listener = or_panic!(UnixListener::bind(&socket_path));
        let thread = thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let mut stream = or_panic!(stream);
                let mut buf = [0];
                or_panic!(stream.read(&mut buf));
            }
        });

        for _ in 0..2 {
            let mut stream = or_panic!(UnixStream::connect(&socket_path));
            or_panic!(stream.write_all(&[0]));
        }

        thread.join().unwrap();
    }

    #[test]
    fn long_path() {
        let dir = tmpdir();
        let socket_path = dir.path()
                             .join("asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfa\
                                    sasdfasdfasdasdfasdfasdfadfasdfasdfasdfasdfasdf");
        match UnixStream::connect(&socket_path) {
            Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
            Err(e) => panic!("unexpected error {}", e),
            Ok(_) => panic!("unexpected success"),
        }

        match UnixListener::bind(&socket_path) {
            Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
            Err(e) => panic!("unexpected error {}", e),
            Ok(_) => panic!("unexpected success"),
        }

        match UnixDatagram::bind(&socket_path) {
            Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
            Err(e) => panic!("unexpected error {}", e),
            Ok(_) => panic!("unexpected success"),
        }
    }

    #[test]
    fn timeouts() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");

        let _listener = or_panic!(UnixListener::bind(&socket_path));

        let stream = or_panic!(UnixStream::connect(&socket_path));
        let dur = Duration::new(15410, 0);

        assert_eq!(None, or_panic!(stream.read_timeout()));

        or_panic!(stream.set_read_timeout(Some(dur)));
        assert_eq!(Some(dur), or_panic!(stream.read_timeout()));

        assert_eq!(None, or_panic!(stream.write_timeout()));

        or_panic!(stream.set_write_timeout(Some(dur)));
        assert_eq!(Some(dur), or_panic!(stream.write_timeout()));

        or_panic!(stream.set_read_timeout(None));
        assert_eq!(None, or_panic!(stream.read_timeout()));

        or_panic!(stream.set_write_timeout(None));
        assert_eq!(None, or_panic!(stream.write_timeout()));
    }

    #[test]
    fn test_read_timeout() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");

        let _listener = or_panic!(UnixListener::bind(&socket_path));

        let mut stream = or_panic!(UnixStream::connect(&socket_path));
        or_panic!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        let mut buf = [0; 10];
        let kind = stream.read_exact(&mut buf).err().expect("expected error").kind();
        assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut,
                "unexpected_error: {:?}", kind);
    }

    #[test]
    fn test_read_with_timeout() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");

        let listener = or_panic!(UnixListener::bind(&socket_path));

        let mut stream = or_panic!(UnixStream::connect(&socket_path));
        or_panic!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        let mut other_end = or_panic!(listener.accept()).0;
        or_panic!(other_end.write_all(b"hello world"));

        let mut buf = [0; 11];
        or_panic!(stream.read(&mut buf));
        assert_eq!(b"hello world", &buf[..]);

        let kind = stream.read_exact(&mut buf).err().expect("expected error").kind();
        assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut,
                "unexpected_error: {:?}", kind);
    }

    // Ensure the `set_read_timeout` and `set_write_timeout` calls return errors
    // when passed zero Durations
    #[test]
    fn test_unix_stream_timeout_zero_duration() {
        let dir = tmpdir();
        let socket_path = dir.path().join("sock");

        let listener = or_panic!(UnixListener::bind(&socket_path));
        let stream = or_panic!(UnixStream::connect(&socket_path));

        let result = stream.set_write_timeout(Some(Duration::new(0, 0)));
        let err = result.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);

        let result = stream.set_read_timeout(Some(Duration::new(0, 0)));
        let err = result.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);

        drop(listener);
    }

    #[test]
    fn test_unix_datagram() {
        let dir = tmpdir();
        let path1 = dir.path().join("sock1");
        let path2 = dir.path().join("sock2");

        let sock1 = or_panic!(UnixDatagram::bind(&path1));
        let sock2 = or_panic!(UnixDatagram::bind(&path2));

        let msg = b"hello world";
        or_panic!(sock1.send_to(msg, &path2));
        let mut buf = [0; 11];
        or_panic!(sock2.recv_from(&mut buf));
        assert_eq!(msg, &buf[..]);
    }

    #[test]
    fn test_unnamed_unix_datagram() {
        let dir = tmpdir();
        let path1 = dir.path().join("sock1");

        let sock1 = or_panic!(UnixDatagram::bind(&path1));
        let sock2 = or_panic!(UnixDatagram::unbound());

        let msg = b"hello world";
        or_panic!(sock2.send_to(msg, &path1));
        let mut buf = [0; 11];
        let (usize, addr) = or_panic!(sock1.recv_from(&mut buf));
        assert_eq!(usize, 11);
        assert!(addr.is_unnamed());
        assert_eq!(msg, &buf[..]);
    }

    #[test]
    fn test_connect_unix_datagram() {
        let dir = tmpdir();
        let path1 = dir.path().join("sock1");
        let path2 = dir.path().join("sock2");

        let bsock1 = or_panic!(UnixDatagram::bind(&path1));
        let bsock2 = or_panic!(UnixDatagram::bind(&path2));
        let sock = or_panic!(UnixDatagram::unbound());
        or_panic!(sock.connect(&path1));

        // Check send()
        let msg = b"hello there";
        or_panic!(sock.send(msg));
        let mut buf = [0; 11];
        let (usize, addr) = or_panic!(bsock1.recv_from(&mut buf));
        assert_eq!(usize, 11);
        assert!(addr.is_unnamed());
        assert_eq!(msg, &buf[..]);

        // Changing default socket works too
        or_panic!(sock.connect(&path2));
        or_panic!(sock.send(msg));
        or_panic!(bsock2.recv_from(&mut buf));
    }

    #[test]
    fn test_unix_datagram_recv() {
        let dir = tmpdir();
        let path1 = dir.path().join("sock1");

        let sock1 = or_panic!(UnixDatagram::bind(&path1));
        let sock2 = or_panic!(UnixDatagram::unbound());
        or_panic!(sock2.connect(&path1));

        let msg = b"hello world";
        or_panic!(sock2.send(msg));
        let mut buf = [0; 11];
        let size = or_panic!(sock1.recv(&mut buf));
        assert_eq!(size, 11);
        assert_eq!(msg, &buf[..]);
    }

    #[test]
    fn datagram_pair() {
        let msg1 = b"hello";
        let msg2 = b"world!";

        let (s1, s2) = or_panic!(UnixDatagram::pair());
        let thread = thread::spawn(move || {
            // s1 must be moved in or the test will hang!
            let mut buf = [0; 5];
            or_panic!(s1.recv(&mut buf));
            assert_eq!(&msg1[..], &buf[..]);
            or_panic!(s1.send(msg2));
        });

        or_panic!(s2.send(msg1));
        let mut buf = [0; 6];
        or_panic!(s2.recv(&mut buf));
        assert_eq!(&msg2[..], &buf[..]);
        drop(s2);

        thread.join().unwrap();
    }

    // Ensure the `set_read_timeout` and `set_write_timeout` calls return errors
    // when passed zero Durations
    #[test]
    fn test_unix_datagram_timeout_zero_duration() {
        let dir = tmpdir();
        let path = dir.path().join("sock");

        let datagram = or_panic!(UnixDatagram::bind(&path));

        let result = datagram.set_write_timeout(Some(Duration::new(0, 0)));
        let err = result.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);

        let result = datagram.set_read_timeout(Some(Duration::new(0, 0)));
        let err = result.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn abstract_namespace_not_allowed() {
        assert!(UnixStream::connect("\0asdf").is_err());
    }
}
