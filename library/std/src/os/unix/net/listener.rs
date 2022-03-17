use super::{sockaddr_un, SocketAddr, UnixStream};
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::path::Path;
use crate::sys::cvt;
use crate::sys::net::Socket;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, io, mem};

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
/// fn main() -> std::io::Result<()> {
///     let listener = UnixListener::bind("/path/to/the/socket")?;
///
///     // accept connections and process them, spawning a new thread for each one
///     for stream in listener.incoming() {
///         match stream {
///             Ok(stream) => {
///                 /* connection succeeded */
///                 thread::spawn(|| handle_client(stream));
///             }
///             Err(err) => {
///                 /* connection failed */
///                 break;
///             }
///         }
///     }
///     Ok(())
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
        unsafe {
            let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
            let (addr, len) = sockaddr_un(path.as_ref())?;

            cvt(libc::bind(inner.as_inner().as_raw_fd(), &addr as *const _ as *const _, len as _))?;
            cvt(libc::listen(inner.as_inner().as_raw_fd(), 128))?;

            Ok(UnixListener(inner))
        }
    }

    /// Creates a new `UnixListener` bound to the specified [`socket address`].
    ///
    /// [`socket address`]: crate::os::unix::net::SocketAddr
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_abstract)]
    /// use std::os::unix::net::{UnixListener};
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener1 = UnixListener::bind("path/to/socket")?;
    ///     let addr = listener1.local_addr()?;
    ///
    ///     let listener2 = match UnixListener::bind_addr(&addr) {
    ///         Ok(sock) => sock,
    ///         Err(err) => {
    ///             println!("Couldn't bind: {:?}", err);
    ///             return Err(err);
    ///         }
    ///     };
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_abstract", issue = "85410")]
    pub fn bind_addr(socket_addr: &SocketAddr) -> io::Result<UnixListener> {
        unsafe {
            let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
            cvt(libc::bind(
                inner.as_raw_fd(),
                &socket_addr.addr as *const _ as *const _,
                socket_addr.len as _,
            ))?;
            cvt(libc::listen(inner.as_raw_fd(), 128))?;
            Ok(UnixListener(inner))
        }
    }

    /// Accepts a new incoming connection to this listener.
    ///
    /// This function will block the calling thread until a new Unix connection
    /// is established. When established, the corresponding [`UnixStream`] and
    /// the remote peer's address will be returned.
    ///
    /// [`UnixStream`]: crate::os::unix::net::UnixStream
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///
    ///     match listener.accept() {
    ///         Ok((socket, addr)) => println!("Got a client: {:?}", addr),
    ///         Err(e) => println!("accept function failed: {:?}", e),
    ///     }
    ///     Ok(())
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
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     let listener_copy = listener.try_clone().expect("try_clone failed");
    ///     Ok(())
    /// }
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
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     let addr = listener.local_addr().expect("Couldn't get local address");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(self.as_raw_fd(), addr, len) })
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// This will result in the `accept` operation becoming nonblocking,
    /// i.e., immediately returning from their calls. If the IO operation is
    /// successful, `Ok` is returned and no further action is required. If the
    /// IO operation could not be completed and needs to be retried, an error
    /// with kind [`io::ErrorKind::WouldBlock`] is returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     listener.set_nonblocking(true).expect("Couldn't set non blocking");
    ///     Ok(())
    /// }
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
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/tmp/sock")?;
    ///
    ///     if let Ok(Some(err)) = listener.take_error() {
    ///         println!("Got error: {:?}", err);
    ///     }
    ///     Ok(())
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
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///
    ///     for stream in listener.incoming() {
    ///         match stream {
    ///             Ok(stream) => {
    ///                 thread::spawn(|| handle_client(stream));
    ///             }
    ///             Err(err) => {
    ///                 break;
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixListener {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_inner().as_raw_fd()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixListener {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> UnixListener {
        UnixListener(Socket::from_inner(FromInner::from_inner(OwnedFd::from_raw_fd(fd))))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixListener {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner().into_inner().into_raw_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for UnixListener {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_inner().as_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<OwnedFd> for UnixListener {
    #[inline]
    fn from(fd: OwnedFd) -> UnixListener {
        UnixListener(Socket::from_inner(FromInner::from_inner(fd)))
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<UnixListener> for OwnedFd {
    #[inline]
    fn from(listener: UnixListener) -> OwnedFd {
        listener.0.into_inner().into_inner()
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
/// fn main() -> std::io::Result<()> {
///     let listener = UnixListener::bind("/path/to/the/socket")?;
///
///     for stream in listener.incoming() {
///         match stream {
///             Ok(stream) => {
///                 thread::spawn(|| handle_client(stream));
///             }
///             Err(err) => {
///                 break;
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
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
        (usize::MAX, None)
    }
}
