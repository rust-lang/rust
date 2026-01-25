use crate::os::windows::io::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::os::windows::net::{SocketAddr, UnixStream};
use crate::path::Path;
#[cfg(not(doc))]
use crate::sys::c::{AF_UNIX, SOCK_STREAM, SOCKADDR_UN, bind, getsockname, listen};
use crate::sys::net::Socket;
#[cfg(not(doc))]
use crate::sys::winsock::startup;
use crate::sys::{AsInner, cvt_nz};
use crate::{fmt, io};

/// A structure representing a Unix domain socket server.
///
/// # Examples
///
/// ```no_run
/// use std::thread;
/// use std::os::windows::net::{UnixStream, UnixListener};
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
pub struct UnixListener(Socket);

impl fmt::Debug for UnixListener {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixListener");
        builder.field("sock", self.0.as_inner());
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
    /// use std::os::windows::net::UnixListener;
    ///
    /// let listener = match UnixListener::bind("/path/to/the/socket") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't connect: {e:?}");
    ///         return
    ///     }
    /// };
    /// ```
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
        let socket_addr = SocketAddr::from_pathname(path)?;
        Self::bind_addr(&socket_addr)
    }

    /// Creates a new `UnixListener` bound to the specified [`socket address`].
    ///
    /// [`socket address`]: crate::os::windows::net::SocketAddr
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::windows::net::{UnixListener};
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener1 = UnixListener::bind("path/to/socket")?;
    ///     let addr = listener1.local_addr()?;
    ///
    ///     let listener2 = match UnixListener::bind_addr(&addr) {
    ///         Ok(sock) => sock,
    ///         Err(err) => {
    ///             println!("Couldn't bind: {err:?}");
    ///             return Err(err);
    ///         }
    ///     };
    ///     Ok(())
    /// }
    /// ```
    pub fn bind_addr(socket_addr: &SocketAddr) -> io::Result<UnixListener> {
        startup();
        let inner = Socket::new(AF_UNIX as _, SOCK_STREAM)?;
        unsafe {
            cvt_nz(bind(inner.as_raw(), &raw const socket_addr.addr as _, socket_addr.len as _))?;
            cvt_nz(listen(inner.as_raw(), 128))?;
        }
        Ok(UnixListener(inner))
    }

    /// Accepts a new incoming connection to this listener.
    ///
    /// This function will block the calling thread until a new Unix connection
    /// is established. When established, the corresponding [`UnixStream`] and
    /// the remote peer's address will be returned.
    ///
    /// [`UnixStream`]: crate::os::windows::net::UnixStream
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::windows::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///
    ///     match listener.accept() {
    ///         Ok((socket, addr)) => println!("Got a client: {addr:?}"),
    ///         Err(e) => println!("accept function failed: {e:?}"),
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        let mut storage = SOCKADDR_UN::default();
        let mut len = size_of::<SOCKADDR_UN>() as _;
        let inner = self.0.accept(&raw mut storage as *mut _, &raw mut len)?;
        let addr = SocketAddr::from_parts(storage, len)?;
        Ok((UnixStream(inner), addr))
    }

    /// Returns the local socket address of this listener.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::windows::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     let addr = listener.local_addr().expect("Couldn't get local address");
    ///     Ok(())
    /// }
    /// ```
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { getsockname(self.0.as_raw(), addr, len) })
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
    /// use std::os::windows::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     let listener_copy = listener.try_clone().expect("try_clone failed");
    ///     Ok(())
    /// }
    /// ```
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
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
    /// use std::os::windows::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     listener.set_nonblocking(true).expect("Couldn't set non blocking");
    ///     Ok(())
    /// }
    /// ```
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::windows::net::UnixListener;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/tmp/sock")?;
    ///
    ///     if let Ok(Some(err)) = listener.take_error() {
    ///         println!("Got error: {err:?}");
    ///     }
    ///     Ok(())
    /// }
    /// ```
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
    /// use std::os::windows::net::{UnixStream, UnixListener};
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
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
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
/// use std::os::windows::net::{UnixStream, UnixListener};
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
pub struct Incoming<'a> {
    listener: &'a UnixListener,
}

impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<UnixStream>;

    fn next(&mut self) -> Option<io::Result<UnixStream>> {
        Some(self.listener.accept().map(|s| s.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

impl AsRawSocket for UnixListener {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.0.as_raw_socket()
    }
}

impl FromRawSocket for UnixListener {
    #[inline]
    unsafe fn from_raw_socket(sock: RawSocket) -> Self {
        UnixListener(unsafe { Socket::from_raw_socket(sock) })
    }
}

impl IntoRawSocket for UnixListener {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        self.0.into_raw_socket()
    }
}

impl<'a> IntoIterator for &'a UnixListener {
    type Item = io::Result<UnixStream>;
    type IntoIter = Incoming<'a>;

    fn into_iter(self) -> Incoming<'a> {
        self.incoming()
    }
}
