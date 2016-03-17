- Feature Name: `unix_socket`
- Start Date: 2016-01-25
- RFC PR: [rust-lang/rfcs#1479](https://github.com/rust-lang/rfcs/pull/1479)
- Rust Issue: [rust-lang/rust#32312](https://github.com/rust-lang/rust/issues/32312)

# Summary
[summary]: #summary

[Unix domain sockets](https://en.wikipedia.org/wiki/Unix_domain_socket) provide
a commonly used form of IPC on Unix-derived systems. This RFC proposes move the
[unix_socket](https://crates.io/crates/unix_socket/) nursery crate into the
`std::os::unix` module.

# Motivation
[motivation]: #motivation

Unix sockets are a common form of IPC on unixy systems. Databases like
PostgreSQL and Redis allow connections via Unix sockets, and Servo uses them to
communicate with subprocesses. Even though Unix sockets are not present on
Windows, their use is sufficiently widespread to warrant inclusion in the
platform-specific sections of the standard library.

# Detailed design
[design]: #detailed-design

Unix sockets can be configured with the `SOCK_STREAM`, `SOCK_DGRAM`, and
`SOCK_SEQPACKET` types. `SOCK_STREAM` creates a connection-oriented socket that
behaves like a TCP socket, `SOCK_DGRAM` creates a packet-oriented socket that
behaves like a UDP socket, and `SOCK_SEQPACKET` provides something of a hybrid
between the other two - a connection-oriented, reliable, ordered stream of
delimited packets. `SOCK_SEQPACKET` support has not yet been implemented in the
unix_socket crate, so only the first two socket types will initially be
supported in the standard library.

While a TCP or UDP socket would be identified by a IP address and port number,
Unix sockets are typically identified by a filesystem path. For example, a
Postgres server will listen on a Unix socket located at
`/run/postgresql/.s.PGSQL.5432` in some configurations. However, the
`socketpair` function can make a pair of *unnamed* connected Unix sockets not
associated with a filesystem path. In addition, Linux provides a separate
*abstract* namespace not associated with the filesystem, indicated by a leading
null byte in the address. In the initial implementation, the abstract namespace
will not be supported - the various socket constructors will check for and
reject addresses with interior null bytes.

A `std::os::unix::net` module will be created with the following contents:

The `UnixStream` type mirrors `TcpStream`:
```rust
pub struct UnixStream {
    ...
}

impl UnixStream {
    /// Connects to the socket named by `path`.
    ///
    /// `path` may not contain any null bytes.
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
        ...
    }

    /// Creates an unnamed pair of connected sockets.
    ///
    /// Returns two `UnixStream`s which are connected to each other.
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        ...
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixStream` is a reference to the same stream that this
    /// object references. Both handles will read and write the same stream of
    /// data, and options set on one stream will be propogated to the other
    /// stream.
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        ...
    }

    /// Returns the socket address of the local half of this connection.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        ...
    }

    /// Returns the socket address of the remote half of this connection.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        ...
    }

    /// Sets the read timeout for the socket.
    ///
    /// If the provided value is `None`, then `read` calls will block
    /// indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        ...
    }

    /// Sets the write timeout for the socket.
    ///
    /// If the provided value is `None`, then `write` calls will block
    /// indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        ...
    }

    /// Returns the read timeout of this socket.
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        ...
    }

    /// Returns the write timeout of this socket.
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        ...
    }

    /// Moves the socket into or out of nonblocking mode.
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        ...
    }

    /// Returns the value of the `SO_ERROR` option.
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        ...
    }

    /// Shuts down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O calls on the
    /// specified portions to immediately return with an appropriate value
    /// (see the documentation of `Shutdown`).
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        ...
    }
}

impl Read for UnixStream {
    ...
}

impl<'a> Read for &'a UnixStream {
    ...
}

impl Write for UnixStream {
    ...
}

impl<'a> Write for UnixStream {
    ...
}

impl FromRawFd for UnixStream {
    ...
}

impl AsRawFd for UnixStream {
    ...
}

impl IntoRawFd for UnixStream {
    ...
}
```

Differences from `TcpStream`:
* `connect` takes an `AsRef<Path>` rather than a `ToSocketAddrs`.
* The `pair` method creates a pair of connected, unnamed sockets, as this is
  commonly used for IPC.
* The `SocketAddr` returned by the `local_addr` and `peer_addr` methods is
  different.
* The `set_nonblocking` and `take_error` methods are not currently present on
  `TcpStream` but are provided in the `net2` crate and are being proposed for
  addition to the standard library in a separate RFC.

As noted above, a Unix socket can either be unnamed, be associated with a path
on the filesystem, or (on Linux) be associated with an ID in the abstract
namespace. The `SocketAddr` struct is fairly simple:

```rust
pub struct SocketAddr {
    ...
}

impl SocketAddr {
    /// Returns true if the address is unnamed.
    pub fn is_unnamed(&self) -> bool {
        ...
    }

    /// Returns the contents of this address if it corresponds to a filesystem path.
    pub fn as_pathname(&self) -> Option<&Path> {
        ...
    }
}
```

The `UnixListener` type mirrors the `TcpListener` type:
```rust
pub struct UnixListener {
    ...
}

impl UnixListener {
    /// Creates a new `UnixListener` bound to the specified socket.
    ///
    /// `path` may not contain any null bytes.
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
        ...
    }

    /// Accepts a new incoming connection to this listener.
    ///
    /// This function will block the calling thread until a new Unix connection
    /// is established. When established, the corersponding `UnixStream` and
    /// the remote peer's address will be returned.
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        ...
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixListener` is a reference to the same socket that this
    /// object references. Both handles can be used to accept incoming
    /// connections and options set on one listener will affect the other.
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        ...
    }

    /// Returns the local socket address of this listener.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        ...
    }

    /// Moves the socket into or out of nonblocking mode.
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        ...
    }

    /// Returns the value of the `SO_ERROR` option.
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        ...
    }

    /// Returns an iterator over incoming connections.
    ///
    /// The iterator will never return `None` and will also not yield the
    /// peer's `SocketAddr` structure.
    pub fn incoming<'a>(&'a self) -> Incoming<'a> {
        ...
    }
}

impl FromRawFd for UnixListener {
    ...
}

impl AsRawFd for UnixListener {
    ...
}

impl IntoRawFd for UnixListener {
    ...
}
```

Differences from `TcpListener`:
* `bind` takes an `AsRef<Path>` rather than a `ToSocketAddrs`.
* The `SocketAddr` type is different.
* The `set_nonblocking` and `take_error` methods are not currently present on
  `TcpListener` but are provided in the `net2` crate and are being proposed for
  addition to the standard library in a separate RFC.

Finally, the `UnixDatagram` type mirrors the `UpdSocket` type:
```rust
pub struct UnixDatagram {
    ...
}

impl UnixDatagram {
    /// Creates a Unix datagram socket bound to the given path.
    ///
    /// `path` may not contain any null bytes.
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixDatagram> {
        ...
    }

    /// Creates a Unix Datagram socket which is not bound to any address.
    pub fn unbound() -> io::Result<UnixDatagram> {
        ...
    }

    /// Create an unnamed pair of connected sockets.
    ///
    /// Returns two `UnixDatagrams`s which are connected to each other.
    pub fn pair() -> io::Result<(UnixDatagram, UnixDatagram)> {
        ...
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixDatagram` is a reference to the same stream that this
    /// object references. Both handles will read and write the same stream of
    /// data, and options set on one stream will be propogated to the other
    /// stream.
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        ...
    }

    /// Connects the socket to the specified address.
    ///
    /// The `send` method may be used to send data to the specified address.
    /// `recv` and `recv_from` will only receive data from that address.
    ///
    /// `path` may not contain any null bytes.
    pub fn connect<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        ...
    }

    /// Returns the address of this socket.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        ...
    }

    /// Returns the address of this socket's peer.
    ///
    /// The `connect` method will connect the socket to a peer.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        ...
    }

    /// Receives data from the socket.
    ///
    /// On success, returns the number of bytes read and the address from
    /// whence the data came.
    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        ...
    }

    /// Receives data from the socket.
    ///
    /// On success, returns the number of bytes read.
    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        ...
    }

    /// Sends data on the socket to the specified address.
    ///
    /// On success, returns the number of bytes written.
    ///
    /// `path` may not contain any null bytes.
    pub fn send_to<P: AsRef<Path>>(&self, buf: &[u8], path: P) -> io::Result<usize> {
        ...
    }

    /// Sends data on the socket to the socket's peer.
    ///
    /// The peer address may be set by the `connect` method, and this method
    /// will return an error if the socket has not already been connected.
    ///
    /// On success, returns the number of bytes written.
    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        ...
    }

    /// Sets the read timeout for the socket.
    ///
    /// If the provided value is `None`, then `recv` and `recv_from` calls will
    /// block indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        ...
    }

    /// Sets the write timeout for the socket.
    ///
    /// If the provided value is `None`, then `send` and `send_to` calls will
    /// block indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        ...
    }

    /// Returns the read timeout of this socket.
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        ...
    }

    /// Returns the write timeout of this socket.
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        ...
    }

    /// Moves the socket into or out of nonblocking mode.
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        ...
    }

    /// Returns the value of the `SO_ERROR` option.
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        ...
    }

    /// Shut down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O calls on the
    /// specified portions to immediately return with an appropriate value
    /// (see the documentation of `Shutdown`).
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        ...
    }
}

impl FromRawFd for UnixDatagram {
    ...
}

impl AsRawFd for UnixDatagram {
    ...
}

impl IntoRawFd for UnixDatagram {
    ...
}
```

Differences from `UdpSocket`:
* `bind` takes an `AsRef<Path>` rather than a `ToSocketAddrs`.
* The `unbound` method creates an unbound socket, as a Unix socket does not need
  to be bound to send messages.
* The `pair` method creates a pair of connected, unnamed sockets, as this is
  commonly used for IPC.
* The `SocketAddr` returned by the `local_addr` and `peer_addr` methods is
  different.
* The `connect`, `send`, `recv`, `set_nonblocking`, and `take_error` methods are
  not currently present on `UdpSocket` but are provided in the `net2` crate and
  are being proposed for addition to the standard library in a separate RFC.

## Functionality not present

Some functionality is notably absent from this proposal:

* Linux's abstract namespace is not supported. Functionality may be added in
  the future via extension traits in `std::os::linux::net`.
* No support for `SOCK_SEQPACKET` sockets is proposed, as it has not yet been
  implemented. Since it is connection oriented, there will be a socket type
  `UnixSeqPacket` and a listener type `UnixSeqListener`. The naming of the
  listener is a bit unfortunate, but use of `SOCK_SEQPACKET` is rare compared
  to `SOCK_STREAM` so naming priority can go to that version.
* Unix sockets support file descriptor and credential transfer, but these will
  not initially be supported as the `sendmsg`/`recvmsg` interface is complex
  and bindings will need some time to prototype.

These features can bake in the `rust-lang-nursery/unix-socket` as they're
developed.

# Drawbacks
[drawbacks]: #drawbacks

While there is precedent for platform specific components in the standard
library, this will be the by far the largest platform specific addition.

# Alternatives
[alternatives]: #alternatives

Unix socket support could be left out of tree.

The naming convention of `UnixStream` and `UnixDatagram` doesn't perfectly
mirror `TcpStream` and `UdpSocket`, but `UnixStream` and `UnixSocket` seems way
too confusing.

# Unresolved questions
[unresolved]: #unresolved-questions

Is `std::os::unix::net` the right name for this module? It's not strictly
"networking" as all communication is local to one machine. `std::os::unix::unix`
is more accurate but weirdly repetitive and the extension trait module
`std::os::linux::unix` is even weirder. `std::os::unix::socket` is an option,
but seems like too general of a name for specifically `AF_UNIX` sockets as
opposed to *all* sockets.
