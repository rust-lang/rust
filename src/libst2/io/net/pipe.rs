// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Named pipes

This module contains the ability to communicate over named pipes with
synchronous I/O. On windows, this corresponds to talking over a Named Pipe,
while on Unix it corresponds to UNIX domain sockets.

These pipes are similar to TCP in the sense that you can have both a stream to a
server and a server itself. The server provided accepts other `UnixStream`
instances as clients.

*/

#![allow(missing_docs)]

use prelude::*;

use io::{Listener, Acceptor, IoResult, TimedOut, standard_error};
use time::Duration;

use sys::pipe::UnixStream as UnixStreamImp;
use sys::pipe::UnixListener as UnixListenerImp;
use sys::pipe::UnixAcceptor as UnixAcceptorImp;

/// A stream which communicates over a named pipe.
pub struct UnixStream {
    inner: UnixStreamImp,
}

impl UnixStream {

    /// Connect to a pipe named by `path`. This will attempt to open a
    /// connection to the underlying socket.
    ///
    /// The returned stream will be closed when the object falls out of scope.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #![allow(unused_must_use)]
    /// use std::io::net::pipe::UnixStream;
    ///
    /// let server = Path::new("path/to/my/socket");
    /// let mut stream = UnixStream::connect(&server);
    /// stream.write(&[1, 2, 3]);
    /// ```
    pub fn connect<P: ToCStr>(path: &P) -> IoResult<UnixStream> { unimplemented!() }

    /// Connect to a pipe named by `path`, timing out if the specified number of
    /// milliseconds.
    ///
    /// This function is similar to `connect`, except that if `timeout`
    /// elapses the function will return an error of kind `TimedOut`.
    ///
    /// If a `timeout` with zero or negative duration is specified then
    /// the function returns `Err`, with the error kind set to `TimedOut`.
    #[experimental = "the timeout argument is likely to change types"]
    pub fn connect_timeout<P: ToCStr>(path: &P,
                                      timeout: Duration) -> IoResult<UnixStream> { unimplemented!() }


    /// Closes the reading half of this connection.
    ///
    /// This method will close the reading portion of this connection, causing
    /// all pending and future reads to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_read(&mut self) -> IoResult<()> { unimplemented!() }

    /// Closes the writing half of this connection.
    ///
    /// This method will close the writing portion of this connection, causing
    /// all pending and future writes to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_write(&mut self) -> IoResult<()> { unimplemented!() }

    /// Sets the read/write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Sets the read timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_read_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Sets the write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_write_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }
}

impl Clone for UnixStream {
    fn clone(&self) -> UnixStream { unimplemented!() }
}

impl Reader for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl Writer for UnixStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
}

/// A value that can listen for incoming named pipe connection requests.
pub struct UnixListener {
    /// The internal, opaque runtime Unix listener.
    inner: UnixListenerImp,
}

impl UnixListener {
    /// Creates a new listener, ready to receive incoming connections on the
    /// specified socket. The server will be named by `path`.
    ///
    /// This listener will be closed when it falls out of scope.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() {}
    /// # fn foo() {
    /// # #![allow(unused_must_use)]
    /// use std::io::net::pipe::UnixListener;
    /// use std::io::{Listener, Acceptor};
    ///
    /// let server = Path::new("/path/to/my/socket");
    /// let stream = UnixListener::bind(&server);
    /// for mut client in stream.listen().incoming() {
    ///     client.write(&[1, 2, 3, 4]);
    /// }
    /// # }
    /// ```
    pub fn bind<P: ToCStr>(path: &P) -> IoResult<UnixListener> { unimplemented!() }
}

impl Listener<UnixStream, UnixAcceptor> for UnixListener {
    fn listen(self) -> IoResult<UnixAcceptor> { unimplemented!() }
}

/// A value that can accept named pipe connections, returned from `listen()`.
pub struct UnixAcceptor {
    /// The internal, opaque runtime Unix acceptor.
    inner: UnixAcceptorImp
}

impl UnixAcceptor {
    /// Sets a timeout for this acceptor, after which accept() will no longer
    /// block indefinitely.
    ///
    /// The argument specified is the amount of time, in milliseconds, into the
    /// future after which all invocations of accept() will not block (and any
    /// pending invocation will return). A value of `None` will clear any
    /// existing timeout.
    ///
    /// When using this method, it is likely necessary to reset the timeout as
    /// appropriate, the timeout specified is specific to this object, not
    /// specific to the next request.
    #[experimental = "the name and arguments to this function are likely \
                      to change"]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Closes the accepting capabilities of this acceptor.
    ///
    /// This function has the same semantics as `TcpAcceptor::close_accept`, and
    /// more information can be found in that documentation.
    #[experimental]
    pub fn close_accept(&mut self) -> IoResult<()> { unimplemented!() }
}

impl Acceptor<UnixStream> for UnixAcceptor {
    fn accept(&mut self) -> IoResult<UnixStream> { unimplemented!() }
}

impl Clone for UnixAcceptor {
    /// Creates a new handle to this unix acceptor, allowing for simultaneous
    /// accepts.
    ///
    /// The underlying unix acceptor will not be closed until all handles to the
    /// acceptor have been deallocated. Incoming connections will be received on
    /// at most once acceptor, the same connection will not be accepted twice.
    ///
    /// The `close_accept` method will shut down *all* acceptors cloned from the
    /// same original acceptor, whereas the `set_timeout` method only affects
    /// the selector that it is called on.
    ///
    /// This function is useful for creating a handle to invoke `close_accept`
    /// on to wake up any other task blocked in `accept`.
    fn clone(&self) -> UnixAcceptor { unimplemented!() }
}
