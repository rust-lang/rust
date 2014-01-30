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

use prelude::*;

use c_str::ToCStr;
use rt::rtio::{IoFactory, LocalIo, RtioUnixListener};
use rt::rtio::{RtioUnixAcceptor, RtioPipe};
use io::pipe::PipeStream;
use io::{io_error, Listener, Acceptor, Reader, Writer};

/// A stream which communicates over a named pipe.
pub struct UnixStream {
    priv obj: PipeStream,
}

impl UnixStream {
    fn new(obj: ~RtioPipe) -> UnixStream {
        UnixStream { obj: PipeStream::new(obj) }
    }

    /// Connect to a pipe named by `path`. This will attempt to open a
    /// connection to the underlying socket.
    ///
    /// The returned stream will be closed when the object falls out of scope.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if the connection
    /// could not be made.
    ///
    /// # Example
    ///
    ///     use std::io::net::unix::UnixStream;
    ///
    ///     let server = Path("path/to/my/socket");
    ///     let mut stream = UnixStream::connect(&server);
    ///     stream.write([1, 2, 3]);
    ///
    pub fn connect<P: ToCStr>(path: &P) -> Option<UnixStream> {
        LocalIo::maybe_raise(|io| {
            io.unix_connect(&path.to_c_str()).map(UnixStream::new)
        })
    }
}

impl Reader for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.obj.read(buf) }
}

impl Writer for UnixStream {
    fn write(&mut self, buf: &[u8]) { self.obj.write(buf) }
}

pub struct UnixListener {
    priv obj: ~RtioUnixListener,
}

impl UnixListener {

    /// Creates a new listener, ready to receive incoming connections on the
    /// specified socket. The server will be named by `path`.
    ///
    /// This listener will be closed when it falls out of scope.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if the specified
    /// path could not be bound.
    ///
    /// # Example
    ///
    ///     use std::io::net::unix::UnixListener;
    ///
    ///     let server = Path("path/to/my/socket");
    ///     let mut stream = UnixListener::bind(&server);
    ///     for client in stream.incoming() {
    ///         let mut client = client;
    ///         client.write([1, 2, 3, 4]);
    ///     }
    ///
    pub fn bind<P: ToCStr>(path: &P) -> Option<UnixListener> {
        LocalIo::maybe_raise(|io| {
            io.unix_bind(&path.to_c_str()).map(|s| UnixListener { obj: s })
        })
    }
}

impl Listener<UnixStream, UnixAcceptor> for UnixListener {
    fn listen(self) -> Option<UnixAcceptor> {
        match self.obj.listen() {
            Ok(acceptor) => Some(UnixAcceptor { obj: acceptor }),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

pub struct UnixAcceptor {
    priv obj: ~RtioUnixAcceptor,
}

impl Acceptor<UnixStream> for UnixAcceptor {
    fn accept(&mut self) -> Option<UnixStream> {
        match self.obj.accept() {
            Ok(s) => Some(UnixStream::new(s)),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use io::*;
    use io::test::*;

    fn smalltest(server: proc(UnixStream), client: proc(UnixStream)) {
        let path1 = next_test_unix();
        let path2 = path1.clone();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            client(UnixStream::connect(&path2).unwrap());
        });

        let mut acceptor = UnixListener::bind(&path1).listen();
        chan.send(());
        server(acceptor.accept().unwrap());
    }

    #[test]
    fn bind_error() {
        let mut called = false;
        io_error::cond.trap(|e| {
            assert!(e.kind == PermissionDenied);
            called = true;
        }).inside(|| {
            let listener = UnixListener::bind(&("path/to/nowhere"));
            assert!(listener.is_none());
        });
        assert!(called);
    }

    #[test]
    fn connect_error() {
        let mut called = false;
        io_error::cond.trap(|e| {
            assert_eq!(e.kind,
                       if cfg!(windows) {OtherIoError} else {FileNotFound});
            called = true;
        }).inside(|| {
            let stream = UnixStream::connect(&("path/to/nowhere"));
            assert!(stream.is_none());
        });
        assert!(called);
    }

    #[test]
    fn smoke() {
        smalltest(proc(mut server) {
            let mut buf = [0];
            server.read(buf);
            assert!(buf[0] == 99);
        }, proc(mut client) {
            client.write([99]);
        })
    }

    #[test]
    fn read_eof() {
        smalltest(proc(mut server) {
            let mut buf = [0];
            assert!(server.read(buf).is_none());
            assert!(server.read(buf).is_none());
        }, proc(_client) {
            // drop the client
        })
    }

    #[test]
    fn write_begone() {
        smalltest(proc(mut server) {
            let buf = [0];
            let mut stop = false;
            while !stop{
                io_error::cond.trap(|e| {
                    assert!(e.kind == BrokenPipe || e.kind == NotConnected,
                            "unknown error {:?}", e);
                    stop = true;
                }).inside(|| {
                    server.write(buf);
                })
            }
        }, proc(_client) {
            // drop the client
        })
    }

    #[test]
    fn accept_lots() {
        let times = 10;
        let path1 = next_test_unix();
        let path2 = path1.clone();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            for _ in range(0, times) {
                let mut stream = UnixStream::connect(&path2);
                stream.write([100]);
            }
        });

        let mut acceptor = UnixListener::bind(&path1).listen();
        chan.send(());
        for _ in range(0, times) {
            let mut client = acceptor.accept();
            let mut buf = [0];
            client.read(buf);
            assert_eq!(buf[0], 100);
        }
    }

    #[test]
    fn path_exists() {
        let path = next_test_unix();
        let _acceptor = UnixListener::bind(&path).listen();
        assert!(path.exists());
    }
}
