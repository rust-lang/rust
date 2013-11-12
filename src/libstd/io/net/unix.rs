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
use rt::rtio::{IoFactory, RtioUnixListener, with_local_io};
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
        do with_local_io |io| {
            match io.unix_connect(&path.to_c_str()) {
                Ok(s) => Some(UnixStream::new(s)),
                Err(ioerr) => {
                    io_error::cond.raise(ioerr);
                    None
                }
            }
        }
    }
}

impl Reader for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.obj.read(buf) }
    fn eof(&mut self) -> bool { self.obj.eof() }
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
        do with_local_io |io| {
            match io.unix_bind(&path.to_c_str()) {
                Ok(s) => Some(UnixListener{ obj: s }),
                Err(ioerr) => {
                    io_error::cond.raise(ioerr);
                    None
                }
            }
        }
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
    use cell::Cell;
    use rt::test::*;
    use io::*;
    use rt::comm::oneshot;

    fn smalltest(server: ~fn(UnixStream), client: ~fn(UnixStream)) {
        let server = Cell::new(server);
        let client = Cell::new(client);
        do run_in_mt_newsched_task {
            let server = Cell::new(server.take());
            let client = Cell::new(client.take());
            let path1 = next_test_unix();
            let path2 = path1.clone();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = UnixListener::bind(&path1).listen();
                chan.take().send(());
                server.take()(acceptor.accept().unwrap());
            }

            do spawntask {
                port.take().recv();
                client.take()(UnixStream::connect(&path2).unwrap());
            }
        }
    }

    #[test]
    fn bind_error() {
        do run_in_mt_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                assert!(e.kind == PermissionDenied);
                called = true;
            }).inside {
                let listener = UnixListener::bind(&("path/to/nowhere"));
                assert!(listener.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn connect_error() {
        do run_in_mt_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                assert_eq!(e.kind, OtherIoError);
                called = true;
            }).inside {
                let stream = UnixStream::connect(&("path/to/nowhere"));
                assert!(stream.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn smoke() {
        smalltest(|mut server| {
            let mut buf = [0];
            server.read(buf);
            assert!(buf[0] == 99);
        }, |mut client| {
            client.write([99]);
        })
    }

    #[test]
    fn read_eof() {
        smalltest(|mut server| {
            let mut buf = [0];
            assert!(server.read(buf).is_none());
            assert!(server.read(buf).is_none());
        }, |_client| {
            // drop the client
        })
    }

    #[test]
    fn write_begone() {
        smalltest(|mut server| {
            let buf = [0];
            let mut stop = false;
            while !stop{
                do io_error::cond.trap(|e| {
                    assert!(e.kind == BrokenPipe || e.kind == NotConnected,
                            "unknown error {:?}", e);
                    stop = true;
                }).inside {
                    server.write(buf);
                }
            }
        }, |_client| {
            // drop the client
        })
    }

    #[test]
    fn accept_lots() {
        do run_in_mt_newsched_task {
            let times = 10;
            let path1 = next_test_unix();
            let path2 = path1.clone();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = UnixListener::bind(&path1).listen();
                chan.take().send(());
                do times.times {
                    let mut client = acceptor.accept();
                    let mut buf = [0];
                    client.read(buf);
                    assert_eq!(buf[0], 100);
                }
            }

            do spawntask {
                port.take().recv();
                do times.times {
                    let mut stream = UnixStream::connect(&path2);
                    stream.write([100]);
                }
            }
        }
    }

    #[test]
    fn path_exists() {
        do run_in_mt_newsched_task {
            let path = next_test_unix();
            let _acceptor = UnixListener::bind(&path).listen();
            assert!(path.exists());
        }
    }
}
