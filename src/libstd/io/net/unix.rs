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
use clone::Clone;
use rt::rtio::{IoFactory, LocalIo, RtioUnixListener};
use rt::rtio::{RtioUnixAcceptor, RtioPipe};
use io::pipe::PipeStream;
use io::{Listener, Acceptor, Reader, Writer, IoResult};

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
    /// # Example
    ///
    /// ```rust
    /// # #[allow(unused_must_use)];
    /// use std::io::net::unix::UnixStream;
    ///
    /// let server = Path::new("path/to/my/socket");
    /// let mut stream = UnixStream::connect(&server);
    /// stream.write([1, 2, 3]);
    /// ```
    pub fn connect<P: ToCStr>(path: &P) -> IoResult<UnixStream> {
        LocalIo::maybe_raise(|io| {
            io.unix_connect(&path.to_c_str()).map(UnixStream::new)
        })
    }
}

impl Clone for UnixStream {
    fn clone(&self) -> UnixStream {
        UnixStream { obj: self.obj.clone() }
    }
}

impl Reader for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.obj.read(buf) }
}

impl Writer for UnixStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.obj.write(buf) }
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
    /// # Example
    ///
    /// ```
    /// use std::io::net::unix::UnixListener;
    /// use std::io::Listener;
    ///
    /// let server = Path::new("path/to/my/socket");
    /// let mut stream = UnixListener::bind(&server);
    /// for client in stream.incoming() {
    ///     let mut client = client;
    ///     client.write([1, 2, 3, 4]);
    /// }
    /// ```
    pub fn bind<P: ToCStr>(path: &P) -> IoResult<UnixListener> {
        LocalIo::maybe_raise(|io| {
            io.unix_bind(&path.to_c_str()).map(|s| UnixListener { obj: s })
        })
    }
}

impl Listener<UnixStream, UnixAcceptor> for UnixListener {
    fn listen(self) -> IoResult<UnixAcceptor> {
        self.obj.listen().map(|obj| UnixAcceptor { obj: obj })
    }
}

pub struct UnixAcceptor {
    priv obj: ~RtioUnixAcceptor,
}

impl Acceptor<UnixStream> for UnixAcceptor {
    fn accept(&mut self) -> IoResult<UnixStream> {
        self.obj.accept().map(UnixStream::new)
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
        match UnixListener::bind(&("path/to/nowhere")) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, PermissionDenied),
        }
    }

    #[test]
    fn connect_error() {
        match UnixStream::connect(&("path/to/nowhere")) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind,
                        if cfg!(windows) {OtherIoError} else {FileNotFound})
        }
    }

    #[test]
    fn smoke() {
        smalltest(proc(mut server) {
            let mut buf = [0];
            server.read(buf).unwrap();
            assert!(buf[0] == 99);
        }, proc(mut client) {
            client.write([99]).unwrap();
        })
    }

    #[test]
    fn read_eof() {
        smalltest(proc(mut server) {
            let mut buf = [0];
            assert!(server.read(buf).is_err());
            assert!(server.read(buf).is_err());
        }, proc(_client) {
            // drop the client
        })
    }

    #[test]
    fn write_begone() {
        smalltest(proc(mut server) {
            let buf = [0];
            loop {
                match server.write(buf) {
                    Ok(..) => {}
                    Err(e) => {
                        assert!(e.kind == BrokenPipe || e.kind == NotConnected,
                                "unknown error {:?}", e);
                        break;
                    }
                }
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
                stream.write([100]).unwrap();
            }
        });

        let mut acceptor = UnixListener::bind(&path1).listen();
        chan.send(());
        for _ in range(0, times) {
            let mut client = acceptor.accept();
            let mut buf = [0];
            client.read(buf).unwrap();
            assert_eq!(buf[0], 100);
        }
    }

    #[test]
    fn path_exists() {
        let path = next_test_unix();
        let _acceptor = UnixListener::bind(&path).listen();
        assert!(path.exists());
    }

    #[test]
    fn unix_clone_smoke() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();

        spawn(proc() {
            let mut s = UnixStream::connect(&addr);
            let mut buf = [0, 0];
            assert_eq!(s.read(buf), Ok(1));
            assert_eq!(buf[0], 1);
            s.write([2]).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (p1, c1) = Chan::new();
        let (p2, c2) = Chan::new();
        spawn(proc() {
            let mut s2 = s2;
            p1.recv();
            s2.write([1]).unwrap();
            c2.send(());
        });
        c1.send(());
        let mut buf = [0, 0];
        assert_eq!(s1.read(buf), Ok(1));
        p2.recv();
    }

    #[test]
    fn unix_clone_two_read() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();
        let (p, c) = Chan::new();
        let c2 = c.clone();

        spawn(proc() {
            let mut s = UnixStream::connect(&addr);
            s.write([1]).unwrap();
            p.recv();
            s.write([2]).unwrap();
            p.recv();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (p, done) = Chan::new();
        spawn(proc() {
            let mut s2 = s2;
            let mut buf = [0, 0];
            s2.read(buf).unwrap();
            c2.send(());
            done.send(());
        });
        let mut buf = [0, 0];
        s1.read(buf).unwrap();
        c.send(());

        p.recv();
    }

    #[test]
    fn unix_clone_two_write() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();

        spawn(proc() {
            let mut s = UnixStream::connect(&addr);
            let mut buf = [0, 1];
            s.read(buf).unwrap();
            s.read(buf).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (p, done) = Chan::new();
        spawn(proc() {
            let mut s2 = s2;
            s2.write([1]).unwrap();
            done.send(());
        });
        s1.write([2]).unwrap();

        p.recv();
    }
}
