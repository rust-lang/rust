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

#![allow(missing_doc)]

use prelude::*;

use c_str::ToCStr;
use clone::Clone;
use io::pipe::PipeStream;
use io::{Listener, Acceptor, Reader, Writer, IoResult};
use kinds::Send;
use rt::rtio::{IoFactory, LocalIo, RtioUnixListener};
use rt::rtio::{RtioUnixAcceptor, RtioPipe};

/// A stream which communicates over a named pipe.
pub struct UnixStream {
    obj: PipeStream,
}

impl UnixStream {
    fn new(obj: ~RtioPipe:Send) -> UnixStream {
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

/// A value that can listen for incoming named pipe connection requests.
pub struct UnixListener {
    /// The internal, opaque runtime Unix listener.
    obj: ~RtioUnixListener:Send,
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
    /// # #[allow(unused_must_use)];
    /// use std::io::net::unix::UnixListener;
    /// use std::io::{Listener, Acceptor};
    ///
    /// let server = Path::new("/path/to/my/socket");
    /// let stream = UnixListener::bind(&server);
    /// for mut client in stream.listen().incoming() {
    ///     client.write([1, 2, 3, 4]);
    /// }
    /// # }
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

/// A value that can accept named pipe connections, returned from `listen()`.
pub struct UnixAcceptor {
    /// The internal, opaque runtime Unix acceptor.
    obj: ~RtioUnixAcceptor:Send,
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

    pub fn smalltest(server: proc(UnixStream):Send, client: proc(UnixStream):Send) {
        let path1 = next_test_unix();
        let path2 = path1.clone();

        let mut acceptor = UnixListener::bind(&path1).listen();

        spawn(proc() {
            match UnixStream::connect(&path2) {
                Ok(c) => client(c),
                Err(e) => fail!("failed connect: {}", e),
            }
        });

        match acceptor.accept() {
            Ok(c) => server(c),
            Err(e) => fail!("failed accept: {}", e),
        }
    }

    iotest!(fn bind_error() {
        let path = "path/to/nowhere";
        match UnixListener::bind(&path) {
            Ok(..) => fail!(),
            Err(e) => {
                assert!(e.kind == PermissionDenied || e.kind == FileNotFound ||
                        e.kind == InvalidInput);
            }
        }
    })

    iotest!(fn connect_error() {
        let path = if cfg!(windows) {
            r"\\.\pipe\this_should_not_exist_ever"
        } else {
            "path/to/nowhere"
        };
        match UnixStream::connect(&path) {
            Ok(..) => fail!(),
            Err(e) => {
                assert!(e.kind == FileNotFound || e.kind == OtherIoError);
            }
        }
    })

    iotest!(fn smoke() {
        smalltest(proc(mut server) {
            let mut buf = [0];
            server.read(buf).unwrap();
            assert!(buf[0] == 99);
        }, proc(mut client) {
            client.write([99]).unwrap();
        })
    })

    iotest!(fn read_eof() {
        smalltest(proc(mut server) {
            let mut buf = [0];
            assert!(server.read(buf).is_err());
            assert!(server.read(buf).is_err());
        }, proc(_client) {
            // drop the client
        })
    } #[ignore(cfg(windows))]) // FIXME(#12516)

    iotest!(fn write_begone() {
        smalltest(proc(mut server) {
            let buf = [0];
            loop {
                match server.write(buf) {
                    Ok(..) => {}
                    Err(e) => {
                        assert!(e.kind == BrokenPipe ||
                                e.kind == NotConnected ||
                                e.kind == ConnectionReset,
                                "unknown error {:?}", e);
                        break;
                    }
                }
            }
        }, proc(_client) {
            // drop the client
        })
    })

    iotest!(fn accept_lots() {
        let times = 10;
        let path1 = next_test_unix();
        let path2 = path1.clone();

        let mut acceptor = match UnixListener::bind(&path1).listen() {
            Ok(a) => a,
            Err(e) => fail!("failed listen: {}", e),
        };

        spawn(proc() {
            for _ in range(0, times) {
                let mut stream = UnixStream::connect(&path2);
                match stream.write([100]) {
                    Ok(..) => {}
                    Err(e) => fail!("failed write: {}", e)
                }
            }
        });

        for _ in range(0, times) {
            let mut client = acceptor.accept();
            let mut buf = [0];
            match client.read(buf) {
                Ok(..) => {}
                Err(e) => fail!("failed read/accept: {}", e),
            }
            assert_eq!(buf[0], 100);
        }
    })

    #[cfg(unix)]
    iotest!(fn path_exists() {
        let path = next_test_unix();
        let _acceptor = UnixListener::bind(&path).listen();
        assert!(path.exists());
    })

    iotest!(fn unix_clone_smoke() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();

        spawn(proc() {
            let mut s = UnixStream::connect(&addr);
            let mut buf = [0, 0];
            debug!("client reading");
            assert_eq!(s.read(buf), Ok(1));
            assert_eq!(buf[0], 1);
            debug!("client writing");
            s.write([2]).unwrap();
            debug!("client dropping");
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        spawn(proc() {
            let mut s2 = s2;
            rx1.recv();
            debug!("writer writing");
            s2.write([1]).unwrap();
            debug!("writer done");
            tx2.send(());
        });
        tx1.send(());
        let mut buf = [0, 0];
        debug!("reader reading");
        assert_eq!(s1.read(buf), Ok(1));
        debug!("reader done");
        rx2.recv();
    })

    iotest!(fn unix_clone_two_read() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();

        spawn(proc() {
            let mut s = UnixStream::connect(&addr);
            s.write([1]).unwrap();
            rx.recv();
            s.write([2]).unwrap();
            rx.recv();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (done, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            let mut buf = [0, 0];
            s2.read(buf).unwrap();
            tx2.send(());
            done.send(());
        });
        let mut buf = [0, 0];
        s1.read(buf).unwrap();
        tx1.send(());

        rx.recv();
    })

    iotest!(fn unix_clone_two_write() {
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

        let (tx, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            s2.write([1]).unwrap();
            tx.send(());
        });
        s1.write([2]).unwrap();

        rx.recv();
    })
}
