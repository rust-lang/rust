// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Named pipes
//!
//! This module contains the ability to communicate over named pipes with
//! synchronous I/O. On windows, this corresponds to talking over a Named Pipe,
//! while on Unix it corresponds to UNIX domain sockets.
//!
//! These pipes are similar to TCP in the sense that you can have both a stream to a
//! server and a server itself. The server provided accepts other `UnixStream`
//! instances as clients.

#![allow(missing_docs)]

use prelude::v1::*;

use ffi::CString;
use old_path::BytesContainer;
use old_io::{Listener, Acceptor, IoResult, TimedOut, standard_error};
use sys::pipe::UnixAcceptor as UnixAcceptorImp;
use sys::pipe::UnixListener as UnixListenerImp;
use sys::pipe::UnixStream as UnixStreamImp;
use time::Duration;

use sys_common;

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
    /// use std::old_io::net::pipe::UnixStream;
    ///
    /// let server = Path::new("path/to/my/socket");
    /// let mut stream = UnixStream::connect(&server);
    /// stream.write(&[1, 2, 3]);
    /// ```
    pub fn connect<P: BytesContainer>(path: P) -> IoResult<UnixStream> {
        let path = try!(CString::new(path.container_as_bytes()));
        UnixStreamImp::connect(&path, None)
            .map(|inner| UnixStream { inner: inner })
    }

    /// Connect to a pipe named by `path`, timing out if the specified number of
    /// milliseconds.
    ///
    /// This function is similar to `connect`, except that if `timeout`
    /// elapses the function will return an error of kind `TimedOut`.
    ///
    /// If a `timeout` with zero or negative duration is specified then
    /// the function returns `Err`, with the error kind set to `TimedOut`.
    #[unstable(feature = "io",
               reason = "the timeout argument is likely to change types")]
    pub fn connect_timeout<P>(path: P, timeout: Duration)
                              -> IoResult<UnixStream>
                              where P: BytesContainer {
        if timeout <= Duration::milliseconds(0) {
            return Err(standard_error(TimedOut));
        }

        let path = try!(CString::new(path.container_as_bytes()));
        UnixStreamImp::connect(&path, Some(timeout.num_milliseconds() as u64))
            .map(|inner| UnixStream { inner: inner })
    }


    /// Closes the reading half of this connection.
    ///
    /// This method will close the reading portion of this connection, causing
    /// all pending and future reads to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_read(&mut self) -> IoResult<()> {
        self.inner.close_read()
    }

    /// Closes the writing half of this connection.
    ///
    /// This method will close the writing portion of this connection, causing
    /// all pending and future writes to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_write(&mut self) -> IoResult<()> {
        self.inner.close_write()
    }

    /// Sets the read/write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[unstable(feature = "io",
               reason = "the timeout argument may change in type and value")]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_timeout(timeout_ms)
    }

    /// Sets the read timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[unstable(feature = "io",
               reason = "the timeout argument may change in type and value")]
    pub fn set_read_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_read_timeout(timeout_ms)
    }

    /// Sets the write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[unstable(feature = "io",
               reason = "the timeout argument may change in type and value")]
    pub fn set_write_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_write_timeout(timeout_ms)
    }
}

impl Clone for UnixStream {
    fn clone(&self) -> UnixStream {
        UnixStream { inner: self.inner.clone() }
    }
}

impl Reader for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.inner.read(buf)
    }
}

impl Writer for UnixStream {
    fn write_all(&mut self, buf: &[u8]) -> IoResult<()> {
        self.inner.write(buf)
    }
}

impl sys_common::AsInner<UnixStreamImp> for UnixStream {
    fn as_inner(&self) -> &UnixStreamImp {
        &self.inner
    }
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
    /// # fn foo() {
    /// use std::old_io::net::pipe::UnixListener;
    /// use std::old_io::{Listener, Acceptor};
    ///
    /// let server = Path::new("/path/to/my/socket");
    /// let stream = UnixListener::bind(&server);
    /// for mut client in stream.listen().incoming() {
    ///     client.write(&[1, 2, 3, 4]);
    /// }
    /// # }
    /// ```
    pub fn bind<P: BytesContainer>(path: P) -> IoResult<UnixListener> {
        let path = try!(CString::new(path.container_as_bytes()));
        UnixListenerImp::bind(&path)
            .map(|inner| UnixListener { inner: inner })
    }
}

impl Listener<UnixStream, UnixAcceptor> for UnixListener {
    fn listen(self) -> IoResult<UnixAcceptor> {
        self.inner.listen()
            .map(|inner| UnixAcceptor { inner: inner })
    }
}

impl sys_common::AsInner<UnixListenerImp> for UnixListener {
    fn as_inner(&self) -> &UnixListenerImp {
        &self.inner
    }
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
    #[unstable(feature = "io",
               reason = "the name and arguments to this function are likely \
                         to change")]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_timeout(timeout_ms)
    }

    /// Closes the accepting capabilities of this acceptor.
    ///
    /// This function has the same semantics as `TcpAcceptor::close_accept`, and
    /// more information can be found in that documentation.
    #[unstable(feature = "io")]
    pub fn close_accept(&mut self) -> IoResult<()> {
        self.inner.close_accept()
    }
}

impl Acceptor<UnixStream> for UnixAcceptor {
    fn accept(&mut self) -> IoResult<UnixStream> {
        self.inner.accept().map(|s| {
            UnixStream { inner: s }
        })
    }
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
    fn clone(&self) -> UnixAcceptor {
        UnixAcceptor { inner: self.inner.clone() }
    }
}

impl sys_common::AsInner<UnixAcceptorImp> for UnixAcceptor {
    fn as_inner(&self) -> &UnixAcceptorImp {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use old_io::fs::PathExtensions;
    use old_io::{EndOfFile, TimedOut, ShortWrite, IoError, ConnectionReset};
    use old_io::{NotConnected, BrokenPipe, FileNotFound, InvalidInput, OtherIoError};
    use old_io::{PermissionDenied, Acceptor, Listener};
    use old_io::test::*;
    use super::*;
    use sync::mpsc::channel;
    use thread;
    use time::Duration;

    pub fn smalltest<F,G>(server: F, client: G)
        where F : FnOnce(UnixStream), F : Send,
              G : FnOnce(UnixStream), G : Send + 'static
    {
        let path1 = next_test_unix();
        let path2 = path1.clone();

        let mut acceptor = UnixListener::bind(&path1).listen();

        let _t = thread::spawn(move|| {
            match UnixStream::connect(&path2) {
                Ok(c) => client(c),
                Err(e) => panic!("failed connect: {}", e),
            }
        });

        match acceptor.accept() {
            Ok(c) => server(c),
            Err(e) => panic!("failed accept: {}", e),
        }
    }

    #[test]
    fn bind_error() {
        let path = "path/to/nowhere";
        match UnixListener::bind(&path) {
            Ok(..) => panic!(),
            Err(e) => {
                assert!(e.kind == PermissionDenied || e.kind == FileNotFound ||
                        e.kind == InvalidInput);
            }
        }
    }

    #[test]
    fn connect_error() {
        let path = if cfg!(windows) {
            r"\\.\pipe\this_should_not_exist_ever"
        } else {
            "path/to/nowhere"
        };
        match UnixStream::connect(&path) {
            Ok(..) => panic!(),
            Err(e) => {
                assert!(e.kind == FileNotFound || e.kind == OtherIoError);
            }
        }
    }

    #[test]
    fn smoke() {
        smalltest(move |mut server| {
            let mut buf = [0];
            server.read(&mut buf).unwrap();
            assert!(buf[0] == 99);
        }, move|mut client| {
            client.write(&[99]).unwrap();
        })
    }

    #[cfg_attr(windows, ignore)] // FIXME(#12516)
    #[test]
    fn read_eof() {
        smalltest(move|mut server| {
            let mut buf = [0];
            assert!(server.read(&mut buf).is_err());
            assert!(server.read(&mut buf).is_err());
        }, move|_client| {
            // drop the client
        })
    }

    #[test]
    fn write_begone() {
        smalltest(move|mut server| {
            let buf = [0];
            loop {
                match server.write(&buf) {
                    Ok(..) => {}
                    Err(e) => {
                        assert!(e.kind == BrokenPipe ||
                                e.kind == NotConnected ||
                                e.kind == ConnectionReset,
                                "unknown error {}", e);
                        break;
                    }
                }
            }
        }, move|_client| {
            // drop the client
        })
    }

    #[test]
    fn accept_lots() {
        let times = 10;
        let path1 = next_test_unix();
        let path2 = path1.clone();

        let mut acceptor = match UnixListener::bind(&path1).listen() {
            Ok(a) => a,
            Err(e) => panic!("failed listen: {}", e),
        };

        let _t = thread::spawn(move|| {
            for _ in 0..times {
                let mut stream = UnixStream::connect(&path2);
                match stream.write(&[100]) {
                    Ok(..) => {}
                    Err(e) => panic!("failed write: {}", e)
                }
            }
        });

        for _ in 0..times {
            let mut client = acceptor.accept();
            let mut buf = [0];
            match client.read(&mut buf) {
                Ok(..) => {}
                Err(e) => panic!("failed read/accept: {}", e),
            }
            assert_eq!(buf[0], 100);
        }
    }

    #[cfg(unix)]
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

        let _t = thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr);
            let mut buf = [0, 0];
            debug!("client reading");
            assert_eq!(s.read(&mut buf), Ok(1));
            assert_eq!(buf[0], 1);
            debug!("client writing");
            s.write(&[2]).unwrap();
            debug!("client dropping");
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        let _t = thread::spawn(move|| {
            let mut s2 = s2;
            rx1.recv().unwrap();
            debug!("writer writing");
            s2.write(&[1]).unwrap();
            debug!("writer done");
            tx2.send(()).unwrap();
        });
        tx1.send(()).unwrap();
        let mut buf = [0, 0];
        debug!("reader reading");
        assert_eq!(s1.read(&mut buf), Ok(1));
        debug!("reader done");
        rx2.recv().unwrap();
    }

    #[test]
    fn unix_clone_two_read() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();

        let _t = thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr);
            s.write(&[1]).unwrap();
            rx.recv().unwrap();
            s.write(&[2]).unwrap();
            rx.recv().unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (done, rx) = channel();
        let _t = thread::spawn(move|| {
            let mut s2 = s2;
            let mut buf = [0, 0];
            s2.read(&mut buf).unwrap();
            tx2.send(()).unwrap();
            done.send(()).unwrap();
        });
        let mut buf = [0, 0];
        s1.read(&mut buf).unwrap();
        tx1.send(()).unwrap();

        rx.recv().unwrap();
    }

    #[test]
    fn unix_clone_two_write() {
        let addr = next_test_unix();
        let mut acceptor = UnixListener::bind(&addr).listen();

        let _t = thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr);
            let buf = &mut [0, 1];
            s.read(buf).unwrap();
            s.read(buf).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            let mut s2 = s2;
            s2.write(&[1]).unwrap();
            tx.send(()).unwrap();
        });
        s1.write(&[2]).unwrap();

        rx.recv().unwrap();
    }

    #[cfg(not(windows))]
    #[test]
    fn drop_removes_listener_path() {
        let path = next_test_unix();
        let l = UnixListener::bind(&path).unwrap();
        assert!(path.exists());
        drop(l);
        assert!(!path.exists());
    }

    #[cfg(not(windows))]
    #[test]
    fn drop_removes_acceptor_path() {
        let path = next_test_unix();
        let l = UnixListener::bind(&path).unwrap();
        assert!(path.exists());
        drop(l.listen().unwrap());
        assert!(!path.exists());
    }

    #[test]
    fn accept_timeout() {
        let addr = next_test_unix();
        let mut a = UnixListener::bind(&addr).unwrap().listen().unwrap();

        a.set_timeout(Some(10));

        // Make sure we time out once and future invocations also time out
        let err = a.accept().err().unwrap();
        assert_eq!(err.kind, TimedOut);
        let err = a.accept().err().unwrap();
        assert_eq!(err.kind, TimedOut);

        // Also make sure that even though the timeout is expired that we will
        // continue to receive any pending connections.
        let (tx, rx) = channel();
        let addr2 = addr.clone();
        let _t = thread::spawn(move|| {
            tx.send(UnixStream::connect(&addr2).unwrap()).unwrap();
        });
        let l = rx.recv().unwrap();
        for i in 0..1001 {
            match a.accept() {
                Ok(..) => break,
                Err(ref e) if e.kind == TimedOut => {}
                Err(e) => panic!("error: {}", e),
            }
            ::thread::yield_now();
            if i == 1000 { panic!("should have a pending connection") }
        }
        drop(l);

        // Unset the timeout and make sure that this always blocks.
        a.set_timeout(None);
        let addr2 = addr.clone();
        let _t = thread::spawn(move|| {
            drop(UnixStream::connect(&addr2).unwrap());
        });
        a.accept().unwrap();
    }

    #[test]
    fn connect_timeout_error() {
        let addr = next_test_unix();
        assert!(UnixStream::connect_timeout(&addr, Duration::milliseconds(100)).is_err());
    }

    #[test]
    fn connect_timeout_success() {
        let addr = next_test_unix();
        let _a = UnixListener::bind(&addr).unwrap().listen().unwrap();
        assert!(UnixStream::connect_timeout(&addr, Duration::milliseconds(100)).is_ok());
    }

    #[test]
    fn connect_timeout_zero() {
        let addr = next_test_unix();
        let _a = UnixListener::bind(&addr).unwrap().listen().unwrap();
        assert!(UnixStream::connect_timeout(&addr, Duration::milliseconds(0)).is_err());
    }

    #[test]
    fn connect_timeout_negative() {
        let addr = next_test_unix();
        let _a = UnixListener::bind(&addr).unwrap().listen().unwrap();
        assert!(UnixStream::connect_timeout(&addr, Duration::milliseconds(-1)).is_err());
    }

    #[test]
    fn close_readwrite_smoke() {
        let addr = next_test_unix();
        let a = UnixListener::bind(&addr).listen().unwrap();
        let (_tx, rx) = channel::<()>();
        thread::spawn(move|| {
            let mut a = a;
            let _s = a.accept().unwrap();
            let _ = rx.recv();
        });

        let mut b = [0];
        let mut s = UnixStream::connect(&addr).unwrap();
        let mut s2 = s.clone();

        // closing should prevent reads/writes
        s.close_write().unwrap();
        assert!(s.write(&[0]).is_err());
        s.close_read().unwrap();
        assert!(s.read(&mut b).is_err());

        // closing should affect previous handles
        assert!(s2.write(&[0]).is_err());
        assert!(s2.read(&mut b).is_err());

        // closing should affect new handles
        let mut s3 = s.clone();
        assert!(s3.write(&[0]).is_err());
        assert!(s3.read(&mut b).is_err());

        // make sure these don't die
        let _ = s2.close_read();
        let _ = s2.close_write();
        let _ = s3.close_read();
        let _ = s3.close_write();
    }

    #[test]
    fn close_read_wakes_up() {
        let addr = next_test_unix();
        let a = UnixListener::bind(&addr).listen().unwrap();
        let (_tx, rx) = channel::<()>();
        thread::spawn(move|| {
            let mut a = a;
            let _s = a.accept().unwrap();
            let _ = rx.recv();
        });

        let mut s = UnixStream::connect(&addr).unwrap();
        let s2 = s.clone();
        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            let mut s2 = s2;
            assert!(s2.read(&mut [0]).is_err());
            tx.send(()).unwrap();
        });
        // this should wake up the child task
        s.close_read().unwrap();

        // this test will never finish if the child doesn't wake up
        rx.recv().unwrap();
    }

    #[test]
    fn readwrite_timeouts() {
        let addr = next_test_unix();
        let mut a = UnixListener::bind(&addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr).unwrap();
            rx.recv().unwrap();
            assert!(s.write(&[0]).is_ok());
            let _ = rx.recv();
        });

        let mut s = a.accept().unwrap();
        s.set_timeout(Some(20));
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);

        s.set_timeout(Some(20));
        for i in 0..1001 {
            match s.write(&[0; 128 * 1024]) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => panic!("{}", e),
           }
           if i == 1000 { panic!("should have filled up?!"); }
        }

        // I'm not sure as to why, but apparently the write on windows always
        // succeeds after the previous timeout. Who knows?
        if !cfg!(windows) {
            assert_eq!(s.write(&[0]).err().unwrap().kind, TimedOut);
        }

        tx.send(()).unwrap();
        s.set_timeout(None);
        assert_eq!(s.read(&mut [0, 0]), Ok(1));
    }

    #[test]
    fn read_timeouts() {
        let addr = next_test_unix();
        let mut a = UnixListener::bind(&addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr).unwrap();
            rx.recv().unwrap();
            let mut amt = 0;
            while amt < 100 * 128 * 1024 {
                match s.read(&mut [0;128 * 1024]) {
                    Ok(n) => { amt += n; }
                    Err(e) => panic!("{}", e),
                }
            }
            let _ = rx.recv();
        });

        let mut s = a.accept().unwrap();
        s.set_read_timeout(Some(20));
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);

        tx.send(()).unwrap();
        for _ in 0..100 {
            assert!(s.write(&[0;128 * 1024]).is_ok());
        }
    }

    #[test]
    fn write_timeouts() {
        let addr = next_test_unix();
        let mut a = UnixListener::bind(&addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr).unwrap();
            rx.recv().unwrap();
            assert!(s.write(&[0]).is_ok());
            let _ = rx.recv();
        });

        let mut s = a.accept().unwrap();
        s.set_write_timeout(Some(20));
        for i in 0..1001 {
            match s.write(&[0; 128 * 1024]) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => panic!("{}", e),
           }
           if i == 1000 { panic!("should have filled up?!"); }
        }

        tx.send(()).unwrap();
        assert!(s.read(&mut [0]).is_ok());
    }

    #[test]
    fn timeout_concurrent_read() {
        let addr = next_test_unix();
        let mut a = UnixListener::bind(&addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        thread::spawn(move|| {
            let mut s = UnixStream::connect(&addr).unwrap();
            rx.recv().unwrap();
            assert!(s.write(&[0]).is_ok());
            let _ = rx.recv();
        });

        let mut s = a.accept().unwrap();
        let s2 = s.clone();
        let (tx2, rx2) = channel();
        let _t = thread::spawn(move|| {
            let mut s2 = s2;
            assert!(s2.read(&mut [0]).is_ok());
            tx2.send(()).unwrap();
        });

        s.set_read_timeout(Some(20));
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);
        tx.send(()).unwrap();

        rx2.recv().unwrap();
    }

    #[cfg(not(windows))]
    #[test]
    fn clone_accept_smoke() {
        let addr = next_test_unix();
        let l = UnixListener::bind(&addr);
        let mut a = l.listen().unwrap();
        let mut a2 = a.clone();

        let addr2 = addr.clone();
        let _t = thread::spawn(move|| {
            let _ = UnixStream::connect(&addr2);
        });
        let _t = thread::spawn(move|| {
            let _ = UnixStream::connect(&addr);
        });

        assert!(a.accept().is_ok());
        drop(a);
        assert!(a2.accept().is_ok());
    }

    #[cfg(not(windows))] // FIXME #17553
    #[test]
    fn clone_accept_concurrent() {
        let addr = next_test_unix();
        let l = UnixListener::bind(&addr);
        let a = l.listen().unwrap();
        let a2 = a.clone();

        let (tx, rx) = channel();
        let tx2 = tx.clone();

        let _t = thread::spawn(move|| {
            let mut a = a;
            tx.send(a.accept()).unwrap()
        });
        let _t = thread::spawn(move|| {
            let mut a = a2;
            tx2.send(a.accept()).unwrap()
        });

        let addr2 = addr.clone();
        let _t = thread::spawn(move|| {
            let _ = UnixStream::connect(&addr2);
        });
        let _t = thread::spawn(move|| {
            let _ = UnixStream::connect(&addr);
        });

        assert!(rx.recv().unwrap().is_ok());
        assert!(rx.recv().unwrap().is_ok());
    }

    #[test]
    fn close_accept_smoke() {
        let addr = next_test_unix();
        let l = UnixListener::bind(&addr);
        let mut a = l.listen().unwrap();

        a.close_accept().unwrap();
        assert_eq!(a.accept().err().unwrap().kind, EndOfFile);
    }

    #[test]
    fn close_accept_concurrent() {
        let addr = next_test_unix();
        let l = UnixListener::bind(&addr);
        let a = l.listen().unwrap();
        let mut a2 = a.clone();

        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            let mut a = a;
            tx.send(a.accept()).unwrap();
        });
        a2.close_accept().unwrap();

        assert_eq!(rx.recv().unwrap().err().unwrap().kind, EndOfFile);
    }
}
