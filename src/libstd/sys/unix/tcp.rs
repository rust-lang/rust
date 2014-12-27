// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::net::ip;
use io::IoResult;
use libc;
use mem;
use ptr;
use prelude::*;
use super::{last_error, last_net_error, retry, sock_t};
use sync::{Arc, atomic};
use sys::fs::FileDesc;
use sys::{set_nonblocking, wouldblock};
use sys;
use sys_common;
use sys_common::net;
use sys_common::net::SocketStatus::Readable;

pub use sys_common::net::TcpStream;

////////////////////////////////////////////////////////////////////////////////
// TCP listeners
////////////////////////////////////////////////////////////////////////////////

pub struct TcpListener {
    pub inner: FileDesc,
}

unsafe impl Sync for TcpListener {}

impl TcpListener {
    pub fn bind(addr: ip::SocketAddr) -> IoResult<TcpListener> {
        let fd = try!(net::socket(addr, libc::SOCK_STREAM));
        let ret = TcpListener { inner: FileDesc::new(fd, true) };

        let mut storage = unsafe { mem::zeroed() };
        let len = net::addr_to_sockaddr(addr, &mut storage);
        let addrp = &storage as *const _ as *const libc::sockaddr;

        // On platforms with Berkeley-derived sockets, this allows
        // to quickly rebind a socket, without needing to wait for
        // the OS to clean up the previous one.
        try!(net::setsockopt(fd, libc::SOL_SOCKET,
                             libc::SO_REUSEADDR,
                             1 as libc::c_int));


        match unsafe { libc::bind(fd, addrp, len) } {
            -1 => Err(last_error()),
            _ => Ok(ret),
        }
    }

    pub fn fd(&self) -> sock_t { self.inner.fd() }

    pub fn listen(self, backlog: int) -> IoResult<TcpAcceptor> {
        match unsafe { libc::listen(self.fd(), backlog as libc::c_int) } {
            -1 => Err(last_net_error()),
            _ => {
                let (reader, writer) = try!(unsafe { sys::os::pipe() });
                try!(set_nonblocking(reader.fd(), true));
                try!(set_nonblocking(writer.fd(), true));
                try!(set_nonblocking(self.fd(), true));
                Ok(TcpAcceptor {
                    inner: Arc::new(AcceptorInner {
                        listener: self,
                        reader: reader,
                        writer: writer,
                        closed: atomic::AtomicBool::new(false),
                    }),
                    deadline: 0,
                })
            }
        }
    }

    pub fn socket_name(&mut self) -> IoResult<ip::SocketAddr> {
        net::sockname(self.fd(), libc::getsockname)
    }
}

pub struct TcpAcceptor {
    inner: Arc<AcceptorInner>,
    deadline: u64,
}

struct AcceptorInner {
    listener: TcpListener,
    reader: FileDesc,
    writer: FileDesc,
    closed: atomic::AtomicBool,
}

unsafe impl Sync for AcceptorInner {}

impl TcpAcceptor {
    pub fn fd(&self) -> sock_t { self.inner.listener.fd() }

    pub fn accept(&mut self) -> IoResult<TcpStream> {
        // In implementing accept, the two main concerns are dealing with
        // close_accept() and timeouts. The unix implementation is based on a
        // nonblocking accept plus a call to select(). Windows ends up having
        // an entirely separate implementation than unix, which is explained
        // below.
        //
        // To implement timeouts, all blocking is done via select() instead of
        // accept() by putting the socket in non-blocking mode. Because
        // select() takes a timeout argument, we just pass through the timeout
        // to select().
        //
        // To implement close_accept(), we have a self-pipe to ourselves which
        // is passed to select() along with the socket being accepted on. The
        // self-pipe is never written to unless close_accept() is called.
        let deadline = if self.deadline == 0 {None} else {Some(self.deadline)};

        while !self.inner.closed.load(atomic::SeqCst) {
            match retry(|| unsafe {
                libc::accept(self.fd(), ptr::null_mut(), ptr::null_mut())
            }) {
                -1 if wouldblock() => {}
                -1 => return Err(last_net_error()),
                fd => return Ok(TcpStream::new(fd as sock_t)),
            }
            try!(net::await(&[self.fd(), self.inner.reader.fd()],
                       deadline, Readable));
        }

        Err(sys_common::eof())
    }

    pub fn socket_name(&mut self) -> IoResult<ip::SocketAddr> {
        net::sockname(self.fd(), libc::getsockname)
    }

    pub fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|a| sys::timer::now() + a).unwrap_or(0);
    }

    pub fn close_accept(&mut self) -> IoResult<()> {
        self.inner.closed.store(true, atomic::SeqCst);
        let fd = FileDesc::new(self.inner.writer.fd(), false);
        match fd.write(&[0]) {
            Ok(..) => Ok(()),
            Err(..) if wouldblock() => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl Clone for TcpAcceptor {
    fn clone(&self) -> TcpAcceptor {
        TcpAcceptor {
            inner: self.inner.clone(),
            deadline: 0,
        }
    }
}
