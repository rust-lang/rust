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
use sys_common::net::*;

pub use sys_common::net::TcpStream;

////////////////////////////////////////////////////////////////////////////////
// TCP listeners
////////////////////////////////////////////////////////////////////////////////

pub struct TcpListener {
    pub inner: FileDesc,
}

impl TcpListener {
    pub fn bind(addr: ip::SocketAddr) -> IoResult<TcpListener> { unimplemented!() }

    pub fn fd(&self) -> sock_t { unimplemented!() }

    pub fn listen(self, backlog: int) -> IoResult<TcpAcceptor> { unimplemented!() }

    pub fn socket_name(&mut self) -> IoResult<ip::SocketAddr> { unimplemented!() }
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

impl TcpAcceptor {
    pub fn fd(&self) -> sock_t { unimplemented!() }

    pub fn accept(&mut self) -> IoResult<TcpStream> { unimplemented!() }

    pub fn socket_name(&mut self) -> IoResult<ip::SocketAddr> { unimplemented!() }

    pub fn set_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }

    pub fn close_accept(&mut self) -> IoResult<()> { unimplemented!() }
}

impl Clone for TcpAcceptor {
    fn clone(&self) -> TcpAcceptor { unimplemented!() }
}
