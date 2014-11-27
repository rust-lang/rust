// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::arc::Arc;
use libc;
use c_str::CString;
use mem;
use rustrt::mutex;
use sync::atomic;
use io::{mod, IoResult, IoError};
use prelude::*;

use sys::{mod, timer, retry, c, set_nonblocking, wouldblock};
use sys::fs::{fd_t, FileDesc};
use sys_common::net::*;
use sys_common::{eof, mkerr_libc};

fn unix_socket(ty: libc::c_int) -> IoResult<fd_t> { unimplemented!() }

fn addr_to_sockaddr_un(addr: &CString,
                       storage: &mut libc::sockaddr_storage)
                       -> IoResult<libc::socklen_t> { unimplemented!() }

struct Inner {
    fd: fd_t,

    // Unused on Linux, where this lock is not necessary.
    #[allow(dead_code)]
    lock: mutex::NativeMutex
}

impl Inner {
    fn new(fd: fd_t) -> Inner { unimplemented!() }
}

impl Drop for Inner {
    fn drop(&mut self) { unimplemented!() }
}

fn connect(addr: &CString, ty: libc::c_int,
           timeout: Option<u64>) -> IoResult<Inner> { unimplemented!() }

fn bind(addr: &CString, ty: libc::c_int) -> IoResult<Inner> { unimplemented!() }

////////////////////////////////////////////////////////////////////////////////
// Unix Streams
////////////////////////////////////////////////////////////////////////////////

pub struct UnixStream {
    inner: Arc<Inner>,
    read_deadline: u64,
    write_deadline: u64,
}

impl UnixStream {
    pub fn connect(addr: &CString,
                   timeout: Option<u64>) -> IoResult<UnixStream> { unimplemented!() }

    fn new(inner: Arc<Inner>) -> UnixStream { unimplemented!() }

    fn fd(&self) -> fd_t { unimplemented!() }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) { unimplemented!() }

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> { unimplemented!() }

    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }

    pub fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    pub fn close_write(&mut self) -> IoResult<()> { unimplemented!() }

    pub fn close_read(&mut self) -> IoResult<()> { unimplemented!() }

    pub fn set_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }

    pub fn set_read_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }

    pub fn set_write_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }
}

impl Clone for UnixStream {
    fn clone(&self) -> UnixStream { unimplemented!() }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Listener
////////////////////////////////////////////////////////////////////////////////

pub struct UnixListener {
    inner: Inner,
    path: CString,
}

impl UnixListener {
    pub fn bind(addr: &CString) -> IoResult<UnixListener> { unimplemented!() }

    fn fd(&self) -> fd_t { unimplemented!() }

    pub fn listen(self) -> IoResult<UnixAcceptor> { unimplemented!() }
}

pub struct UnixAcceptor {
    inner: Arc<AcceptorInner>,
    deadline: u64,
}

struct AcceptorInner {
    listener: UnixListener,
    reader: FileDesc,
    writer: FileDesc,
    closed: atomic::AtomicBool,
}

impl UnixAcceptor {
    fn fd(&self) -> fd_t { unimplemented!() }

    pub fn accept(&mut self) -> IoResult<UnixStream> { unimplemented!() }

    pub fn set_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }

    pub fn close_accept(&mut self) -> IoResult<()> { unimplemented!() }
}

impl Clone for UnixAcceptor {
    fn clone(&self) -> UnixAcceptor { unimplemented!() }
}

impl Drop for UnixListener {
    fn drop(&mut self) { unimplemented!() }
}
