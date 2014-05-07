// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use std::c_str::CString;
use std::cast;
use std::intrinsics;
use std::io;
use std::mem;
use std::rt::rtio;
use std::sync::arc::UnsafeArc;

use super::{IoResult, retry, keep_going};
use super::util;
use super::file::fd_t;

fn unix_socket(ty: libc::c_int) -> IoResult<fd_t> {
    match unsafe { libc::socket(libc::AF_UNIX, ty, 0) } {
        -1 => Err(super::last_error()),
        fd => Ok(fd)
    }
}

fn addr_to_sockaddr_un(addr: &CString) -> IoResult<(libc::sockaddr_storage, uint)> {
    // the sun_path length is limited to SUN_LEN (with null)
    assert!(mem::size_of::<libc::sockaddr_storage>() >=
            mem::size_of::<libc::sockaddr_un>());
    let mut storage: libc::sockaddr_storage = unsafe { intrinsics::init() };
    let s: &mut libc::sockaddr_un = unsafe { cast::transmute(&mut storage) };

    let len = addr.len();
    if len > s.sun_path.len() - 1 {
        return Err(io::IoError {
            kind: io::InvalidInput,
            desc: "path must be smaller than SUN_LEN",
            detail: None,
        })
    }
    s.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, value) in s.sun_path.mut_iter().zip(addr.iter()) {
        *slot = value;
    }

    // count the null terminator
    let len = mem::size_of::<libc::sa_family_t>() + len + 1;
    return Ok((storage, len));
}

struct Inner {
    fd: fd_t,
}

impl Drop for Inner {
    fn drop(&mut self) { unsafe { let _ = libc::close(self.fd); } }
}

fn connect(addr: &CString, ty: libc::c_int,
           timeout: Option<u64>) -> IoResult<Inner> {
    let (addr, len) = try!(addr_to_sockaddr_un(addr));
    let inner = Inner { fd: try!(unix_socket(ty)) };
    let addrp = &addr as *_ as *libc::sockaddr;
    let len = len as libc::socklen_t;

    match timeout {
        None => {
            match retry(|| unsafe { libc::connect(inner.fd, addrp, len) }) {
                -1 => Err(super::last_error()),
                _  => Ok(inner)
            }
        }
        Some(timeout_ms) => {
            try!(util::connect_timeout(inner.fd, addrp, len, timeout_ms));
            Ok(inner)
        }
    }
}

fn bind(addr: &CString, ty: libc::c_int) -> IoResult<Inner> {
    let (addr, len) = try!(addr_to_sockaddr_un(addr));
    let inner = Inner { fd: try!(unix_socket(ty)) };
    let addrp = &addr as *libc::sockaddr_storage;
    match unsafe {
        libc::bind(inner.fd, addrp as *libc::sockaddr, len as libc::socklen_t)
    } {
        -1 => Err(super::last_error()),
        _  => Ok(inner)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Streams
////////////////////////////////////////////////////////////////////////////////

pub struct UnixStream {
    inner: UnsafeArc<Inner>,
}

impl UnixStream {
    pub fn connect(addr: &CString,
                   timeout: Option<u64>) -> IoResult<UnixStream> {
        connect(addr, libc::SOCK_STREAM, timeout).map(|inner| {
            UnixStream { inner: UnsafeArc::new(inner) }
        })
    }

    fn fd(&self) -> fd_t { unsafe { (*self.inner.get()).fd } }
}

impl rtio::RtioPipe for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let ret = retry(|| unsafe {
            libc::recv(self.fd(),
                       buf.as_ptr() as *mut libc::c_void,
                       buf.len() as libc::size_t,
                       0) as libc::c_int
        });
        if ret == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as uint)
        }
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let ret = keep_going(buf, |buf, len| unsafe {
            libc::send(self.fd(),
                       buf as *mut libc::c_void,
                       len as libc::size_t,
                       0) as i64
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }

    fn clone(&self) -> Box<rtio::RtioPipe:Send> {
        box UnixStream {
            inner: self.inner.clone(),
        } as Box<rtio::RtioPipe:Send>
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Listener
////////////////////////////////////////////////////////////////////////////////

pub struct UnixListener {
    inner: Inner,
    path: CString,
}

impl UnixListener {
    pub fn bind(addr: &CString) -> IoResult<UnixListener> {
        bind(addr, libc::SOCK_STREAM).map(|fd| {
            UnixListener { inner: fd, path: addr.clone() }
        })
    }

    fn fd(&self) -> fd_t { self.inner.fd }

    pub fn native_listen(self, backlog: int) -> IoResult<UnixAcceptor> {
        match unsafe { libc::listen(self.fd(), backlog as libc::c_int) } {
            -1 => Err(super::last_error()),
            _ => Ok(UnixAcceptor { listener: self, deadline: 0 })
        }
    }
}

impl rtio::RtioUnixListener for UnixListener {
    fn listen(~self) -> IoResult<Box<rtio::RtioUnixAcceptor:Send>> {
        self.native_listen(128).map(|a| {
            box a as Box<rtio::RtioUnixAcceptor:Send>
        })
    }
}

pub struct UnixAcceptor {
    listener: UnixListener,
    deadline: u64,
}

impl UnixAcceptor {
    fn fd(&self) -> fd_t { self.listener.fd() }

    pub fn native_accept(&mut self) -> IoResult<UnixStream> {
        if self.deadline != 0 {
            try!(util::accept_deadline(self.fd(), self.deadline));
        }
        let mut storage: libc::sockaddr_storage = unsafe { intrinsics::init() };
        let storagep = &mut storage as *mut libc::sockaddr_storage;
        let size = mem::size_of::<libc::sockaddr_storage>();
        let mut size = size as libc::socklen_t;
        match retry(|| unsafe {
            libc::accept(self.fd(),
                         storagep as *mut libc::sockaddr,
                         &mut size as *mut libc::socklen_t) as libc::c_int
        }) {
            -1 => Err(super::last_error()),
            fd => Ok(UnixStream { inner: UnsafeArc::new(Inner { fd: fd }) })
        }
    }
}

impl rtio::RtioUnixAcceptor for UnixAcceptor {
    fn accept(&mut self) -> IoResult<Box<rtio::RtioPipe:Send>> {
        self.native_accept().map(|s| box s as Box<rtio::RtioPipe:Send>)
    }
    fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
}

impl Drop for UnixListener {
    fn drop(&mut self) {
        // Unlink the path to the socket to ensure that it doesn't linger. We're
        // careful to unlink the path before we close the file descriptor to
        // prevent races where we unlink someone else's path.
        unsafe {
            let _ = libc::unlink(self.path.with_ref(|p| p));
        }
    }
}
