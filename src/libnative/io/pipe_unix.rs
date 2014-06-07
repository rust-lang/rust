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
use std::c_str::CString;
use std::mem;
use std::rt::mutex;
use std::rt::rtio;
use std::rt::rtio::{IoResult, IoError};

use super::retry;
use super::net;
use super::util;
use super::c;
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
    let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
    let s: &mut libc::sockaddr_un = unsafe { mem::transmute(&mut storage) };

    let len = addr.len();
    if len > s.sun_path.len() - 1 {
        #[cfg(unix)] use ERROR = libc::EINVAL;
        #[cfg(windows)] use ERROR = libc::WSAEINVAL;
        return Err(IoError {
            code: ERROR as uint,
            extra: 0,
            detail: Some("path must be smaller than SUN_LEN".to_str()),
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
    lock: mutex::NativeMutex,
}

impl Inner {
    fn new(fd: fd_t) -> Inner {
        Inner { fd: fd, lock: unsafe { mutex::NativeMutex::new() } }
    }
}

impl Drop for Inner {
    fn drop(&mut self) { unsafe { let _ = libc::close(self.fd); } }
}

fn connect(addr: &CString, ty: libc::c_int,
           timeout: Option<u64>) -> IoResult<Inner> {
    let (addr, len) = try!(addr_to_sockaddr_un(addr));
    let inner = Inner::new(try!(unix_socket(ty)));
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
    let inner = Inner::new(try!(unix_socket(ty)));
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
    inner: Arc<Inner>,
    read_deadline: u64,
    write_deadline: u64,
}

impl UnixStream {
    pub fn connect(addr: &CString,
                   timeout: Option<u64>) -> IoResult<UnixStream> {
        connect(addr, libc::SOCK_STREAM, timeout).map(|inner| {
            UnixStream::new(Arc::new(inner))
        })
    }

    fn new(inner: Arc<Inner>) -> UnixStream {
        UnixStream {
            inner: inner,
            read_deadline: 0,
            write_deadline: 0,
        }
    }

    fn fd(&self) -> fd_t { self.inner.fd }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) {}

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> net::Guard<'a> {
        let ret = net::Guard {
            fd: self.fd(),
            guard: unsafe { self.inner.lock.lock() },
        };
        assert!(util::set_nonblocking(self.fd(), true).is_ok());
        ret
    }
}

impl rtio::RtioPipe for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let doread = |nb| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::recv(fd,
                       buf.as_mut_ptr() as *mut libc::c_void,
                       buf.len() as libc::size_t,
                       flags) as libc::c_int
        };
        net::read(fd, self.read_deadline, dolock, doread)
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let dowrite = |nb: bool, buf: *u8, len: uint| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::send(fd,
                       buf as *mut libc::c_void,
                       len as libc::size_t,
                       flags) as i64
        };
        match net::write(fd, self.write_deadline, buf, true, dolock, dowrite) {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }

    fn clone(&self) -> Box<rtio::RtioPipe:Send> {
        box UnixStream::new(self.inner.clone()) as Box<rtio::RtioPipe:Send>
    }

    fn close_write(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_WR) })
    }
    fn close_read(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_RD) })
    }
    fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }
    fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
    fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
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
            try!(util::await(self.fd(), Some(self.deadline), util::Readable));
        }
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let storagep = &mut storage as *mut libc::sockaddr_storage;
        let size = mem::size_of::<libc::sockaddr_storage>();
        let mut size = size as libc::socklen_t;
        match retry(|| unsafe {
            libc::accept(self.fd(),
                         storagep as *mut libc::sockaddr,
                         &mut size as *mut libc::socklen_t) as libc::c_int
        }) {
            -1 => Err(super::last_error()),
            fd => Ok(UnixStream::new(Arc::new(Inner::new(fd))))
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
