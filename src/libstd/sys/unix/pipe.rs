// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use ffi::CString;
use libc;
use mem;
use sync::{Arc, Mutex};
use sync::atomic::{AtomicBool, Ordering};
use old_io::{self, IoResult, IoError};

use sys::{self, timer, retry, c, set_nonblocking, wouldblock};
use sys::fs::{fd_t, FileDesc};
use sys_common::net::*;
use sys_common::net::SocketStatus::*;
use sys_common::{eof, mkerr_libc};

fn unix_socket(ty: libc::c_int) -> IoResult<fd_t> {
    match unsafe { libc::socket(libc::AF_UNIX, ty, 0) } {
        -1 => Err(super::last_error()),
        fd => Ok(fd)
    }
}

fn addr_to_sockaddr_un(addr: &CString,
                       storage: &mut libc::sockaddr_storage)
                       -> IoResult<libc::socklen_t> {
    // the sun_path length is limited to SUN_LEN (with null)
    assert!(mem::size_of::<libc::sockaddr_storage>() >=
            mem::size_of::<libc::sockaddr_un>());
    let s = unsafe { &mut *(storage as *mut _ as *mut libc::sockaddr_un) };

    let len = addr.as_bytes().len();
    if len > s.sun_path.len() - 1 {
        return Err(IoError {
            kind: old_io::InvalidInput,
            desc: "invalid argument: path must be smaller than SUN_LEN",
            detail: None,
        })
    }
    s.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, value) in s.sun_path.iter_mut().zip(addr.as_bytes().iter()) {
        *slot = *value as libc::c_char;
    }

    // count the null terminator
    let len = mem::size_of::<libc::sa_family_t>() + len + 1;
    return Ok(len as libc::socklen_t);
}

struct Inner {
    fd: fd_t,

    // Unused on Linux, where this lock is not necessary.
    #[allow(dead_code)]
    lock: Mutex<()>,
}

impl Inner {
    fn new(fd: fd_t) -> Inner {
        Inner { fd: fd, lock: Mutex::new(()) }
    }
}

impl Drop for Inner {
    fn drop(&mut self) { unsafe { let _ = libc::close(self.fd); } }
}

fn connect(addr: &CString, ty: libc::c_int,
           timeout: Option<u64>) -> IoResult<Inner> {
    let mut storage = unsafe { mem::zeroed() };
    let len = try!(addr_to_sockaddr_un(addr, &mut storage));
    let inner = Inner::new(try!(unix_socket(ty)));
    let addrp = &storage as *const _ as *const libc::sockaddr;

    match timeout {
        None => {
            match retry(|| unsafe { libc::connect(inner.fd, addrp, len) }) {
                -1 => Err(super::last_error()),
                _  => Ok(inner)
            }
        }
        Some(timeout_ms) => {
            try!(connect_timeout(inner.fd, addrp, len, timeout_ms));
            Ok(inner)
        }
    }
}

fn bind(addr: &CString, ty: libc::c_int) -> IoResult<Inner> {
    let mut storage = unsafe { mem::zeroed() };
    let len = try!(addr_to_sockaddr_un(addr, &mut storage));
    let inner = Inner::new(try!(unix_socket(ty)));
    let addrp = &storage as *const _ as *const libc::sockaddr;
    match unsafe {
        libc::bind(inner.fd, addrp, len)
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

    pub fn fd(&self) -> fd_t { self.inner.fd }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) {}

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> {
        let ret = Guard {
            fd: self.fd(),
            guard: unsafe { self.inner.lock.lock().unwrap() },
        };
        assert!(set_nonblocking(self.fd(), true).is_ok());
        ret
    }

    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let doread = |nb| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::recv(fd,
                       buf.as_mut_ptr() as *mut libc::c_void,
                       buf.len() as libc::size_t,
                       flags) as libc::c_int
        };
        read(fd, self.read_deadline, dolock, doread)
    }

    pub fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let dowrite = |nb: bool, buf: *const u8, len: uint| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::send(fd,
                       buf as *const _,
                       len as libc::size_t,
                       flags) as i64
        };
        match write(fd, self.write_deadline, buf, true, dolock, dowrite) {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }

    pub fn close_write(&mut self) -> IoResult<()> {
        mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_WR) })
    }

    pub fn close_read(&mut self) -> IoResult<()> {
        mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_RD) })
    }

    pub fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }

    pub fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }

    pub fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }
}

impl Clone for UnixStream {
    fn clone(&self) -> UnixStream {
        UnixStream::new(self.inner.clone())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Listener
////////////////////////////////////////////////////////////////////////////////

pub struct UnixListener {
    inner: Inner,
    path: CString,
}

// we currently own the CString, so these impls should be safe
unsafe impl Send for UnixListener {}
unsafe impl Sync for UnixListener {}

impl UnixListener {
    pub fn bind(addr: &CString) -> IoResult<UnixListener> {
        bind(addr, libc::SOCK_STREAM).map(|fd| {
            UnixListener { inner: fd, path: addr.clone() }
        })
    }

    pub fn fd(&self) -> fd_t { self.inner.fd }

    pub fn listen(self) -> IoResult<UnixAcceptor> {
        match unsafe { libc::listen(self.fd(), 128) } {
            -1 => Err(super::last_error()),

            _ => {
                let (reader, writer) = try!(unsafe { sys::os::pipe() });
                try!(set_nonblocking(reader.fd(), true));
                try!(set_nonblocking(writer.fd(), true));
                try!(set_nonblocking(self.fd(), true));
                Ok(UnixAcceptor {
                    inner: Arc::new(AcceptorInner {
                        listener: self,
                        reader: reader,
                        writer: writer,
                        closed: AtomicBool::new(false),
                    }),
                    deadline: 0,
                })
            }
        }
    }
}

pub struct UnixAcceptor {
    inner: Arc<AcceptorInner>,
    deadline: u64,
}

struct AcceptorInner {
    listener: UnixListener,
    reader: FileDesc,
    writer: FileDesc,
    closed: AtomicBool,
}

impl UnixAcceptor {
    pub fn fd(&self) -> fd_t { self.inner.listener.fd() }

    pub fn accept(&mut self) -> IoResult<UnixStream> {
        let deadline = if self.deadline == 0 {None} else {Some(self.deadline)};

        while !self.inner.closed.load(Ordering::SeqCst) {
            unsafe {
                let mut storage: libc::sockaddr_storage = mem::zeroed();
                let storagep = &mut storage as *mut libc::sockaddr_storage;
                let size = mem::size_of::<libc::sockaddr_storage>();
                let mut size = size as libc::socklen_t;
                match retry(|| {
                    libc::accept(self.fd(),
                                 storagep as *mut libc::sockaddr,
                                 &mut size as *mut libc::socklen_t) as libc::c_int
                }) {
                    -1 if wouldblock() => {}
                    -1 => return Err(super::last_error()),
                    fd => return Ok(UnixStream::new(Arc::new(Inner::new(fd)))),
                }
            }
            try!(await(&[self.fd(), self.inner.reader.fd()],
                       deadline, Readable));
        }

        Err(eof())
    }

    pub fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }

    pub fn close_accept(&mut self) -> IoResult<()> {
        self.inner.closed.store(true, Ordering::SeqCst);
        let fd = FileDesc::new(self.inner.writer.fd(), false);
        match fd.write(&[0]) {
            Ok(..) => Ok(()),
            Err(..) if wouldblock() => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl Clone for UnixAcceptor {
    fn clone(&self) -> UnixAcceptor {
        UnixAcceptor { inner: self.inner.clone(), deadline: 0 }
    }
}

impl Drop for UnixListener {
    fn drop(&mut self) {
        // Unlink the path to the socket to ensure that it doesn't linger. We're
        // careful to unlink the path before we close the file descriptor to
        // prevent races where we unlink someone else's path.
        unsafe {
            let _ = libc::unlink(self.path.as_ptr());
        }
    }
}
