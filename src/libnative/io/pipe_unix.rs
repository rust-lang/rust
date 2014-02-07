// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

////////////////////////////////////////////////////////////////////////////////
// Unix
////////////////////////////////////////////////////////////////////////////////

pub struct UnixStream {
    priv fd: sock_t,
}

impl UnixStream {
    pub fn connect(addr: &CString, ty: libc::c_int) -> IoResult<UnixStream> {
        unix_socket(ty).and_then(|fd| {
            match addr_to_sockaddr_un(addr) {
                Err(e)          => return Err(e),
                Ok((addr, len)) => {
                    let ret = UnixStream{ fd: fd };
                    let addrp = &addr as *libc::sockaddr_storage;
                    match retry(|| {
                      libc::connect(fd, addrp as *libc::sockaddr,
                                     len as libc::socklen_t)
                    }) {
                        -1 => return Err(super::last_error()),
                        _  => return Ok(ret)
                    }
                }
            }
        })
    }

    pub fn fd(&self) -> sock_t { self.fd }
}

impl rtio::RtioPipe for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let ret = retry(|| {
            unsafe {
                libc::recv(self.fd,
                           buf.as_ptr() as *mut libc::c_void,
                           buf.len() as wrlen,
                           0) as libc::c_int
            }
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
        let ret = keep_going(buf, |buf, len| {
            unsafe {
                libc::send(self.fd,
                           buf as *mut libc::c_void,
                           len as wrlen,
                           0) as i64
            }
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for UnixStream {
    fn drop(&mut self) { unsafe { close(self.fd); } }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Datagram
////////////////////////////////////////////////////////////////////////////////

pub struct UnixDatagram {
    priv fd: sock_t,
}

impl UnixDatagram {
    pub fn connect(addr: &CString, ty: libc::c_int) -> IoResult<UnixDatagram> {
        unsafe {
            unix_socket(ty).and_then(|fd| {
                match addr_to_sockaddr_un(addr) {
                    Err(e)          => return Err(e),
                    Ok((addr, len)) => {
                        let ret = UnixDatagram{ fd: fd };
                        let addrp = &addr as *libc::sockaddr_storage;
                        match retry(|| {
                          libc::connect(fd, addrp as *libc::sockaddr,
                                         len as libc::socklen_t)
                        }) {
                            -1 => return Err(super::last_error()),
                            _  => return Ok(ret)
                        }
                    }
                }
            })
        }
    }

    pub fn bind(addr: &CString) -> IoResult<UnixDatagram> {
        unsafe {
            unix_socket(libc::SOCK_DGRAM).and_then(|fd| {
                match addr_to_sockaddr_un(addr) {
                    Err(e)          => return Err(e),
                    Ok((addr, len)) => {
                        let ret = UnixDatagram{ fd: fd };
                        let addrp = &addr as *libc::sockaddr_storage;
                        match libc::bind(fd, addrp as *libc::sockaddr,
                                         len as libc::socklen_t) {
                            -1 => return Err(super::last_error()),
                            _  => return Ok(ret)
                        }
                    }
                }
            })
        }
    }

    pub fn fd(&self) -> sock_t { self.fd }
}

impl rtio::RtioPipe for UnixDatagram {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let ret = retry(|| {
            unsafe {
                libc::recv(self.fd,
                           buf.as_ptr() as *mut libc::c_void,
                           buf.len() as wrlen,
                           0) as libc::c_int
            }
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
        let ret = keep_going(buf, |buf, len| {
            unsafe {
                libc::send(self.fd,
                           buf as *mut libc::c_void,
                           len as wrlen,
                           0) as i64
            }
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }
}

impl rtio::RtioDatagramPipe for UnixDatagram {
    fn recvfrom(&mut self, buf: &mut [u8]) -> IoResult<(uint, CString)> {
        unsafe {
            let mut storage: libc::sockaddr_storage = intrinsics::init();
            let storagep = &mut storage as *mut libc::sockaddr_storage;
            let mut addrlen: libc::socklen_t =
                    mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            let ret = retry(|| {
                libc::recvfrom(self.fd,
                               buf.as_ptr() as *mut libc::c_void,
                               buf.len() as msglen_t,
                               0,
                               storagep as *mut libc::sockaddr,
                               &mut addrlen) as libc::c_int
            });
            if ret < 0 { return Err(super::last_error()) }
            sockaddr_to_unix(&storage, addrlen as uint).and_then(|addr| {
                Ok((ret as uint, addr))
            })
        }
    }

    fn sendto(&mut self, buf: &[u8], dst: &CString) -> IoResult<()> {
        match addr_to_sockaddr_un(dst) {
            Err(e)         => Err(e),
            Ok((dst, len)) => {
                let dstp = &dst as *libc::sockaddr_storage;
                unsafe {
                    let ret = retry(|| {
                        libc::sendto(self.fd,
                                     buf.as_ptr() as *libc::c_void,
                                     buf.len() as msglen_t,
                                     0,
                                     dstp as *libc::sockaddr,
                                     len as libc::socklen_t) as libc::c_int
                    });
                    match ret {
                        -1 => Err(super::last_error()),
                        n if n as uint != buf.len() => {
                            Err(io::IoError {
                                kind: io::OtherIoError,
                                desc: "couldn't send entire packet at once",
                                detail: None,
                            })
                        }
                        _ => Ok(())
                    }
                }
            }
        }
    }
}

impl Drop for UnixDatagram {
    fn drop(&mut self) { unsafe { close(self.fd); } }
}
////////////////////////////////////////////////////////////////////////////////
// Unix Listener
////////////////////////////////////////////////////////////////////////////////

pub struct UnixListener {
    priv fd: sock_t,
}

impl UnixListener {
    pub fn bind(addr: &CString) -> IoResult<UnixListener> {
        unsafe {
            unix_socket(libc::SOCK_STREAM).and_then(|fd| {
                match addr_to_sockaddr_un(addr) {
                    Err(e)          => return Err(e),
                    Ok((addr, len)) => {
                        let ret = UnixListener{ fd: fd };
                        let addrp = &addr as *libc::sockaddr_storage;
                        match libc::bind(fd, addrp as *libc::sockaddr,
                                         len as libc::socklen_t) {
                            -1 => return Err(super::last_error()),
                            _  => return Ok(ret)
                        }
                    }
                }
            })
        }
    }

    pub fn fd(&self) -> sock_t { self.fd }

    pub fn native_listen(self, backlog: int) -> IoResult<UnixAcceptor> {
        match unsafe { libc::listen(self.fd, backlog as libc::c_int) } {
            -1 => Err(super::last_error()),
            _ => Ok(UnixAcceptor { listener: self })
        }
    }
}

impl rtio::RtioUnixListener for UnixListener {
    fn listen(~self) -> IoResult<~rtio::RtioUnixAcceptor> {
        self.native_listen(128).map(|a| ~a as ~rtio::RtioUnixAcceptor)
    }
}

impl Drop for UnixListener {
    fn drop(&mut self) { unsafe { close(self.fd); } }
}

pub struct UnixAcceptor {
    priv listener: UnixListener,
}

impl UnixAcceptor {
    pub fn fd(&self) -> sock_t { self.listener.fd }

    pub fn native_accept(&mut self) -> IoResult<UnixStream> {
        let mut storage: libc::sockaddr_storage = unsafe { intrinsics::init() };
        let storagep = &mut storage as *mut libc::sockaddr_storage;
        let size = mem::size_of::<libc::sockaddr_storage>();
        let mut size = size as libc::socklen_t;
        match retry(|| unsafe {
            libc::accept(self.fd(),
                         storagep as *mut libc::sockaddr,
                         &mut size as *mut libc::socklen_t) as libc::c_int
        }) as sock_t {
            -1 => Err(super::last_error()),
            fd => Ok(UnixStream { fd: fd })
        }
    }
}

impl rtio::RtioUnixAcceptor for UnixAcceptor {
    fn accept(&mut self) -> IoResult<~rtio::RtioPipe> {
        self.native_accept().map(|s| ~s as ~rtio::RtioPipe)
    }
}

