// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(reason = "not public", issue = "0", feature = "fd")]

use io::{self, Read, Initializer};
use mem;
use sys::wasm_cloudabi::cloudabi;
use sys::wasm_cloudabi::err::cvt;
use sys_common::AsInner;

pub type RawFd = u32;

#[derive(Debug)]
pub struct FileDesc {
    fd: RawFd,
}

impl FileDesc {
    pub fn new(fd: RawFd) -> FileDesc {
        FileDesc { fd }
    }

    pub fn raw(&self) -> RawFd { self.fd }

    pub fn into_raw(self) -> RawFd {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    fn cloudabi_raw(&self) -> cloudabi::fd {
        cloudabi::fd(self.fd)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut read = unsafe { mem::uninitialized() };
        cvt(unsafe {
            let data = vec![cloudabi::iovec { buf: (buf.as_mut_ptr() as *mut (), buf.len()) }];
            cloudabi::fd_read(self.cloudabi_raw(), &data, &mut read)
        })?;
        Ok(read)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut read = unsafe { mem::uninitialized() };
        cvt(unsafe {
            let data = vec![cloudabi::iovec { buf: (buf.as_mut_ptr() as *mut (), buf.len()) }];
            cloudabi::fd_pread(self.cloudabi_raw(), &data, offset, &mut read)
        })?;
        Ok(read)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut written = unsafe { mem::uninitialized() };
        cvt(unsafe {
            let data = vec![cloudabi::ciovec { buf: (buf.as_ptr() as *const (), buf.len()) }];
            cloudabi::fd_write(self.cloudabi_raw(), &data, &mut written)
        })?;
        Ok(written)
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        let mut written = unsafe { mem::uninitialized() };
        cvt(unsafe {
            let data = vec![cloudabi::ciovec { buf: (buf.as_ptr() as *const (), buf.len()) }];
            cloudabi::fd_pwrite(self.cloudabi_raw(), &data, offset, &mut written)
        })?;
        Ok(written)
    }


    pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "ni"))
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        let mut fd = unsafe { mem::uninitialized() };
        cvt(unsafe {
            cloudabi::fd_dup(self.cloudabi_raw(), &mut fd)
        })?;
        Ok(FileDesc::new(fd.0 as RawFd))
    }
}

impl<'a> Read for &'a FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

impl AsInner<RawFd> for FileDesc {
    fn as_inner(&self) -> &RawFd { &self.fd }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        unsafe {
            cloudabi::fd_close(self.cloudabi_raw());
        }
    }
}
