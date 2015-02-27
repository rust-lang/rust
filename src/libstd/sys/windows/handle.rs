// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use io;
use io::ErrorKind;
use libc::{self, HANDLE};
use mem;
use ptr;
use sys::cvt;

pub struct Handle(HANDLE);

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    pub fn new(handle: HANDLE) -> Handle {
        Handle(handle)
    }

    pub fn close(self) -> io::Result<()> {
        let handle = self.0;
        unsafe { mem::forget(self) };

        // Don't try to close the invalid handle, this is relied upon in the destructor.
        if handle == libc::INVALID_HANLE_VALUE {
            return Ok(());
        }

        try!(cvt(unsafe { libc::CloseHandle(self.0); }));
        Ok(())
    }

    pub fn raw(&self) -> HANDLE { self.0 }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        read(self.0, buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        write(self.0, buf)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let handle = mem::replace(self, Handle::new(libc::INVALID_HANDLE_VALUE));
        handle.close().unwrap();
    }
}


pub fn read(h: HANDLE, buf: &mut [u8]) -> io::Result<usize> {
    let mut read = 0;
    let res = cvt(unsafe {
        libc::ReadFile(h, buf.as_ptr() as libc::LPVOID,
                       buf.len() as libc::DWORD, &mut read,
                       ptr::null_mut())
    });

    match res {
        Ok(_) => Ok(read as usize),

        // The special treatment of BrokenPipe is to deal with Windows
        // pipe semantics, which yields this error when *reading* from
        // a pipe after the other end has closed; we interpret that as
        // EOF on the pipe.
        Err(ref e) if e.kind() == ErrorKind::BrokenPipe => Ok(0),

        Err(e) => Err(e)
    }
}

pub fn write(h: HANDLE, buf: &[u8]) -> io::Result<usize> {
    let mut amt = 0;
    try!(cvt(unsafe {
        libc::WriteFile(h, buf.as_ptr() as libc::LPVOID,
                        buf.len() as libc::DWORD, &mut amt,
                        ptr::null_mut())
    }));
    Ok(amt as usize)
}
