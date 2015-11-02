// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::ErrorKind;
use libc::funcs::extra::kernel32::{GetCurrentProcess, DuplicateHandle};
use libc::{self, HANDLE};
use ptr;
use sys::windows::c::cvt;
use sys::error::Result;
use sys::inner::*;

/// An owned container for `HANDLE` object, closing them on Drop.
///
/// All methods are inherited through a `Deref` impl to `RawHandle`
pub struct Handle(HANDLE);

impl_inner!(Handle(HANDLE): AsInner + IntoInnerForget + FromInner);

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        let mut read = 0;
        let res = cvt(unsafe {
            libc::ReadFile(self.0, buf.as_ptr() as libc::LPVOID,
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

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        let mut amt = 0;
        try!(cvt(unsafe {
            libc::WriteFile(self.0, buf.as_ptr() as libc::LPVOID,
                            buf.len() as libc::DWORD, &mut amt,
                            ptr::null_mut())
        }));
        Ok(amt as usize)
    }

    pub fn duplicate(&self, access: libc::DWORD, inherit: bool,
                     options: libc::DWORD) -> Result<Handle> {
        let mut ret = 0 as libc::HANDLE;
        try!(cvt(unsafe {
            let cur_proc = GetCurrentProcess();
            DuplicateHandle(cur_proc, self.0, cur_proc, &mut ret,
                            access, inherit as libc::BOOL,
                            options)
        }));
        Ok(Handle::from_inner(ret))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { let _ = libc::CloseHandle(*self.as_inner()); }
    }
}
