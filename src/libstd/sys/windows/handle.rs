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
use io;
use libc::funcs::extra::kernel32::{GetCurrentProcess, DuplicateHandle};
use libc::{self, HANDLE};
use mem;
use ops::Deref;
use ptr;
use sys::cvt;

/// An owned container for `HANDLE` object, closing them on Drop.
///
/// All methods are inherited through a `Deref` impl to `RawHandle`
#[derive(PartialEq, Eq)]
pub struct Handle(RawHandle);

/// A wrapper type for `HANDLE` objects to give them proper Send/Sync inference
/// as well as Rust-y methods.
///
/// This does **not** drop the handle when it goes out of scope, use `Handle`
/// instead for that.
#[derive(Copy, Clone, Eq)]
pub struct RawHandle(HANDLE);

unsafe impl Send for RawHandle {}
unsafe impl Sync for RawHandle {}

impl Handle {
    pub fn new(handle: HANDLE) -> Handle {
        Handle(RawHandle::new(handle))
    }

    pub fn into_raw(self) -> HANDLE {
        let ret = self.raw();
        mem::forget(self);
        return ret;
    }
}

impl Deref for Handle {
    type Target = RawHandle;
    fn deref(&self) -> &RawHandle { &self.0 }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { let _ = libc::CloseHandle(self.raw()); }
    }
}

impl RawHandle {
    pub fn new(handle: HANDLE) -> RawHandle {
        RawHandle(handle)
    }

    pub fn raw(&self) -> HANDLE { self.0 }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
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

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut amt = 0;
        try!(cvt(unsafe {
            libc::WriteFile(self.0, buf.as_ptr() as libc::LPVOID,
                            buf.len() as libc::DWORD, &mut amt,
                            ptr::null_mut())
        }));
        Ok(amt as usize)
    }

    pub fn duplicate(&self, access: libc::DWORD, inherit: bool,
                     options: libc::DWORD) -> io::Result<Handle> {
        let mut ret = 0 as libc::HANDLE;
        try!(cvt(unsafe {
            let cur_proc = GetCurrentProcess();
            DuplicateHandle(cur_proc, self.0, cur_proc, &mut ret,
                            access, inherit as libc::BOOL,
                            options)
        }));
        Ok(Handle::new(ret))
    }
}

impl PartialEq for RawHandle {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            libc::CompareObjectHandles(self.0, other.0) != libc::FALSE
        }
    }
}
