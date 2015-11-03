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
use mem;
use ops::Deref;
use ptr;
use sys::c;
use sys::cvt;

/// An owned container for `HANDLE` object, closing them on Drop.
///
/// All methods are inherited through a `Deref` impl to `RawHandle`
pub struct Handle(RawHandle);

/// A wrapper type for `HANDLE` objects to give them proper Send/Sync inference
/// as well as Rust-y methods.
///
/// This does **not** drop the handle when it goes out of scope, use `Handle`
/// instead for that.
#[derive(Copy, Clone)]
pub struct RawHandle(c::HANDLE);

unsafe impl Send for RawHandle {}
unsafe impl Sync for RawHandle {}

impl Handle {
    pub fn new(handle: c::HANDLE) -> Handle {
        Handle(RawHandle::new(handle))
    }

    pub fn into_raw(self) -> c::HANDLE {
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
        unsafe { let _ = c::CloseHandle(self.raw()); }
    }
}

impl RawHandle {
    pub fn new(handle: c::HANDLE) -> RawHandle {
        RawHandle(handle)
    }

    pub fn raw(&self) -> c::HANDLE { self.0 }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut read = 0;
        let res = cvt(unsafe {
            c::ReadFile(self.0, buf.as_ptr() as c::LPVOID,
                           buf.len() as c::DWORD, &mut read,
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
            c::WriteFile(self.0, buf.as_ptr() as c::LPVOID,
                            buf.len() as c::DWORD, &mut amt,
                            ptr::null_mut())
        }));
        Ok(amt as usize)
    }

    pub fn duplicate(&self, access: c::DWORD, inherit: bool,
                     options: c::DWORD) -> io::Result<Handle> {
        let mut ret = 0 as c::HANDLE;
        try!(cvt(unsafe {
            let cur_proc = c::GetCurrentProcess();
            c::DuplicateHandle(cur_proc, self.0, cur_proc, &mut ret,
                            access, inherit as c::BOOL,
                            options)
        }));
        Ok(Handle::new(ret))
    }
}
