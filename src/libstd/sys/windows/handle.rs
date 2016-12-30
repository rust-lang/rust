// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(issue = "0", feature = "windows_handle")]

use cmp;
use io::{ErrorKind, Read};
use io;
use mem;
use ops::Deref;
use ptr;
use sys::c;
use sys::cvt;
use sys_common::io::read_to_end_uninitialized;

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

    pub fn new_event(manual: bool, init: bool) -> io::Result<Handle> {
        unsafe {
            let event = c::CreateEventW(ptr::null_mut(),
                                        manual as c::BOOL,
                                        init as c::BOOL,
                                        ptr::null());
            if event.is_null() {
                Err(io::Error::last_os_error())
            } else {
                Ok(Handle::new(event))
            }
        }
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
        let len = cmp::min(buf.len(), <c::DWORD>::max_value() as usize) as c::DWORD;
        let res = cvt(unsafe {
            c::ReadFile(self.0, buf.as_mut_ptr() as c::LPVOID,
                        len, &mut read, ptr::null_mut())
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

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut read = 0;
        let len = cmp::min(buf.len(), <c::DWORD>::max_value() as usize) as c::DWORD;
        let res = unsafe {
            let mut overlapped: c::OVERLAPPED = mem::zeroed();
            overlapped.Offset = offset as u32;
            overlapped.OffsetHigh = (offset >> 32) as u32;
            cvt(c::ReadFile(self.0, buf.as_mut_ptr() as c::LPVOID,
                            len, &mut read, &mut overlapped))
        };
        match res {
            Ok(_) => Ok(read as usize),
            Err(ref e) if e.raw_os_error() == Some(c::ERROR_HANDLE_EOF as i32) => Ok(0),
            Err(e) => Err(e),
        }
    }

    pub unsafe fn read_overlapped(&self,
                                  buf: &mut [u8],
                                  overlapped: *mut c::OVERLAPPED)
                                  -> io::Result<Option<usize>> {
        let len = cmp::min(buf.len(), <c::DWORD>::max_value() as usize) as c::DWORD;
        let mut amt = 0;
        let res = cvt({
            c::ReadFile(self.0, buf.as_ptr() as c::LPVOID,
                        len, &mut amt, overlapped)
        });
        match res {
            Ok(_) => Ok(Some(amt as usize)),
            Err(e) => {
                if e.raw_os_error() == Some(c::ERROR_IO_PENDING as i32) {
                    Ok(None)
                } else if e.raw_os_error() == Some(c::ERROR_BROKEN_PIPE as i32) {
                    Ok(Some(0))
                } else {
                    Err(e)
                }
            }
        }
    }

    pub fn overlapped_result(&self,
                             overlapped: *mut c::OVERLAPPED,
                             wait: bool) -> io::Result<usize> {
        unsafe {
            let mut bytes = 0;
            let wait = if wait {c::TRUE} else {c::FALSE};
            let res = cvt({
                c::GetOverlappedResult(self.raw(), overlapped, &mut bytes, wait)
            });
            match res {
                Ok(_) => Ok(bytes as usize),
                Err(e) => {
                    if e.raw_os_error() == Some(c::ERROR_HANDLE_EOF as i32) ||
                       e.raw_os_error() == Some(c::ERROR_BROKEN_PIPE as i32) {
                        Ok(0)
                    } else {
                        Err(e)
                    }
                }
            }
        }
    }

    pub fn cancel_io(&self) -> io::Result<()> {
        unsafe {
            cvt(c::CancelIo(self.raw())).map(|_| ())
        }
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut amt = 0;
        let len = cmp::min(buf.len(), <c::DWORD>::max_value() as usize) as c::DWORD;
        cvt(unsafe {
            c::WriteFile(self.0, buf.as_ptr() as c::LPVOID,
                         len, &mut amt, ptr::null_mut())
        })?;
        Ok(amt as usize)
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        let mut written = 0;
        let len = cmp::min(buf.len(), <c::DWORD>::max_value() as usize) as c::DWORD;
        unsafe {
            let mut overlapped: c::OVERLAPPED = mem::zeroed();
            overlapped.Offset = offset as u32;
            overlapped.OffsetHigh = (offset >> 32) as u32;
            cvt(c::WriteFile(self.0, buf.as_ptr() as c::LPVOID,
                             len, &mut written, &mut overlapped))?;
        }
        Ok(written as usize)
    }

    pub fn duplicate(&self, access: c::DWORD, inherit: bool,
                     options: c::DWORD) -> io::Result<Handle> {
        let mut ret = 0 as c::HANDLE;
        cvt(unsafe {
            let cur_proc = c::GetCurrentProcess();
            c::DuplicateHandle(cur_proc, self.0, cur_proc, &mut ret,
                            access, inherit as c::BOOL,
                            options)
        })?;
        Ok(Handle::new(ret))
    }
}

impl<'a> Read for &'a RawHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        unsafe { read_to_end_uninitialized(self, buf) }
    }
}
