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
use os::windows::prelude::*;

use ffi::OsStr;
use path::Path;
use io;
use mem;
use rand::{self, Rng};
use slice;
use sys::c;
use sys::fs::{File, OpenOptions};
use sys::handle::Handle;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe {
    inner: Handle,
}

pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    // Note that we specifically do *not* use `CreatePipe` here because
    // unfortunately the anonymous pipes returned do not support overlapped
    // operations.
    //
    // Instead, we create a "hopefully unique" name and create a named pipe
    // which has overlapped operations enabled.
    //
    // Once we do this, we connect do it as usual via `CreateFileW`, and then we
    // return thos reader/writer halves.
    unsafe {
        let key: u64 = rand::thread_rng().gen();
        let name = format!(r"\\.\pipe\__rust_anonymous_pipe1__.{}.{}",
                           c::GetCurrentProcessId(),
                           key);
        let wide_name = OsStr::new(&name)
                              .encode_wide()
                              .chain(Some(0))
                              .collect::<Vec<_>>();

        let reader = c::CreateNamedPipeW(wide_name.as_ptr(),
                                         c::PIPE_ACCESS_INBOUND |
                                             c::FILE_FLAG_FIRST_PIPE_INSTANCE |
                                             c::FILE_FLAG_OVERLAPPED,
                                         c::PIPE_TYPE_BYTE |
                                             c::PIPE_READMODE_BYTE |
                                             c::PIPE_WAIT |
                                             c::PIPE_REJECT_REMOTE_CLIENTS,
                                         1,
                                         4096,
                                         4096,
                                         0,
                                         0 as *mut _);
        if reader == c::INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error())
        }
        let reader = AnonPipe { inner: Handle::new(reader) };

        let mut opts = OpenOptions::new();
        opts.write(true);
        opts.read(false);
        opts.attributes(c::FILE_FLAG_OVERLAPPED);
        let writer = try!(File::open(Path::new(&name), &opts));
        let writer = AnonPipe { inner: writer.into_handle() };

        Ok((reader, writer))
    }
}

impl AnonPipe {
    pub fn handle(&self) -> &Handle { &self.inner }
    pub fn into_handle(self) -> Handle { self.inner }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.inner.read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }
}

pub fn read2(p1: AnonPipe,
             v1: &mut Vec<u8>,
             p2: AnonPipe,
             v2: &mut Vec<u8>) -> io::Result<()> {
    let p1 = p1.into_handle();
    let p2 = p2.into_handle();

    let mut p1 = try!(AsyncPipe::new(&p1, v1));
    let mut p2 = try!(AsyncPipe::new(&p2, v2));
    let objs = [p1.event.raw(), p2.event.raw()];

    try!(p1.schedule_read());
    try!(p2.schedule_read());

    // In a loop we wait for either pipe's scheduled read operation to complete.
    // If the operation completes with 0 bytes, that means EOF was reached, in
    // which case we just finish out the other pipe entirely.
    //
    // Note that overlapped I/O is in general super unsafe because we have to
    // be careful to ensure that all pointers in play are valid for the entire
    // duration of the I/O operation (where tons of operations can also fail).
    // The destructor for `AsyncPipe` ends up taking care of most of this.
    loop {
        let res = unsafe {
            c::WaitForMultipleObjects(2, objs.as_ptr(), c::FALSE, c::INFINITE)
        };
        if res == c::WAIT_OBJECT_0 {
            if try!(p1.result()) == 0 {
                return p2.finish()
            }
            try!(p1.schedule_read());
        } else if res == c::WAIT_OBJECT_0 + 1 {
            if try!(p2.result()) == 0 {
                return p1.finish()
            }
            try!(p2.schedule_read());
        } else {
            return Err(io::Error::last_os_error())
        }
    }
}

struct AsyncPipe<'a> {
    pipe: &'a Handle,
    event: Handle,
    overlapped: Box<c::OVERLAPPED>, // needs a stable address
    dst: &'a mut Vec<u8>,
    reading: bool,
}

impl<'a> AsyncPipe<'a> {
    fn new(pipe: &'a Handle, dst: &'a mut Vec<u8>) -> io::Result<AsyncPipe<'a>> {
        unsafe {
            // Create an event which we'll use to coordinate our overlapped
            // opreations, this event will be used in WaitForMultipleObjects
            // and passed as part of the OVERLAPPED handle.
            let event = c::CreateEventW(0 as *mut _, c::FALSE, c::FALSE,
                                        0 as *const _);
            let event = if event.is_null() {
                return Err(io::Error::last_os_error())
            } else {
                Handle::new(event)
            };
            let mut overlapped: Box<c::OVERLAPPED> = Box::new(mem::zeroed());
            overlapped.hEvent = event.raw();
            Ok(AsyncPipe {
                pipe: pipe,
                overlapped: overlapped,
                event: event,
                dst: dst,
                reading: false,
            })
        }
    }

    /// Executes an overlapped read operation, returning whether the operation
    /// was successfully issued.
    ///
    /// Must not currently be reading, and once the read is done `result` must
    /// be called to figure out how the read went.
    fn schedule_read(&mut self) -> io::Result<()> {
        assert!(!self.reading);
        unsafe {
            let slice = slice_to_end(self.dst);
            try!(self.pipe.read_overlapped(slice, &mut *self.overlapped));
        }
        self.reading = true;
        Ok(())
    }

    /// Wait for the result of the overlapped operation previously executed.
    ///
    /// If this pipe is being read, this will wait for the scheduled overlapped
    /// operation to be completed. Returns how many bytes were read from the
    /// operation.
    fn result(&mut self) -> io::Result<usize> {
        if !self.reading {
            return Ok(0)
        }
        let amt = try!(self.pipe.overlapped_result(&mut *self.overlapped, true));
        self.reading = false;
        unsafe {
            let len = self.dst.len();
            self.dst.set_len(len + amt);
        }
        Ok(amt)
    }

    /// Finishes out reading this pipe entirely.
    ///
    /// Waits for any pending and schedule read, and then calls `read_to_end`
    /// if necessary to read all the remaining information.
    fn finish(&mut self) -> io::Result<()> {
        while try!(self.result()) != 0 {
            try!(self.schedule_read());
        }
        Ok(())
    }
}

impl<'a> Drop for AsyncPipe<'a> {
    fn drop(&mut self) {
        if !self.reading {
            return
        }
        // If we have a pending read operation, then we have to make sure that
        // it's *done* before we actually drop this type. The kernel requires
        // that the `OVERLAPPED` and buffer pointers are valid for the entire
        // I/O operation.
        //
        // To do that, we call `CancelIo` to cancel any pending operation, and
        // if that succeeds we wait for the overlapped result.
        //
        // If anything here fails, there's not really much we can do, so we leak
        // the buffer/OVERLAPPED pointers to ensure we're at least memory safe.
        if self.pipe.cancel_io().is_err() || self.result().is_err() {
            let buf = mem::replace(self.dst, Vec::new());
            let overlapped = Box::new(unsafe { mem::zeroed() });
            let overlapped = mem::replace(&mut self.overlapped, overlapped);
            mem::forget((buf, overlapped));
        }
    }
}

unsafe fn slice_to_end(v: &mut Vec<u8>) -> &mut [u8] {
    if v.capacity() == 0 {
        v.reserve(16);
    }
    if v.capacity() == v.len() {
        v.reserve(1);
    }
    slice::from_raw_parts_mut(v.as_mut_ptr().offset(v.len() as isize),
                              v.capacity() - v.len())
}
