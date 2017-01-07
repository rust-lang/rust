// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use os::windows::prelude::*;

use ffi::OsStr;
use io;
use mem;
use path::Path;
use ptr;
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

pub struct Pipes {
    pub ours: AnonPipe,
    pub theirs: AnonPipe,
}

/// Although this looks similar to `anon_pipe` in the Unix module it's actually
/// subtly different. Here we'll return two pipes in the `Pipes` return value,
/// but one is intended for "us" where as the other is intended for "someone
/// else".
///
/// Currently the only use case for this function is pipes for stdio on
/// processes in the standard library, so "ours" is the one that'll stay in our
/// process whereas "theirs" will be inherited to a child.
///
/// The ours/theirs pipes are *not* specifically readable or writable. Each
/// one only supports a read or a write, but which is which depends on the
/// boolean flag given. If `ours_readable` is true then `ours` is readable where
/// `theirs` is writable. Conversely if `ours_readable` is false then `ours` is
/// writable where `theirs` is readable.
///
/// Also note that the `ours` pipe is always a handle opened up in overlapped
/// mode. This means that technically speaking it should only ever be used
/// with `OVERLAPPED` instances, but also works out ok if it's only ever used
/// once at a time (which we do indeed guarantee).
pub fn anon_pipe(ours_readable: bool) -> io::Result<Pipes> {
    // Note that we specifically do *not* use `CreatePipe` here because
    // unfortunately the anonymous pipes returned do not support overlapped
    // operations. Instead, we create a "hopefully unique" name and create a
    // named pipe which has overlapped operations enabled.
    //
    // Once we do this, we connect do it as usual via `CreateFileW`, and then
    // we return those reader/writer halves. Note that the `ours` pipe return
    // value is always the named pipe, whereas `theirs` is just the normal file.
    // This should hopefully shield us from child processes which assume their
    // stdout is a named pipe, which would indeed be odd!
    unsafe {
        let ours;
        let mut name;
        let mut tries = 0;
        let mut reject_remote_clients_flag = c::PIPE_REJECT_REMOTE_CLIENTS;
        loop {
            tries += 1;
            let key: u64 = rand::thread_rng().gen();
            name = format!(r"\\.\pipe\__rust_anonymous_pipe1__.{}.{}",
                           c::GetCurrentProcessId(),
                           key);
            let wide_name = OsStr::new(&name)
                                  .encode_wide()
                                  .chain(Some(0))
                                  .collect::<Vec<_>>();
            let mut flags = c::FILE_FLAG_FIRST_PIPE_INSTANCE |
                c::FILE_FLAG_OVERLAPPED;
            if ours_readable {
                flags |= c::PIPE_ACCESS_INBOUND;
            } else {
                flags |= c::PIPE_ACCESS_OUTBOUND;
            }

            let handle = c::CreateNamedPipeW(wide_name.as_ptr(),
                                             flags,
                                             c::PIPE_TYPE_BYTE |
                                             c::PIPE_READMODE_BYTE |
                                             c::PIPE_WAIT |
                                             reject_remote_clients_flag,
                                             1,
                                             4096,
                                             4096,
                                             0,
                                             ptr::null_mut());

            // We pass the FILE_FLAG_FIRST_PIPE_INSTANCE flag above, and we're
            // also just doing a best effort at selecting a unique name. If
            // ERROR_ACCESS_DENIED is returned then it could mean that we
            // accidentally conflicted with an already existing pipe, so we try
            // again.
            //
            // Don't try again too much though as this could also perhaps be a
            // legit error.
            // If ERROR_INVALID_PARAMETER is returned, this probably means we're
            // running on pre-Vista version where PIPE_REJECT_REMOTE_CLIENTS is
            // not supported, so we continue retrying without it. This implies
            // reduced security on Windows versions older than Vista by allowing
            // connections to this pipe from remote machines.
            // Proper fix would increase the number of FFI imports and introduce
            // significant amount of Windows XP specific code with no clean
            // testing strategy
            // for more info see https://github.com/rust-lang/rust/pull/37677
            if handle == c::INVALID_HANDLE_VALUE {
                let err = io::Error::last_os_error();
                let raw_os_err = err.raw_os_error();
                if tries < 10 {
                    if raw_os_err == Some(c::ERROR_ACCESS_DENIED as i32) {
                        continue
                    } else if reject_remote_clients_flag != 0 &&
                        raw_os_err == Some(c::ERROR_INVALID_PARAMETER as i32) {
                        reject_remote_clients_flag = 0;
                        tries -= 1;
                        continue
                    }
                }
                return Err(err)
            }
            ours = Handle::new(handle);
            break
        }

        // Connect to the named pipe we just created. This handle is going to be
        // returned in `theirs`, so if `ours` is readable we want this to be
        // writable, otherwise if `ours` is writable we want this to be
        // readable.
        //
        // Additionally we don't enable overlapped mode on this because most
        // client processes aren't enabled to work with that.
        let mut opts = OpenOptions::new();
        opts.write(ours_readable);
        opts.read(!ours_readable);
        opts.share_mode(0);
        let theirs = File::open(Path::new(&name), &opts)?;
        let theirs = AnonPipe { inner: theirs.into_handle() };

        Ok(Pipes {
            ours: AnonPipe { inner: ours },
            theirs: AnonPipe { inner: theirs.into_handle() },
        })
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

    let mut p1 = AsyncPipe::new(p1, v1)?;
    let mut p2 = AsyncPipe::new(p2, v2)?;
    let objs = [p1.event.raw(), p2.event.raw()];

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
            if !p1.result()? || !p1.schedule_read()? {
                return p2.finish()
            }
        } else if res == c::WAIT_OBJECT_0 + 1 {
            if !p2.result()? || !p2.schedule_read()? {
                return p1.finish()
            }
        } else {
            return Err(io::Error::last_os_error())
        }
    }
}

struct AsyncPipe<'a> {
    pipe: Handle,
    event: Handle,
    overlapped: Box<c::OVERLAPPED>, // needs a stable address
    dst: &'a mut Vec<u8>,
    state: State,
}

#[derive(PartialEq, Debug)]
enum State {
    NotReading,
    Reading,
    Read(usize),
}

impl<'a> AsyncPipe<'a> {
    fn new(pipe: Handle, dst: &'a mut Vec<u8>) -> io::Result<AsyncPipe<'a>> {
        // Create an event which we'll use to coordinate our overlapped
        // opreations, this event will be used in WaitForMultipleObjects
        // and passed as part of the OVERLAPPED handle.
        //
        // Note that we do a somewhat clever thing here by flagging the
        // event as being manually reset and setting it initially to the
        // signaled state. This means that we'll naturally fall through the
        // WaitForMultipleObjects call above for pipes created initially,
        // and the only time an even will go back to "unset" will be once an
        // I/O operation is successfully scheduled (what we want).
        let event = Handle::new_event(true, true)?;
        let mut overlapped: Box<c::OVERLAPPED> = unsafe {
            Box::new(mem::zeroed())
        };
        overlapped.hEvent = event.raw();
        Ok(AsyncPipe {
            pipe: pipe,
            overlapped: overlapped,
            event: event,
            dst: dst,
            state: State::NotReading,
        })
    }

    /// Executes an overlapped read operation.
    ///
    /// Must not currently be reading, and returns whether the pipe is currently
    /// at EOF or not. If the pipe is not at EOF then `result()` must be called
    /// to complete the read later on (may block), but if the pipe is at EOF
    /// then `result()` should not be called as it will just block forever.
    fn schedule_read(&mut self) -> io::Result<bool> {
        assert_eq!(self.state, State::NotReading);
        let amt = unsafe {
            let slice = slice_to_end(self.dst);
            self.pipe.read_overlapped(slice, &mut *self.overlapped)?
        };

        // If this read finished immediately then our overlapped event will
        // remain signaled (it was signaled coming in here) and we'll progress
        // down to the method below.
        //
        // Otherwise the I/O operation is scheduled and the system set our event
        // to not signaled, so we flag ourselves into the reading state and move
        // on.
        self.state = match amt {
            Some(0) => return Ok(false),
            Some(amt) => State::Read(amt),
            None => State::Reading,
        };
        Ok(true)
    }

    /// Wait for the result of the overlapped operation previously executed.
    ///
    /// Takes a parameter `wait` which indicates if this pipe is currently being
    /// read whether the function should block waiting for the read to complete.
    ///
    /// Return values:
    ///
    /// * `true` - finished any pending read and the pipe is not at EOF (keep
    ///            going)
    /// * `false` - finished any pending read and pipe is at EOF (stop issuing
    ///             reads)
    fn result(&mut self) -> io::Result<bool> {
        let amt = match self.state {
            State::NotReading => return Ok(true),
            State::Reading => {
                self.pipe.overlapped_result(&mut *self.overlapped, true)?
            }
            State::Read(amt) => amt,
        };
        self.state = State::NotReading;
        unsafe {
            let len = self.dst.len();
            self.dst.set_len(len + amt);
        }
        Ok(amt != 0)
    }

    /// Finishes out reading this pipe entirely.
    ///
    /// Waits for any pending and schedule read, and then calls `read_to_end`
    /// if necessary to read all the remaining information.
    fn finish(&mut self) -> io::Result<()> {
        while self.result()? && self.schedule_read()? {
            // ...
        }
        Ok(())
    }
}

impl<'a> Drop for AsyncPipe<'a> {
    fn drop(&mut self) {
        match self.state {
            State::Reading => {}
            _ => return,
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
