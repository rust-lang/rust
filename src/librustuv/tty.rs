// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use std::ptr;
use std::rt::rtio::{RtioTTY, IoResult};

use homing::{HomingIO, HomeHandle};
use stream::StreamWatcher;
use super::{UvError, UvHandle, uv_error_to_io_error};
use uvio::UvIoFactory;
use uvll;

pub struct TtyWatcher{
    tty: *uvll::uv_tty_t,
    stream: StreamWatcher,
    home: HomeHandle,
    fd: libc::c_int,
}

impl TtyWatcher {
    pub fn new(io: &mut UvIoFactory, fd: libc::c_int, readable: bool)
        -> Result<TtyWatcher, UvError>
    {
        // libuv may succeed in giving us a handle (via uv_tty_init), but if the
        // handle isn't actually connected to a terminal there are frequently
        // many problems in using it with libuv. To get around this, always
        // return a failure if the specified file descriptor isn't actually a
        // TTY.
        //
        // Related:
        // - https://github.com/joyent/libuv/issues/982
        // - https://github.com/joyent/libuv/issues/988
        let guess = unsafe { uvll::guess_handle(fd) };
        if guess != uvll::UV_TTY as libc::c_int {
            return Err(UvError(uvll::EBADF));
        }

        // libuv was recently changed to not close the stdio file descriptors,
        // but it did not change the behavior for windows. Until this issue is
        // fixed, we need to dup the stdio file descriptors because otherwise
        // uv_close will close them
        let fd = if cfg!(windows) && fd <= libc::STDERR_FILENO {
            unsafe { libc::dup(fd) }
        } else { fd };

        // If this file descriptor is indeed guessed to be a tty, then go ahead
        // with attempting to open it as a tty.
        let handle = UvHandle::alloc(None::<TtyWatcher>, uvll::UV_TTY);
        let mut watcher = TtyWatcher {
            tty: handle,
            stream: StreamWatcher::new(handle),
            home: io.make_handle(),
            fd: fd,
        };
        match unsafe {
            uvll::uv_tty_init(io.uv_loop(), handle, fd as libc::c_int,
                              readable as libc::c_int)
        } {
            0 => Ok(watcher),
            n => {
                // On windows, libuv returns errors before initializing the
                // handle, so our only cleanup is to free the handle itself
                if cfg!(windows) {
                    unsafe { uvll::free_handle(handle); }
                    watcher.tty = ptr::null();
                }
                Err(UvError(n))
            }
        }
    }
}

impl RtioTTY for TtyWatcher {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let _m = self.fire_homing_missile();
        self.stream.read(buf).map_err(uv_error_to_io_error)
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let _m = self.fire_homing_missile();
        self.stream.write(buf, false).map_err(uv_error_to_io_error)
    }

    fn set_raw(&mut self, raw: bool) -> IoResult<()> {
        let raw = raw as libc::c_int;
        let _m = self.fire_homing_missile();
        match unsafe { uvll::uv_tty_set_mode(self.tty, raw) } {
            0 => Ok(()),
            n => Err(uv_error_to_io_error(UvError(n)))
        }
    }

    #[allow(unused_mut)]
    fn get_winsize(&mut self) -> IoResult<(int, int)> {
        let mut width: libc::c_int = 0;
        let mut height: libc::c_int = 0;
        let widthptr: *libc::c_int = &width;
        let heightptr: *libc::c_int = &width;

        let _m = self.fire_homing_missile();
        match unsafe { uvll::uv_tty_get_winsize(self.tty,
                                                widthptr, heightptr) } {
            0 => Ok((width as int, height as int)),
            n => Err(uv_error_to_io_error(UvError(n)))
        }
    }

    fn isatty(&self) -> bool {
        unsafe { uvll::guess_handle(self.fd) == uvll::UV_TTY as libc::c_int }
    }
}

impl UvHandle<uvll::uv_tty_t> for TtyWatcher {
    fn uv_handle(&self) -> *uvll::uv_tty_t { self.tty }
}

impl HomingIO for TtyWatcher {
    fn home<'a>(&'a mut self) -> &'a mut HomeHandle { &mut self.home }
}

impl Drop for TtyWatcher {
    fn drop(&mut self) {
        if !self.tty.is_null() {
            let _m = self.fire_homing_missile();
            self.close_async_();
        }
    }
}
