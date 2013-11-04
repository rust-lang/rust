// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc;

use super::{Watcher, Loop, NativeHandle, UvError};
use net;
use uvll;

/// A process wraps the handle of the underlying uv_process_t.
pub struct TTY(*uvll::uv_tty_t);

impl Watcher for TTY {}

impl TTY {
    #[fixed_stack_segment] #[inline(never)]
    pub fn new(loop_: &Loop, fd: libc::c_int, readable: bool) ->
            Result<TTY, UvError>
    {
        let mut watcher: TTY = NativeHandle::alloc(uvll::UV_TTY);
        let ret = unsafe {
            uvll::uv_tty_init(loop_.native_handle(), *watcher, fd as libc::c_int,
                              readable as libc::c_int)
        };
        match ret {
            0 => {
                watcher.install_watcher_data();
                Ok(watcher)
            }
            n => {
                unsafe { uvll::free_handle(*watcher); }
                Err(UvError(n))
            }
        }
    }

    pub fn as_stream(&self) -> net::StreamWatcher {
        net::StreamWatcher(**self as *uvll::uv_stream_t)
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn set_mode(&self, raw: bool) -> Result<(), UvError> {
        let raw = raw as libc::c_int;
        match unsafe { uvll::uv_tty_set_mode(self.native_handle(), raw) } {
            0 => Ok(()),
            n => Err(UvError(n))
        }
    }

    #[fixed_stack_segment] #[inline(never)] #[allow(unused_mut)]
    pub fn get_winsize(&self) -> Result<(int, int), UvError> {
        let mut width: libc::c_int = 0;
        let mut height: libc::c_int = 0;
        let widthptr: *libc::c_int = &width;
        let heightptr: *libc::c_int = &width;

        match unsafe { uvll::uv_tty_get_winsize(self.native_handle(),
                                                widthptr, heightptr) } {
            0 => Ok((width as int, height as int)),
            n => Err(UvError(n))
        }
    }
}

impl NativeHandle<uvll::uv_tty_t> for TTY {
    fn from_native_handle(handle: *uvll::uv_tty_t) -> TTY {
        TTY(handle)
    }
    fn native_handle(&self) -> *uvll::uv_tty_t {
        match self { &TTY(ptr) => ptr }
    }
}

