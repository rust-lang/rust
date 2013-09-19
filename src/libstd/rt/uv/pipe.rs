// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use libc;

use rt::uv;
use rt::uv::net;
use rt::uv::uvll;

pub struct Pipe(*uvll::uv_pipe_t);

impl uv::Watcher for Pipe {}

impl Pipe {
    pub fn new(loop_: &uv::Loop, ipc: bool) -> Pipe {
        unsafe {
            let handle = uvll::malloc_handle(uvll::UV_NAMED_PIPE);
            assert!(handle.is_not_null());
            let ipc = ipc as libc::c_int;
            assert_eq!(uvll::pipe_init(loop_.native_handle(), handle, ipc), 0);
            let mut ret: Pipe =
                    uv::NativeHandle::from_native_handle(handle);
            ret.install_watcher_data();
            ret
        }
    }

    pub fn as_stream(&self) -> net::StreamWatcher {
        net::StreamWatcher(**self as *uvll::uv_stream_t)
    }

    pub fn close(self, cb: uv::NullCallback) {
        {
            let mut this = self;
            let data = this.get_watcher_data();
            assert!(data.close_cb.is_none());
            data.close_cb = Some(cb);
        }

        unsafe { uvll::close(self.native_handle(), close_cb); }

        extern fn close_cb(handle: *uvll::uv_pipe_t) {
            let mut process: Pipe = uv::NativeHandle::from_native_handle(handle);
            process.get_watcher_data().close_cb.take_unwrap()();
            process.drop_watcher_data();
            unsafe { uvll::free_handle(handle as *libc::c_void) }
        }
    }
}

impl uv::NativeHandle<*uvll::uv_pipe_t> for Pipe {
    fn from_native_handle(handle: *uvll::uv_pipe_t) -> Pipe {
        Pipe(handle)
    }
    fn native_handle(&self) -> *uvll::uv_pipe_t {
        match self { &Pipe(ptr) => ptr }
    }
}
