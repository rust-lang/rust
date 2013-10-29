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
use std::c_str::CString;

use super::{Loop, UvError, Watcher, NativeHandle, status_to_maybe_uv_error};
use super::ConnectionCallback;
use net;
use uvll;

pub struct Pipe(*uvll::uv_pipe_t);

impl Watcher for Pipe {}

impl Pipe {
    pub fn new(loop_: &Loop, ipc: bool) -> Pipe {
        unsafe {
            let handle = uvll::malloc_handle(uvll::UV_NAMED_PIPE);
            assert!(handle.is_not_null());
            let ipc = ipc as libc::c_int;
            assert_eq!(uvll::pipe_init(loop_.native_handle(), handle, ipc), 0);
            let mut ret: Pipe =
                    NativeHandle::from_native_handle(handle);
            ret.install_watcher_data();
            ret
        }
    }

    pub fn as_stream(&self) -> net::StreamWatcher {
        net::StreamWatcher(**self as *uvll::uv_stream_t)
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn open(&mut self, file: libc::c_int) -> Result<(), UvError> {
        match unsafe { uvll::pipe_open(self.native_handle(), file) } {
            0 => Ok(()),
            n => Err(UvError(n))
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn bind(&mut self, name: &CString) -> Result<(), UvError> {
        do name.with_ref |name| {
            match unsafe { uvll::pipe_bind(self.native_handle(), name) } {
                0 => Ok(()),
                n => Err(UvError(n))
            }
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn connect(&mut self, name: &CString, cb: ConnectionCallback) {
        {
            let data = self.get_watcher_data();
            assert!(data.connect_cb.is_none());
            data.connect_cb = Some(cb);
        }

        let connect = net::ConnectRequest::new();
        let name = do name.with_ref |p| { p };

        unsafe {
            uvll::pipe_connect(connect.native_handle(),
                               self.native_handle(),
                               name,
                               connect_cb)
        }

        extern "C" fn connect_cb(req: *uvll::uv_connect_t, status: libc::c_int) {
            let connect_request: net::ConnectRequest =
                    NativeHandle::from_native_handle(req);
            let mut stream_watcher = connect_request.stream();
            connect_request.delete();

            let cb = stream_watcher.get_watcher_data().connect_cb.take_unwrap();
            let status = status_to_maybe_uv_error(status);
            cb(stream_watcher, status);
        }
    }

}

impl NativeHandle<*uvll::uv_pipe_t> for Pipe {
    fn from_native_handle(handle: *uvll::uv_pipe_t) -> Pipe {
        Pipe(handle)
    }
    fn native_handle(&self) -> *uvll::uv_pipe_t {
        match self { &Pipe(ptr) => ptr }
    }
}
