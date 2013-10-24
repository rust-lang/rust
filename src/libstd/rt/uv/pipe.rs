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
use c_str::CString;

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

    #[fixed_stack_segment] #[inline(never)]
    pub fn open(&mut self, file: libc::c_int) -> Result<(), uv::UvError> {
        match unsafe { uvll::pipe_open(self.native_handle(), file) } {
            0 => Ok(()),
            n => Err(uv::UvError(n))
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn bind(&mut self, name: &CString) -> Result<(), uv::UvError> {
        do name.with_ref |name| {
            match unsafe { uvll::pipe_bind(self.native_handle(), name) } {
                0 => Ok(()),
                n => Err(uv::UvError(n))
            }
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn connect(&mut self, name: &CString, cb: uv::ConnectionCallback) {
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
                    uv::NativeHandle::from_native_handle(req);
            let mut stream_watcher = connect_request.stream();
            connect_request.delete();

            let cb = stream_watcher.get_watcher_data().connect_cb.take_unwrap();
            let status = uv::status_to_maybe_uv_error(status);
            cb(stream_watcher, status);
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
