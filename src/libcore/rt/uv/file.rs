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
use ptr::null;
use libc::c_void;
use super::{UvError, Callback, Request, NativeHandle, Loop};
use super::super::uvll;
use super::super::uvll::*;

pub type FsCallback = ~fn(FsRequest, Option<UvError>);
impl Callback for FsCallback { }

pub struct FsRequest(*uvll::uv_fs_t);

impl Request for FsRequest;

impl FsRequest {
    fn new() -> FsRequest {
        let fs_req = unsafe { malloc_req(UV_FS) };
        assert!(fs_req.is_not_null());
        let fs_req = fs_req as *uvll::uv_write_t;
        unsafe { uvll::set_data_for_req(fs_req, null::<()>()); }
        NativeHandle::from_native_handle(fs_req)
    }

    fn delete(self) {
        unsafe { free_req(self.native_handle() as *c_void) }
    }

    fn open(&mut self, _loop_: &Loop, _cb: FsCallback) {
    }

    fn close(&mut self, _loop_: &Loop, _cb: FsCallback) {
    }
}

impl NativeHandle<*uvll::uv_fs_t> for FsRequest {
    fn from_native_handle(handle: *uvll:: uv_fs_t) -> FsRequest {
        FsRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_fs_t {
        match self { &FsRequest(ptr) => ptr }
    }
}
