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
use rt::uv::{Request, NativeHandle, Loop, FsCallback,
             status_to_maybe_uv_error_with_loop};
use rt::uv::uvll;
use rt::uv::uvll::*;
use path::Path;
use cast::transmute;
use libc::{c_int};

pub struct FsRequest(*uvll::uv_fs_t);
impl Request for FsRequest;

#[allow(non_camel_case_types)]
pub enum UvFileFlag {
    O_RDONLY,
    O_WRONLY,
    O_RDWR,
    O_CREAT,
    O_TRUNC
}
pub fn map_flag(v: UvFileFlag) -> int {
    unsafe {
        match v {
            O_RDONLY => uvll::get_O_RDONLY() as int,
            O_WRONLY => uvll::get_O_WRONLY() as int,
            O_RDWR => uvll::get_O_RDWR() as int,
            O_CREAT => uvll::get_O_CREAT() as int,
            O_TRUNC => uvll::get_O_TRUNC() as int
        }
    }
}

pub struct RequestData {
    complete_cb: Option<FsCallback>
}

impl FsRequest {
    pub fn new(cb: Option<FsCallback>) -> FsRequest {
        let fs_req = unsafe { malloc_req(UV_FS) };
        assert!(fs_req.is_not_null());
        let fs_req: FsRequest = NativeHandle::from_native_handle(fs_req);
        fs_req.install_req_data(cb);
        fs_req
    }

    pub fn install_req_data(&self, cb: Option<FsCallback>) {
        let fs_req = (self.native_handle()) as *uvll::uv_write_t;
        let data = ~RequestData {
            complete_cb: cb
        };
        unsafe {
            let data = transmute::<~RequestData, *c_void>(data);
            uvll::set_data_for_req(fs_req, data);
        }
    }

    fn get_req_data<'r>(&'r mut self) -> &'r mut RequestData {
        unsafe {
            let data = uvll::get_data_for_req((self.native_handle()));
            let data = transmute::<&*c_void, &mut ~RequestData>(&data);
            return &mut **data;
        }
    }

    pub fn get_result(&mut self) -> c_int {
        unsafe {
            uvll::get_result_from_fs_req(self.native_handle())
        }
    }

    pub fn get_loop(&self) -> Loop {
        unsafe { Loop{handle:uvll::get_loop_from_fs_req(self.native_handle())} }
    }

    fn cleanup_and_delete(self) {
        unsafe {
            uvll::fs_req_cleanup(self.native_handle());
            let data = uvll::get_data_for_uv_handle(self.native_handle());
            let _data = transmute::<*c_void, ~RequestData>(data);
            uvll::set_data_for_uv_handle(self.native_handle(), null::<()>());
            free_req(self.native_handle() as *c_void)
        }
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

pub struct FileDescriptor(c_int);
impl FileDescriptor {
    fn new(fd: c_int) -> FileDescriptor {
        FileDescriptor(fd)
    }

    pub fn from_open_req(req: &mut FsRequest) -> FileDescriptor {
        FileDescriptor::new(req.get_result())
    }

    pub fn open(loop_: Loop, path: Path, flags: int, mode: int,
               cb: FsCallback) -> int {
        let req = FsRequest::new(Some(cb));
        path.to_str().to_c_str().with_ref(|p| unsafe {
            uvll::fs_open(loop_.native_handle(),
                          req.native_handle(), p, flags, mode, complete_cb) as int
        })

    }

    fn close(self, loop_: Loop, cb: FsCallback) -> int {
        let req = FsRequest::new(Some(cb));
        unsafe {
            uvll::fs_close(loop_.native_handle(), req.native_handle(),
                           self.native_handle(), complete_cb) as int
        }
    }
}
extern fn complete_cb(req: *uv_fs_t) {
    let mut req: FsRequest = NativeHandle::from_native_handle(req);
    let loop_ = req.get_loop();
    // pull the user cb out of the req data
    let cb = {
        let data = req.get_req_data();
        assert!(data.complete_cb.is_some());
        // option dance, option dance. oooooh yeah.
        data.complete_cb.take_unwrap()
    };
    // in uv_fs_open calls, the result will be the fd in the
    // case of success, otherwise it's -1 indicating an error
    let result = req.get_result();
    let status = status_to_maybe_uv_error_with_loop(
        loop_.native_handle(), result);
    // we have a req and status, call the user cb..
    // only giving the user a ref to the FsRequest, as we
    // have to clean it up, afterwards (and they aren't really
    // reusable, anyways
    cb(&mut req, status);
    // clean up the req (and its data!) after calling the user cb
    req.cleanup_and_delete();
}

impl NativeHandle<c_int> for FileDescriptor {
    fn from_native_handle(handle: c_int) -> FileDescriptor {
        FileDescriptor(handle)
    }
    fn native_handle(&self) -> c_int {
        match self { &FileDescriptor(ptr) => ptr }
    }
}

mod test {
    use super::*;
    //use rt::test::*;
    use unstable::run_in_bare_thread;
    use path::Path;
    use rt::uv::Loop;

    // this is equiv to touch, i guess?
    fn file_test_touch_impl() {
        debug!("hello?")
        do run_in_bare_thread {
            debug!("In bare thread")
            let loop_ = Loop::new();
            let flags = map_flag(O_RDWR) |
                map_flag(O_CREAT) | map_flag(O_TRUNC);
            do FileDescriptor::open(loop_, Path("./foo.txt"), flags, 0644)
            |req, uverr| {
                let loop_ = req.get_loop();
                assert!(uverr.is_none());
                let fd = FileDescriptor::from_open_req(req);
                do fd.close(loop_) |_, uverr| {
                    assert!(uverr.is_none());
                };
            };
        }
    }

    #[test]
    fn file_test_touch() {
        file_test_touch_impl();
    }
}
