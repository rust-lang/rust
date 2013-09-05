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
use rt::uv::{Request, NativeHandle, Loop, FsCallback, Buf,
             status_to_maybe_uv_error_with_loop, UvError};
use rt::uv::uvll;
use rt::uv::uvll::*;
use super::super::io::support::PathLike;
use cast::transmute;
use libc::{c_int};
use option::{None, Some, Option};

pub struct FsRequest(*uvll::uv_fs_t);
impl Request for FsRequest;

pub struct RequestData {
    complete_cb: Option<FsCallback>,
    raw_fd: Option<c_int>
}

impl FsRequest {
    pub fn new(cb: Option<FsCallback>) -> FsRequest {
        let fs_req = unsafe { malloc_req(UV_FS) };
        assert!(fs_req.is_not_null());
        let fs_req: FsRequest = NativeHandle::from_native_handle(fs_req);
        fs_req.install_req_data(cb);
        fs_req
    }

    fn open_common<P: PathLike>(loop_: &Loop, path: &P, flags: int, mode: int,
               cb: Option<FsCallback>) -> int {
        let complete_cb_ptr = match cb {
            Some(_) => compl_cb as *u8,
            None => 0 as *u8
        };
        let is_sync = cb.is_none();
        let req = FsRequest::new(cb);
        let result = path.path_as_str(|p| {
            p.to_c_str().with_ref(|p| unsafe {
            uvll::fs_open(loop_.native_handle(),
                          req.native_handle(), p, flags, mode, complete_cb_ptr) as int
            })
        });
        if is_sync { req.cleanup_and_delete(); }
        result
    }
    pub fn open<P: PathLike>(loop_: &Loop, path: &P, flags: int, mode: int,
               cb: FsCallback) {
        FsRequest::open_common(loop_, path, flags, mode, Some(cb));
    }

    pub fn open_sync<P: PathLike>(loop_: &Loop, path: &P, flags: int, mode: int)
          -> Result<int, UvError> {
        let result = FsRequest::open_common(loop_, path, flags, mode, None);
        sync_cleanup(loop_, result)
    }

    fn unlink_common<P: PathLike>(loop_: &Loop, path: &P, cb: Option<FsCallback>) -> int {
        let complete_cb_ptr = match cb {
            Some(_) => compl_cb as *u8,
            None => 0 as *u8
        };
        let is_sync = cb.is_none();
        let req = FsRequest::new(cb);
        let result = path.path_as_str(|p| {
            p.to_c_str().with_ref(|p| unsafe {
                uvll::fs_unlink(loop_.native_handle(),
                              req.native_handle(), p, complete_cb_ptr) as int
            })
        });
        if is_sync { req.cleanup_and_delete(); }
        result
    }
    pub fn unlink<P: PathLike>(loop_: &Loop, path: &P, cb: FsCallback) {
        let result = FsRequest::unlink_common(loop_, path, Some(cb));
        sync_cleanup(loop_, result);
    }
    pub fn unlink_sync<P: PathLike>(loop_: &Loop, path: &P) -> Result<int, UvError> {
        let result = FsRequest::unlink_common(loop_, path, None);
        sync_cleanup(loop_, result)
    }

    pub fn install_req_data(&self, cb: Option<FsCallback>) {
        let fs_req = (self.native_handle()) as *uvll::uv_write_t;
        let data = ~RequestData {
            complete_cb: cb,
            raw_fd: None
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
            let data = uvll::get_data_for_req(self.native_handle());
            let _data = transmute::<*c_void, ~RequestData>(data);
            uvll::set_data_for_req(self.native_handle(), null::<()>());
            uvll::fs_req_cleanup(self.native_handle());
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
    fn sync_cleanup(loop_: &Loop, result: int)
          -> Result<int, UvError> {
        match status_to_maybe_uv_error_with_loop(loop_.native_handle(), result as i32) {
            Some(err) => Err(err),
            None => Ok(result)
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

    // as per bnoordhuis in #libuv: offset >= 0 uses prwrite instead of write
    fn write_common(&mut self, loop_: &Loop, buf: Buf, offset: i64, cb: Option<FsCallback>)
          -> int {
        let complete_cb_ptr = match cb {
            Some(_) => compl_cb as *u8,
            None => 0 as *u8
        };
        let is_sync = cb.is_none();
        let mut req = FsRequest::new(cb);
        let base_ptr = buf.base as *c_void;
        let len = buf.len as uint;
        req.get_req_data().raw_fd = Some(self.native_handle());
        let result = unsafe {
            uvll::fs_write(loop_.native_handle(), req.native_handle(),
                           self.native_handle(), base_ptr,
                           len, offset, complete_cb_ptr) as int
        };
        if is_sync { req.cleanup_and_delete(); }
        result
    }
    pub fn write(&mut self, loop_: &Loop, buf: Buf, offset: i64, cb: FsCallback) {
        self.write_common(loop_, buf, offset, Some(cb));
    }
    pub fn write_sync(&mut self, loop_: &Loop, buf: Buf, offset: i64)
          -> Result<int, UvError> {
        let result = self.write_common(loop_, buf, offset, None);
        sync_cleanup(loop_, result)
    }

    fn read_common(&mut self, loop_: &Loop, buf: Buf,
                   offset: i64, cb: Option<FsCallback>)
          -> int {
        let complete_cb_ptr = match cb {
            Some(_) => compl_cb as *u8,
            None => 0 as *u8
        };
        let is_sync = cb.is_none();
        let mut req = FsRequest::new(cb);
        req.get_req_data().raw_fd = Some(self.native_handle());
        let buf_ptr = buf.base as *c_void;
        let result = unsafe {
            uvll::fs_read(loop_.native_handle(), req.native_handle(),
                           self.native_handle(), buf_ptr,
                           buf.len as uint, offset, complete_cb_ptr) as int
        };
        if is_sync { req.cleanup_and_delete(); }
        result
    }
    pub fn read(&mut self, loop_: &Loop, buf: Buf, offset: i64, cb: FsCallback) {
        self.read_common(loop_, buf, offset, Some(cb));
    }
    pub fn read_sync(&mut self, loop_: &Loop, buf: Buf, offset: i64)
          -> Result<int, UvError> {
        let result = self.read_common(loop_, buf, offset, None);
        sync_cleanup(loop_, result)
    }

    fn close_common(self, loop_: &Loop, cb: Option<FsCallback>) -> int {
        let complete_cb_ptr = match cb {
            Some(_) => compl_cb as *u8,
            None => 0 as *u8
        };
        let is_sync = cb.is_none();
        let req = FsRequest::new(cb);
        let result = unsafe {
            uvll::fs_close(loop_.native_handle(), req.native_handle(),
                           self.native_handle(), complete_cb_ptr) as int
        };
        if is_sync { req.cleanup_and_delete(); }
        result
    }
    pub fn close(self, loop_: &Loop, cb: FsCallback) {
        self.close_common(loop_, Some(cb));
    }
    pub fn close_sync(self, loop_: &Loop) -> Result<int, UvError> {
        let result = self.close_common(loop_, None);
        sync_cleanup(loop_, result)
    }
}
extern fn compl_cb(req: *uv_fs_t) {
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
    use libc::{STDOUT_FILENO};
    use vec;
    use str;
    use unstable::run_in_bare_thread;
    use path::Path;
    use rt::uv::{Loop, Buf, slice_to_uv_buf};
    use libc::{O_CREAT, O_RDWR, O_RDONLY,
               S_IWUSR, S_IRUSR}; //NOTE: need defs for S_**GRP|S_**OTH in libc:: ...
               //S_IRGRP, S_IROTH};

    fn file_test_full_simple_impl() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let create_flags = O_RDWR | O_CREAT;
            let read_flags = O_RDONLY;
            // 0644 BZZT! WRONG! 0600! See below.
            let mode = S_IWUSR |S_IRUSR;
                // these aren't defined in std::libc :(
                //map_mode(S_IRGRP) |
                //map_mode(S_IROTH);
            let path_str = "./tmp/file_full_simple.txt";
            let write_val = "hello".as_bytes().to_owned();
            let write_buf  = slice_to_uv_buf(write_val);
            let write_buf_ptr: *Buf = &write_buf;
            let read_buf_len = 1028;
            let read_mem = vec::from_elem(read_buf_len, 0u8);
            let read_buf = slice_to_uv_buf(read_mem);
            let read_buf_ptr: *Buf = &read_buf;
            let p = Path(path_str);
            do FsRequest::open(&loop_, &p, create_flags as int, mode as int)
            |req, uverr| {
                assert!(uverr.is_none());
                let mut fd = FileDescriptor::from_open_req(req);
                let raw_fd = fd.native_handle();
                let buf = unsafe { *write_buf_ptr };
                do fd.write(&req.get_loop(), buf, -1) |req, uverr| {
                    let fd = FileDescriptor(raw_fd);
                    do fd.close(&req.get_loop()) |req, _| {
                        let loop_ = req.get_loop();
                        assert!(uverr.is_none());
                        do FsRequest::open(&loop_, &Path(path_str), read_flags as int,0)
                            |req, uverr| {
                            assert!(uverr.is_none());
                            let loop_ = req.get_loop();
                            let mut fd = FileDescriptor::from_open_req(req);
                            let raw_fd = fd.native_handle();
                            let read_buf = unsafe { *read_buf_ptr };
                            do fd.read(&loop_, read_buf, 0) |req, uverr| {
                                assert!(uverr.is_none());
                                let loop_ = req.get_loop();
                                // we know nread >=0 because uverr is none..
                                let nread = req.get_result() as uint;
                                // nread == 0 would be EOF
                                if nread > 0 {
                                    let read_str = unsafe {
                                        let read_buf = *read_buf_ptr;
                                        str::from_utf8(
                                            vec::from_buf(
                                                read_buf.base, nread))
                                    };
                                    assert!(read_str == ~"hello");
                                    do FileDescriptor(raw_fd).close(&loop_) |req,uverr| {
                                        assert!(uverr.is_none());
                                        let loop_ = &req.get_loop();
                                        do FsRequest::unlink(loop_, &Path(path_str))
                                        |_,uverr| {
                                            assert!(uverr.is_none());
                                        };
                                    };
                                }
                            };
                        };
                    };
                };
            };
            loop_.run();
            loop_.close();
        }
    }
    fn file_test_full_simple_impl_sync() {
        do run_in_bare_thread {
            // setup
            let mut loop_ = Loop::new();
            let create_flags = O_RDWR |
                O_CREAT;
            let read_flags = O_RDONLY;
            // 0644
            let mode = S_IWUSR |
                S_IRUSR;
                //S_IRGRP |
                //S_IROTH;
            let path_str = "./tmp/file_full_simple_sync.txt";
            let write_val = "hello".as_bytes().to_owned();
            let write_buf = slice_to_uv_buf(write_val);
            // open/create
            let result = FsRequest::open_sync(&loop_, &Path(path_str),
                                                   create_flags as int, mode as int);
            assert!(result.is_ok());
            let mut fd = FileDescriptor(result.unwrap() as i32);
            // write
            let result = fd.write_sync(&loop_, write_buf, -1);
            assert!(result.is_ok());
            // close
            let result = fd.close_sync(&loop_);
            assert!(result.is_ok());
            // re-open
            let result = FsRequest::open_sync(&loop_, &Path(path_str),
                                                   read_flags as int,0);
            assert!(result.is_ok());
            let len = 1028;
            let mut fd = FileDescriptor(result.unwrap() as i32);
            // read
            let read_mem: ~[u8] = vec::from_elem(len, 0u8);
            let buf = slice_to_uv_buf(read_mem);
            let result = fd.read_sync(&loop_, buf, 0);
            assert!(result.is_ok());
            let nread = result.unwrap();
            // nread == 0 would be EOF.. we know it's >= zero because otherwise
            // the above assert would fail
            if nread > 0 {
                let read_str = str::from_utf8(
                    read_mem.slice(0, nread as uint));
                assert!(read_str == ~"hello");
                // close
                let result = fd.close_sync(&loop_);
                assert!(result.is_ok());
                // unlink
                let result = FsRequest::unlink_sync(&loop_, &Path(path_str));
                assert!(result.is_ok());
            } else { fail!("nread was 0.. wudn't expectin' that."); }
            loop_.close();
        }
    }

    #[test]
    #[ignore(cfg(windows))] // FIXME #8814
    fn file_test_full_simple() {
        file_test_full_simple_impl();
    }

    #[test]
    #[ignore(cfg(windows))] // FIXME #8814
    fn file_test_full_simple_sync() {
        file_test_full_simple_impl_sync();
    }

    fn naive_print(loop_: &Loop, input: &str) {
        let mut stdout = FileDescriptor(STDOUT_FILENO);
        let write_val = input.as_bytes();
        let write_buf = slice_to_uv_buf(write_val);
        stdout.write_sync(loop_, write_buf, -1);
    }

    #[test]
    fn file_test_write_to_stdout() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            naive_print(&loop_, "zanzibar!\n");
            loop_.run();
            loop_.close();
        };
    }
}
