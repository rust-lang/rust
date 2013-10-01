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
             status_to_maybe_uv_error, UvError};
use rt::uv::uvll;
use rt::uv::uvll::*;
use super::super::io::support::PathLike;
use cast::transmute;
use libc;
use libc::{c_int};
use option::{None, Some, Option};

pub struct FsRequest(*uvll::uv_fs_t);
impl Request for FsRequest {}

pub struct RequestData {
    complete_cb: Option<FsCallback>
}

impl FsRequest {
    pub fn new() -> FsRequest {
        let fs_req = unsafe { malloc_req(UV_FS) };
        assert!(fs_req.is_not_null());
        let fs_req: FsRequest = NativeHandle::from_native_handle(fs_req);
        fs_req
    }

    pub fn open<P: PathLike>(self, loop_: &Loop, path: &P, flags: int, mode: int,
               cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
            uvll::fs_open(loop_.native_handle(),
                          self.native_handle(), p, flags, mode, complete_cb_ptr)
            })
        });
    }

    pub fn open_sync<P: PathLike>(self, loop_: &Loop, path: &P,
                                  flags: int, mode: int) -> Result<c_int, UvError> {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(None)
        };
        let result = path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
            uvll::fs_open(loop_.native_handle(),
                    self.native_handle(), p, flags, mode, complete_cb_ptr)
            })
        });
        self.sync_cleanup(result)
    }

    pub fn unlink<P: PathLike>(self, loop_: &Loop, path: &P, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
                uvll::fs_unlink(loop_.native_handle(),
                              self.native_handle(), p, complete_cb_ptr)
            })
        });
    }

    pub fn unlink_sync<P: PathLike>(self, loop_: &Loop, path: &P)
      -> Result<c_int, UvError> {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(None)
        };
        let result = path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
                uvll::fs_unlink(loop_.native_handle(),
                              self.native_handle(), p, complete_cb_ptr)
            })
        });
        self.sync_cleanup(result)
    }

    pub fn stat<P: PathLike>(self, loop_: &Loop, path: &P, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
                uvll::fs_stat(loop_.native_handle(),
                              self.native_handle(), p, complete_cb_ptr)
            })
        });
    }

    pub fn write(self, loop_: &Loop, fd: c_int, buf: Buf, offset: i64, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        let base_ptr = buf.base as *c_void;
        let len = buf.len as uint;
        unsafe {
            uvll::fs_write(loop_.native_handle(), self.native_handle(),
                           fd, base_ptr,
                           len, offset, complete_cb_ptr)
        };
    }
    pub fn write_sync(self, loop_: &Loop, fd: c_int, buf: Buf, offset: i64)
          -> Result<c_int, UvError> {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(None)
        };
        let base_ptr = buf.base as *c_void;
        let len = buf.len as uint;
        let result = unsafe {
            uvll::fs_write(loop_.native_handle(), self.native_handle(),
                           fd, base_ptr,
                           len, offset, complete_cb_ptr)
        };
        self.sync_cleanup(result)
    }

    pub fn read(self, loop_: &Loop, fd: c_int, buf: Buf, offset: i64, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        let buf_ptr = buf.base as *c_void;
        let len = buf.len as uint;
        unsafe {
            uvll::fs_read(loop_.native_handle(), self.native_handle(),
                           fd, buf_ptr,
                           len, offset, complete_cb_ptr)
        };
    }
    pub fn read_sync(self, loop_: &Loop, fd: c_int, buf: Buf, offset: i64)
          -> Result<c_int, UvError> {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(None)
        };
        let buf_ptr = buf.base as *c_void;
        let len = buf.len as uint;
        let result = unsafe {
            uvll::fs_read(loop_.native_handle(), self.native_handle(),
                           fd, buf_ptr,
                           len, offset, complete_cb_ptr)
        };
        self.sync_cleanup(result)
    }

    pub fn close(self, loop_: &Loop, fd: c_int, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        unsafe {
            uvll::fs_close(loop_.native_handle(), self.native_handle(),
                           fd, complete_cb_ptr)
        };
    }
    pub fn close_sync(self, loop_: &Loop, fd: c_int) -> Result<c_int, UvError> {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(None)
        };
        let result = unsafe {
            uvll::fs_close(loop_.native_handle(), self.native_handle(),
                           fd, complete_cb_ptr)
        };
        self.sync_cleanup(result)
    }

    pub fn mkdir<P: PathLike>(self, loop_: &Loop, path: &P, mode: int, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
            uvll::fs_mkdir(loop_.native_handle(),
                          self.native_handle(), p, mode, complete_cb_ptr)
            })
        });
    }

    pub fn rmdir<P: PathLike>(self, loop_: &Loop, path: &P, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
            uvll::fs_rmdir(loop_.native_handle(),
                          self.native_handle(), p, complete_cb_ptr)
            })
        });
    }

    pub fn readdir<P: PathLike>(self, loop_: &Loop, path: &P,
                                flags: c_int, cb: FsCallback) {
        let complete_cb_ptr = {
            let mut me = self;
            me.req_boilerplate(Some(cb))
        };
        path.path_as_str(|p| {
            p.with_c_str(|p| unsafe {
            uvll::fs_readdir(loop_.native_handle(),
                          self.native_handle(), p, flags, complete_cb_ptr)
            })
        });
    }

    // accessors/utility funcs
    fn sync_cleanup(self, result: c_int)
          -> Result<c_int, UvError> {
        self.cleanup_and_delete();
        match status_to_maybe_uv_error(result as i32) {
            Some(err) => Err(err),
            None => Ok(result)
        }
    }
    fn req_boilerplate(&mut self, cb: Option<FsCallback>) -> *u8 {
        let result = match cb {
            Some(_) => {
                compl_cb as *u8
            },
            None => 0 as *u8
        };
        self.install_req_data(cb);
        result
    }
    pub fn install_req_data(&mut self, cb: Option<FsCallback>) {
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
            &mut **data
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

    pub fn get_stat(&self) -> uv_stat_t {
        let stat = uv_stat_t::new();
        unsafe { uvll::populate_stat(self.native_handle(), &stat); }
        stat
    }

    pub fn get_ptr(&self) -> *libc::c_void {
        unsafe {
            uvll::get_ptr_from_fs_req(self.native_handle())
        }
    }

    pub fn get_paths(&mut self) -> ~[~str] {
        use str;
        let ptr = self.get_ptr();
        match self.get_result() {
            n if (n <= 0) => {
                ~[]
            },
            n => {
                let n_len = n as uint;
                // we pass in the len that uv tells us is there
                // for the entries and we don't continue past that..
                // it appears that sometimes the multistring isn't
                // correctly delimited and we stray into garbage memory?
                // in any case, passing Some(n_len) fixes it and ensures
                // good results
                let raw_path_strs = unsafe {
                    str::raw::from_c_multistring(ptr as *libc::c_char, Some(n_len)) };
                let raw_len = raw_path_strs.len();
                assert_eq!(raw_len, n_len);
                raw_path_strs
            }
        }
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

fn sync_cleanup(result: int)
    -> Result<int, UvError> {
    match status_to_maybe_uv_error(result as i32) {
        Some(err) => Err(err),
        None => Ok(result)
    }
}

extern fn compl_cb(req: *uv_fs_t) {
    let mut req: FsRequest = NativeHandle::from_native_handle(req);
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
    let status = status_to_maybe_uv_error(result);
    // we have a req and status, call the user cb..
    // only giving the user a ref to the FsRequest, as we
    // have to clean it up, afterwards (and they aren't really
    // reusable, anyways
    cb(&mut req, status);
    // clean up the req (and its data!) after calling the user cb
    req.cleanup_and_delete();
}

#[cfg(test)]
mod test {
    use super::*;
    //use rt::test::*;
    use libc::{STDOUT_FILENO};
    use vec;
    use str;
    use unstable::run_in_bare_thread;
    use path::Path;
    use rt::uv::{Loop, Buf, slice_to_uv_buf};
    use libc::{O_CREAT, O_RDWR, O_RDONLY, S_IWUSR, S_IRUSR};

    #[test]
    fn file_test_full_simple() {
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
            let open_req = FsRequest::new();
            do open_req.open(&loop_, &p, create_flags as int, mode as int)
            |req, uverr| {
                assert!(uverr.is_none());
                let fd = req.get_result();
                let buf = unsafe { *write_buf_ptr };
                let write_req = FsRequest::new();
                do write_req.write(&req.get_loop(), fd, buf, -1) |req, uverr| {
                    let close_req = FsRequest::new();
                    do close_req.close(&req.get_loop(), fd) |req, _| {
                        assert!(uverr.is_none());
                        let loop_ = req.get_loop();
                        let open_req = FsRequest::new();
                        do open_req.open(&loop_, &Path(path_str), read_flags as int,0)
                            |req, uverr| {
                            assert!(uverr.is_none());
                            let loop_ = req.get_loop();
                            let fd = req.get_result();
                            let read_buf = unsafe { *read_buf_ptr };
                            let read_req = FsRequest::new();
                            do read_req.read(&loop_, fd, read_buf, 0) |req, uverr| {
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
                                    let close_req = FsRequest::new();
                                    do close_req.close(&loop_, fd) |req,uverr| {
                                        assert!(uverr.is_none());
                                        let loop_ = &req.get_loop();
                                        let unlink_req = FsRequest::new();
                                        do unlink_req.unlink(loop_, &Path(path_str))
                                        |_,uverr| {
                                            assert!(uverr.is_none());
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            };
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn file_test_full_simple_sync() {
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
            let open_req = FsRequest::new();
            let result = open_req.open_sync(&loop_, &Path(path_str),
                                                   create_flags as int, mode as int);
            assert!(result.is_ok());
            let fd = result.unwrap();
            // write
            let write_req = FsRequest::new();
            let result = write_req.write_sync(&loop_, fd, write_buf, -1);
            assert!(result.is_ok());
            // close
            let close_req = FsRequest::new();
            let result = close_req.close_sync(&loop_, fd);
            assert!(result.is_ok());
            // re-open
            let open_req = FsRequest::new();
            let result = open_req.open_sync(&loop_, &Path(path_str),
                                                   read_flags as int,0);
            assert!(result.is_ok());
            let len = 1028;
            let fd = result.unwrap();
            // read
            let read_mem: ~[u8] = vec::from_elem(len, 0u8);
            let buf = slice_to_uv_buf(read_mem);
            let read_req = FsRequest::new();
            let result = read_req.read_sync(&loop_, fd, buf, 0);
            assert!(result.is_ok());
            let nread = result.unwrap();
            // nread == 0 would be EOF.. we know it's >= zero because otherwise
            // the above assert would fail
            if nread > 0 {
                let read_str = str::from_utf8(
                    read_mem.slice(0, nread as uint));
                assert!(read_str == ~"hello");
                // close
                let close_req = FsRequest::new();
                let result = close_req.close_sync(&loop_, fd);
                assert!(result.is_ok());
                // unlink
                let unlink_req = FsRequest::new();
                let result = unlink_req.unlink_sync(&loop_, &Path(path_str));
                assert!(result.is_ok());
            } else { fail2!("nread was 0.. wudn't expectin' that."); }
            loop_.close();
        }
    }

    fn naive_print(loop_: &Loop, input: &str) {
        let write_val = input.as_bytes();
        let write_buf = slice_to_uv_buf(write_val);
        let write_req = FsRequest::new();
        write_req.write_sync(loop_, STDOUT_FILENO, write_buf, -1);
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
    #[test]
    fn file_test_stat_simple() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let path = "./tmp/file_test_stat_simple.txt";
            let create_flags = O_RDWR |
                O_CREAT;
            let mode = S_IWUSR |
                S_IRUSR;
            let write_val = "hello".as_bytes().to_owned();
            let write_buf  = slice_to_uv_buf(write_val);
            let write_buf_ptr: *Buf = &write_buf;
            let open_req = FsRequest::new();
            do open_req.open(&loop_, &path, create_flags as int, mode as int)
            |req, uverr| {
                assert!(uverr.is_none());
                let fd = req.get_result();
                let buf = unsafe { *write_buf_ptr };
                let write_req = FsRequest::new();
                do write_req.write(&req.get_loop(), fd, buf, 0) |req, uverr| {
                    assert!(uverr.is_none());
                    let loop_ = req.get_loop();
                    let stat_req = FsRequest::new();
                    do stat_req.stat(&loop_, &path) |req, uverr| {
                        assert!(uverr.is_none());
                        let loop_ = req.get_loop();
                        let stat = req.get_stat();
                        let sz: uint = stat.st_size as uint;
                        assert!(sz > 0);
                        let close_req = FsRequest::new();
                        do close_req.close(&loop_, fd) |req, uverr| {
                            assert!(uverr.is_none());
                            let loop_ = req.get_loop();
                            let unlink_req = FsRequest::new();
                            do unlink_req.unlink(&loop_, &path) |req,uverr| {
                                assert!(uverr.is_none());
                                let loop_ = req.get_loop();
                                let stat_req = FsRequest::new();
                                do stat_req.stat(&loop_, &path) |_, uverr| {
                                    // should cause an error because the
                                    // file doesn't exist anymore
                                    assert!(uverr.is_some());
                                };
                            };
                        };
                    };
                };
            };
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn file_test_mk_rm_dir() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let path = "./tmp/mk_rm_dir";
            let mode = S_IWUSR |
                S_IRUSR;
            let mkdir_req = FsRequest::new();
            do mkdir_req.mkdir(&loop_, &path, mode as int) |req,uverr| {
                assert!(uverr.is_none());
                let loop_ = req.get_loop();
                let stat_req = FsRequest::new();
                do stat_req.stat(&loop_, &path) |req, uverr| {
                    assert!(uverr.is_none());
                    let loop_ = req.get_loop();
                    let stat = req.get_stat();
                    naive_print(&loop_, format!("{:?}", stat));
                    assert!(stat.is_dir());
                    let rmdir_req = FsRequest::new();
                    do rmdir_req.rmdir(&loop_, &path) |req,uverr| {
                        assert!(uverr.is_none());
                        let loop_ = req.get_loop();
                        let stat_req = FsRequest::new();
                        do stat_req.stat(&loop_, &path) |_req, uverr| {
                            assert!(uverr.is_some());
                        }
                    }
                }
            }
            loop_.run();
            loop_.close();
        }
    }
    #[test]
    fn file_test_mkdir_chokes_on_double_create() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let path = "./tmp/double_create_dir";
            let mode = S_IWUSR |
                S_IRUSR;
            let mkdir_req = FsRequest::new();
            do mkdir_req.mkdir(&loop_, &path, mode as int) |req,uverr| {
                assert!(uverr.is_none());
                let loop_ = req.get_loop();
                let mkdir_req = FsRequest::new();
                do mkdir_req.mkdir(&loop_, &path, mode as int) |req,uverr| {
                    assert!(uverr.is_some());
                    let loop_ = req.get_loop();
                    let _stat = req.get_stat();
                    let rmdir_req = FsRequest::new();
                    do rmdir_req.rmdir(&loop_, &path) |req,uverr| {
                        assert!(uverr.is_none());
                        let _loop = req.get_loop();
                    }
                }
            }
            loop_.run();
            loop_.close();
        }
    }
    #[test]
    fn file_test_rmdir_chokes_on_nonexistant_path() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let path = "./tmp/never_existed_dir";
            let rmdir_req = FsRequest::new();
            do rmdir_req.rmdir(&loop_, &path) |_req, uverr| {
                assert!(uverr.is_some());
            }
            loop_.run();
            loop_.close();
        }
    }
}
