// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Low-level bindings to the libuv library.
 *
 * This module contains a set of direct, 'bare-metal' wrappers around
 * the libuv C-API.
 *
 * We're not bothering yet to redefine uv's structs as Rust structs
 * because they are quite large and change often between versions.
 * The maintenance burden is just too high. Instead we use the uv's
 * `uv_handle_size` and `uv_req_size` to find the correct size of the
 * structs and allocate them on the heap. This can be revisited later.
 *
 * There are also a collection of helper functions to ease interacting
 * with the low-level API.
 *
 * As new functionality, existent in uv.h, is added to the rust stdlib,
 * the mappings should be added in this module.
 */

#![allow(non_camel_case_types)] // C types

use libc::{size_t, c_int, c_uint, c_void, c_char, c_double};
use libc::{ssize_t, sockaddr, free, addrinfo};
use libc;
use std::rt::libc_heap::malloc_raw;

#[cfg(test)]
use libc::uintptr_t;

pub use self::errors::{EACCES, ECONNREFUSED, ECONNRESET, EPIPE, ECONNABORTED,
                       ECANCELED, EBADF, ENOTCONN, ENOENT, EADDRNOTAVAIL,
                       EADDRINUSE, EPERM};

pub static OK: c_int = 0;
pub static EOF: c_int = -4095;
pub static UNKNOWN: c_int = -4094;

// uv-errno.h redefines error codes for windows, but not for unix...
// https://github.com/joyent/libuv/blob/master/include/uv-errno.h

#[cfg(windows)]
pub mod errors {
    use libc::c_int;

    pub static EACCES: c_int = -4092;
    pub static ECONNREFUSED: c_int = -4078;
    pub static ECONNRESET: c_int = -4077;
    pub static ENOENT: c_int = -4058;
    pub static ENOTCONN: c_int = -4053;
    pub static EPIPE: c_int = -4047;
    pub static ECONNABORTED: c_int = -4079;
    pub static ECANCELED: c_int = -4081;
    pub static EBADF: c_int = -4083;
    pub static EADDRNOTAVAIL: c_int = -4090;
    pub static EADDRINUSE: c_int = -4091;
    pub static EPERM: c_int = -4048;
}
#[cfg(not(windows))]
pub mod errors {
    use libc;
    use libc::c_int;

    pub static EACCES: c_int = -libc::EACCES;
    pub static ECONNREFUSED: c_int = -libc::ECONNREFUSED;
    pub static ECONNRESET: c_int = -libc::ECONNRESET;
    pub static ENOENT: c_int = -libc::ENOENT;
    pub static ENOTCONN: c_int = -libc::ENOTCONN;
    pub static EPIPE: c_int = -libc::EPIPE;
    pub static ECONNABORTED: c_int = -libc::ECONNABORTED;
    pub static ECANCELED : c_int = -libc::ECANCELED;
    pub static EBADF : c_int = -libc::EBADF;
    pub static EADDRNOTAVAIL : c_int = -libc::EADDRNOTAVAIL;
    pub static EADDRINUSE : c_int = -libc::EADDRINUSE;
    pub static EPERM: c_int = -libc::EPERM;
}

pub static PROCESS_SETUID: c_int = 1 << 0;
pub static PROCESS_SETGID: c_int = 1 << 1;
pub static PROCESS_WINDOWS_VERBATIM_ARGUMENTS: c_int = 1 << 2;
pub static PROCESS_DETACHED: c_int = 1 << 3;
pub static PROCESS_WINDOWS_HIDE: c_int = 1 << 4;

pub static STDIO_IGNORE: c_int = 0x00;
pub static STDIO_CREATE_PIPE: c_int = 0x01;
pub static STDIO_INHERIT_FD: c_int = 0x02;
pub static STDIO_INHERIT_STREAM: c_int = 0x04;
pub static STDIO_READABLE_PIPE: c_int = 0x10;
pub static STDIO_WRITABLE_PIPE: c_int = 0x20;

#[cfg(unix)]
pub type uv_buf_len_t = libc::size_t;
#[cfg(windows)]
pub type uv_buf_len_t = libc::c_ulong;

// see libuv/include/uv-unix.h
#[cfg(unix)]
pub struct uv_buf_t {
    pub base: *mut u8,
    pub len: uv_buf_len_t,
}

#[cfg(unix)]
pub type uv_os_socket_t = c_int;

// see libuv/include/uv-win.h
#[cfg(windows)]
pub struct uv_buf_t {
    pub len: uv_buf_len_t,
    pub base: *mut u8,
}

#[cfg(windows)]
pub type uv_os_socket_t = libc::SOCKET;

#[repr(C)]
pub enum uv_run_mode {
    RUN_DEFAULT = 0,
    RUN_ONCE,
    RUN_NOWAIT,
}

#[repr(C)]
pub enum uv_poll_event {
    UV_READABLE = 1,
    UV_WRITABLE = 2,
}

pub struct uv_process_options_t {
    pub exit_cb: uv_exit_cb,
    pub file: *const libc::c_char,
    pub args: *const *const libc::c_char,
    pub env: *const *const libc::c_char,
    pub cwd: *const libc::c_char,
    pub flags: libc::c_uint,
    pub stdio_count: libc::c_int,
    pub stdio: *mut uv_stdio_container_t,
    pub uid: uv_uid_t,
    pub gid: uv_gid_t,
}

// These fields are private because they must be interfaced with through the
// functions below.
#[repr(C)]
pub struct uv_stdio_container_t {
    flags: libc::c_int,
    stream: *mut uv_stream_t,
}

pub type uv_handle_t = c_void;
pub type uv_req_t = c_void;
pub type uv_loop_t = c_void;
pub type uv_idle_t = c_void;
pub type uv_tcp_t = c_void;
pub type uv_udp_t = c_void;
pub type uv_poll_t = c_void;
pub type uv_connect_t = c_void;
pub type uv_connection_t = c_void;
pub type uv_write_t = c_void;
pub type uv_async_t = c_void;
pub type uv_timer_t = c_void;
pub type uv_stream_t = c_void;
pub type uv_fs_t = c_void;
pub type uv_udp_send_t = c_void;
pub type uv_getaddrinfo_t = c_void;
pub type uv_process_t = c_void;
pub type uv_pipe_t = c_void;
pub type uv_tty_t = c_void;
pub type uv_signal_t = c_void;
pub type uv_shutdown_t = c_void;

pub struct uv_timespec_t {
    pub tv_sec: libc::c_long,
    pub tv_nsec: libc::c_long
}

pub struct uv_stat_t {
    pub st_dev: libc::uint64_t,
    pub st_mode: libc::uint64_t,
    pub st_nlink: libc::uint64_t,
    pub st_uid: libc::uint64_t,
    pub st_gid: libc::uint64_t,
    pub st_rdev: libc::uint64_t,
    pub st_ino: libc::uint64_t,
    pub st_size: libc::uint64_t,
    pub st_blksize: libc::uint64_t,
    pub st_blocks: libc::uint64_t,
    pub st_flags: libc::uint64_t,
    pub st_gen: libc::uint64_t,
    pub st_atim: uv_timespec_t,
    pub st_mtim: uv_timespec_t,
    pub st_ctim: uv_timespec_t,
    pub st_birthtim: uv_timespec_t
}

impl uv_stat_t {
    pub fn new() -> uv_stat_t {
        uv_stat_t {
            st_dev: 0,
            st_mode: 0,
            st_nlink: 0,
            st_uid: 0,
            st_gid: 0,
            st_rdev: 0,
            st_ino: 0,
            st_size: 0,
            st_blksize: 0,
            st_blocks: 0,
            st_flags: 0,
            st_gen: 0,
            st_atim: uv_timespec_t { tv_sec: 0, tv_nsec: 0 },
            st_mtim: uv_timespec_t { tv_sec: 0, tv_nsec: 0 },
            st_ctim: uv_timespec_t { tv_sec: 0, tv_nsec: 0 },
            st_birthtim: uv_timespec_t { tv_sec: 0, tv_nsec: 0 }
        }
    }
    pub fn is_file(&self) -> bool {
        ((self.st_mode) & libc::S_IFMT as libc::uint64_t) == libc::S_IFREG as libc::uint64_t
    }
    pub fn is_dir(&self) -> bool {
        ((self.st_mode) & libc::S_IFMT as libc::uint64_t) == libc::S_IFDIR as libc::uint64_t
    }
}

pub type uv_idle_cb = extern "C" fn(handle: *mut uv_idle_t);
pub type uv_alloc_cb = extern "C" fn(stream: *mut uv_stream_t,
                                     suggested_size: size_t,
                                     buf: *mut uv_buf_t);
pub type uv_read_cb = extern "C" fn(stream: *mut uv_stream_t,
                                    nread: ssize_t,
                                    buf: *const uv_buf_t);
pub type uv_udp_send_cb = extern "C" fn(req: *mut uv_udp_send_t,
                                        status: c_int);
pub type uv_udp_recv_cb = extern "C" fn(handle: *mut uv_udp_t,
                                        nread: ssize_t,
                                        buf: *const uv_buf_t,
                                        addr: *const sockaddr,
                                        flags: c_uint);
pub type uv_close_cb = extern "C" fn(handle: *mut uv_handle_t);
pub type uv_poll_cb = extern "C" fn(handle: *mut uv_poll_t,
                                    status: c_int,
                                    events: c_int);
pub type uv_walk_cb = extern "C" fn(handle: *mut uv_handle_t,
                                    arg: *mut c_void);
pub type uv_async_cb = extern "C" fn(handle: *mut uv_async_t);
pub type uv_connect_cb = extern "C" fn(handle: *mut uv_connect_t,
                                       status: c_int);
pub type uv_connection_cb = extern "C" fn(handle: *mut uv_connection_t,
                                          status: c_int);
pub type uv_timer_cb = extern "C" fn(handle: *mut uv_timer_t);
pub type uv_write_cb = extern "C" fn(handle: *mut uv_write_t,
                                     status: c_int);
pub type uv_getaddrinfo_cb = extern "C" fn(req: *mut uv_getaddrinfo_t,
                                           status: c_int,
                                           res: *const addrinfo);
pub type uv_exit_cb = extern "C" fn(handle: *mut uv_process_t,
                                    exit_status: i64,
                                    term_signal: c_int);
pub type uv_signal_cb = extern "C" fn(handle: *mut uv_signal_t,
                                      signum: c_int);
pub type uv_fs_cb = extern "C" fn(req: *mut uv_fs_t);
pub type uv_shutdown_cb = extern "C" fn(req: *mut uv_shutdown_t, status: c_int);

#[cfg(unix)] pub type uv_uid_t = libc::types::os::arch::posix88::uid_t;
#[cfg(unix)] pub type uv_gid_t = libc::types::os::arch::posix88::gid_t;
#[cfg(windows)] pub type uv_uid_t = libc::c_uchar;
#[cfg(windows)] pub type uv_gid_t = libc::c_uchar;

#[repr(C)]
#[deriving(PartialEq)]
pub enum uv_handle_type {
    UV_UNKNOWN_HANDLE,
    UV_ASYNC,
    UV_CHECK,
    UV_FS_EVENT,
    UV_FS_POLL,
    UV_HANDLE,
    UV_IDLE,
    UV_NAMED_PIPE,
    UV_POLL,
    UV_PREPARE,
    UV_PROCESS,
    UV_STREAM,
    UV_TCP,
    UV_TIMER,
    UV_TTY,
    UV_UDP,
    UV_SIGNAL,
    UV_FILE,
    UV_HANDLE_TYPE_MAX
}

#[repr(C)]
#[cfg(unix)]
#[deriving(PartialEq)]
pub enum uv_req_type {
    UV_UNKNOWN_REQ,
    UV_REQ,
    UV_CONNECT,
    UV_WRITE,
    UV_SHUTDOWN,
    UV_UDP_SEND,
    UV_FS,
    UV_WORK,
    UV_GETADDRINFO,
    UV_GETNAMEINFO,
    UV_REQ_TYPE_MAX
}

// uv_req_type may have additional fields defined by UV_REQ_TYPE_PRIVATE.
// See UV_REQ_TYPE_PRIVATE at libuv/include/uv-win.h
#[repr(C)]
#[cfg(windows)]
#[deriving(PartialEq)]
pub enum uv_req_type {
    UV_UNKNOWN_REQ,
    UV_REQ,
    UV_CONNECT,
    UV_WRITE,
    UV_SHUTDOWN,
    UV_UDP_SEND,
    UV_FS,
    UV_WORK,
    UV_GETNAMEINFO,
    UV_GETADDRINFO,
    UV_ACCEPT,
    UV_FS_EVENT_REQ,
    UV_POLL_REQ,
    UV_PROCESS_EXIT,
    UV_READ,
    UV_UDP_RECV,
    UV_WAKEUP,
    UV_SIGNAL_REQ,
    UV_REQ_TYPE_MAX
}

#[repr(C)]
#[deriving(PartialEq)]
pub enum uv_membership {
    UV_LEAVE_GROUP,
    UV_JOIN_GROUP
}

pub unsafe fn malloc_handle(handle: uv_handle_type) -> *mut c_void {
    assert!(handle != UV_UNKNOWN_HANDLE && handle != UV_HANDLE_TYPE_MAX);
    let size = uv_handle_size(handle);
    malloc_raw(size as uint) as *mut c_void
}

pub unsafe fn free_handle(v: *mut c_void) {
    free(v as *mut c_void)
}

pub unsafe fn malloc_req(req: uv_req_type) -> *mut c_void {
    assert!(req != UV_UNKNOWN_REQ && req != UV_REQ_TYPE_MAX);
    let size = uv_req_size(req);
    malloc_raw(size as uint) as *mut c_void
}

pub unsafe fn free_req(v: *mut c_void) {
    free(v as *mut c_void)
}

#[test]
fn handle_sanity_check() {
    unsafe {
        assert_eq!(UV_HANDLE_TYPE_MAX as uint, rust_uv_handle_type_max());
    }
}

#[test]
fn request_sanity_check() {
    unsafe {
        assert_eq!(UV_REQ_TYPE_MAX as uint, rust_uv_req_type_max());
    }
}

// FIXME Event loops ignore SIGPIPE by default.
pub unsafe fn loop_new() -> *mut c_void {
    return rust_uv_loop_new();
}

pub unsafe fn uv_write(req: *mut uv_write_t,
                       stream: *mut uv_stream_t,
                       buf_in: &[uv_buf_t],
                       cb: uv_write_cb) -> c_int {
    extern {
        fn uv_write(req: *mut uv_write_t, stream: *mut uv_stream_t,
                    buf_in: *const uv_buf_t, buf_cnt: c_int,
                    cb: uv_write_cb) -> c_int;
    }

    let buf_ptr = buf_in.as_ptr();
    let buf_cnt = buf_in.len() as i32;
    return uv_write(req, stream, buf_ptr, buf_cnt, cb);
}

pub unsafe fn uv_udp_send(req: *mut uv_udp_send_t,
                          handle: *mut uv_udp_t,
                          buf_in: &[uv_buf_t],
                          addr: *const sockaddr,
                          cb: uv_udp_send_cb) -> c_int {
    extern {
        fn uv_udp_send(req: *mut uv_write_t, stream: *mut uv_stream_t,
                       buf_in: *const uv_buf_t, buf_cnt: c_int,
                       addr: *const sockaddr,
                       cb: uv_udp_send_cb) -> c_int;
    }

    let buf_ptr = buf_in.as_ptr();
    let buf_cnt = buf_in.len() as i32;
    return uv_udp_send(req, handle, buf_ptr, buf_cnt, addr, cb);
}

pub unsafe fn get_udp_handle_from_send_req(send_req: *mut uv_udp_send_t) -> *mut uv_udp_t {
    return rust_uv_get_udp_handle_from_send_req(send_req);
}

pub unsafe fn process_pid(p: *mut uv_process_t) -> c_int {

    return rust_uv_process_pid(p);
}

pub unsafe fn set_stdio_container_flags(c: *mut uv_stdio_container_t,
                                        flags: libc::c_int) {

    rust_set_stdio_container_flags(c, flags);
}

pub unsafe fn set_stdio_container_fd(c: *mut uv_stdio_container_t,
                                     fd: libc::c_int) {

    rust_set_stdio_container_fd(c, fd);
}

pub unsafe fn set_stdio_container_stream(c: *mut uv_stdio_container_t,
                                         stream: *mut uv_stream_t) {
    rust_set_stdio_container_stream(c, stream);
}

// data access helpers
pub unsafe fn get_result_from_fs_req(req: *mut uv_fs_t) -> ssize_t {
    rust_uv_get_result_from_fs_req(req)
}
pub unsafe fn get_ptr_from_fs_req(req: *mut uv_fs_t) -> *mut libc::c_void {
    rust_uv_get_ptr_from_fs_req(req)
}
pub unsafe fn get_path_from_fs_req(req: *mut uv_fs_t) -> *mut c_char {
    rust_uv_get_path_from_fs_req(req)
}
pub unsafe fn get_loop_from_fs_req(req: *mut uv_fs_t) -> *mut uv_loop_t {
    rust_uv_get_loop_from_fs_req(req)
}
pub unsafe fn get_loop_from_getaddrinfo_req(req: *mut uv_getaddrinfo_t) -> *mut uv_loop_t {
    rust_uv_get_loop_from_getaddrinfo_req(req)
}
pub unsafe fn get_loop_for_uv_handle<T>(handle: *mut T) -> *mut c_void {
    return rust_uv_get_loop_for_uv_handle(handle as *mut c_void);
}
pub unsafe fn get_stream_handle_from_connect_req(connect: *mut uv_connect_t) -> *mut uv_stream_t {
    return rust_uv_get_stream_handle_from_connect_req(connect);
}
pub unsafe fn get_stream_handle_from_write_req(write_req: *mut uv_write_t) -> *mut uv_stream_t {
    return rust_uv_get_stream_handle_from_write_req(write_req);
}
pub unsafe fn get_data_for_uv_loop(loop_ptr: *mut c_void) -> *mut c_void {
    rust_uv_get_data_for_uv_loop(loop_ptr)
}
pub unsafe fn set_data_for_uv_loop(loop_ptr: *mut c_void, data: *mut c_void) {
    rust_uv_set_data_for_uv_loop(loop_ptr, data);
}
pub unsafe fn get_data_for_uv_handle<T>(handle: *mut T) -> *mut c_void {
    return rust_uv_get_data_for_uv_handle(handle as *mut c_void);
}
pub unsafe fn set_data_for_uv_handle<T, U>(handle: *mut T, data: *mut U) {
    rust_uv_set_data_for_uv_handle(handle as *mut c_void, data as *mut c_void);
}
pub unsafe fn get_data_for_req<T>(req: *mut T) -> *mut c_void {
    return rust_uv_get_data_for_req(req as *mut c_void);
}
pub unsafe fn set_data_for_req<T, U>(req: *mut T, data: *mut U) {
    rust_uv_set_data_for_req(req as *mut c_void, data as *mut c_void);
}
pub unsafe fn populate_stat(req_in: *mut uv_fs_t, stat_out: *mut uv_stat_t) {
    rust_uv_populate_uv_stat(req_in, stat_out)
}
pub unsafe fn guess_handle(handle: c_int) -> c_int {
    rust_uv_guess_handle(handle)
}


// uv_support is the result of compiling rust_uv.cpp
//
// Note that this is in a cfg'd block so it doesn't get linked during testing.
// There's a bit of a conundrum when testing in that we're actually assuming
// that the tests are running in a uv loop, but they were created from the
// statically linked uv to the original rustuv crate. When we create the test
// executable, on some platforms if we re-link against uv, it actually creates
// second copies of everything. We obviously don't want this, so instead of
// dying horribly during testing, we allow all of the test rustuv's references
// to get resolved to the original rustuv crate.
#[cfg(not(test))]
#[link(name = "uv_support", kind = "static")]
#[link(name = "uv", kind = "static")]
extern {}

extern {
    fn rust_uv_loop_new() -> *mut c_void;

    #[cfg(test)]
    fn rust_uv_handle_type_max() -> uintptr_t;
    #[cfg(test)]
    fn rust_uv_req_type_max() -> uintptr_t;
    fn rust_uv_get_udp_handle_from_send_req(req: *mut uv_udp_send_t) -> *mut uv_udp_t;

    fn rust_uv_populate_uv_stat(req_in: *mut uv_fs_t, stat_out: *mut uv_stat_t);
    fn rust_uv_get_result_from_fs_req(req: *mut uv_fs_t) -> ssize_t;
    fn rust_uv_get_ptr_from_fs_req(req: *mut uv_fs_t) -> *mut libc::c_void;
    fn rust_uv_get_path_from_fs_req(req: *mut uv_fs_t) -> *mut c_char;
    fn rust_uv_get_loop_from_fs_req(req: *mut uv_fs_t) -> *mut uv_loop_t;
    fn rust_uv_get_loop_from_getaddrinfo_req(req: *mut uv_fs_t) -> *mut uv_loop_t;
    fn rust_uv_get_stream_handle_from_connect_req(req: *mut uv_connect_t) -> *mut uv_stream_t;
    fn rust_uv_get_stream_handle_from_write_req(req: *mut uv_write_t) -> *mut uv_stream_t;
    fn rust_uv_get_loop_for_uv_handle(handle: *mut c_void) -> *mut c_void;
    fn rust_uv_get_data_for_uv_loop(loop_ptr: *mut c_void) -> *mut c_void;
    fn rust_uv_set_data_for_uv_loop(loop_ptr: *mut c_void, data: *mut c_void);
    fn rust_uv_get_data_for_uv_handle(handle: *mut c_void) -> *mut c_void;
    fn rust_uv_set_data_for_uv_handle(handle: *mut c_void, data: *mut c_void);
    fn rust_uv_get_data_for_req(req: *mut c_void) -> *mut c_void;
    fn rust_uv_set_data_for_req(req: *mut c_void, data: *mut c_void);
    fn rust_set_stdio_container_flags(c: *mut uv_stdio_container_t, flags: c_int);
    fn rust_set_stdio_container_fd(c: *mut uv_stdio_container_t, fd: c_int);
    fn rust_set_stdio_container_stream(c: *mut uv_stdio_container_t,
                                       stream: *mut uv_stream_t);
    fn rust_uv_process_pid(p: *mut uv_process_t) -> c_int;
    fn rust_uv_guess_handle(fd: c_int) -> c_int;

    // generic uv functions
    pub fn uv_loop_delete(l: *mut uv_loop_t);
    pub fn uv_ref(t: *mut uv_handle_t);
    pub fn uv_unref(t: *mut uv_handle_t);
    pub fn uv_handle_size(ty: uv_handle_type) -> size_t;
    pub fn uv_req_size(ty: uv_req_type) -> size_t;
    pub fn uv_run(l: *mut uv_loop_t, mode: uv_run_mode) -> c_int;
    pub fn uv_close(h: *mut uv_handle_t, cb: uv_close_cb);
    pub fn uv_walk(l: *mut uv_loop_t, cb: uv_walk_cb, arg: *mut c_void);
    pub fn uv_buf_init(base: *mut c_char, len: c_uint) -> uv_buf_t;
    pub fn uv_strerror(err: c_int) -> *const c_char;
    pub fn uv_err_name(err: c_int) -> *const c_char;
    pub fn uv_listen(s: *mut uv_stream_t, backlog: c_int,
                     cb: uv_connection_cb) -> c_int;
    pub fn uv_accept(server: *mut uv_stream_t, client: *mut uv_stream_t) -> c_int;
    pub fn uv_read_start(stream: *mut uv_stream_t,
                         on_alloc: uv_alloc_cb,
                         on_read: uv_read_cb) -> c_int;
    pub fn uv_read_stop(stream: *mut uv_stream_t) -> c_int;
    pub fn uv_shutdown(req: *mut uv_shutdown_t, handle: *mut uv_stream_t,
                       cb: uv_shutdown_cb) -> c_int;

    // idle bindings
    pub fn uv_idle_init(l: *mut uv_loop_t, i: *mut uv_idle_t) -> c_int;
    pub fn uv_idle_start(i: *mut uv_idle_t, cb: uv_idle_cb) -> c_int;
    pub fn uv_idle_stop(i: *mut uv_idle_t) -> c_int;

    // async bindings
    pub fn uv_async_init(l: *mut uv_loop_t, a: *mut uv_async_t,
                         cb: uv_async_cb) -> c_int;
    pub fn uv_async_send(a: *mut uv_async_t);

    // tcp bindings
    pub fn uv_tcp_init(l: *mut uv_loop_t, h: *mut uv_tcp_t) -> c_int;
    pub fn uv_tcp_connect(c: *mut uv_connect_t, h: *mut uv_tcp_t,
                          addr: *const sockaddr, cb: uv_connect_cb) -> c_int;
    pub fn uv_tcp_bind(t: *mut uv_tcp_t,
                       addr: *const sockaddr,
                       flags: c_uint) -> c_int;
    pub fn uv_tcp_nodelay(h: *mut uv_tcp_t, enable: c_int) -> c_int;
    pub fn uv_tcp_keepalive(h: *mut uv_tcp_t, enable: c_int,
                            delay: c_uint) -> c_int;
    pub fn uv_tcp_simultaneous_accepts(h: *mut uv_tcp_t, enable: c_int) -> c_int;
    pub fn uv_tcp_getsockname(h: *const uv_tcp_t, name: *mut sockaddr,
                              len: *mut c_int) -> c_int;
    pub fn uv_tcp_getpeername(h: *const uv_tcp_t, name: *mut sockaddr,
                              len: *mut c_int) -> c_int;

    // udp bindings
    pub fn uv_udp_init(l: *mut uv_loop_t, h: *mut uv_udp_t) -> c_int;
    pub fn uv_udp_bind(h: *mut uv_udp_t, addr: *const sockaddr,
                       flags: c_uint) -> c_int;
    pub fn uv_udp_recv_start(server: *mut uv_udp_t,
                             on_alloc: uv_alloc_cb,
                             on_recv: uv_udp_recv_cb) -> c_int;
    pub fn uv_udp_set_membership(handle: *mut uv_udp_t,
                                 multicast_addr: *const c_char,
                                 interface_addr: *const c_char,
                                 membership: uv_membership) -> c_int;
    pub fn uv_udp_recv_stop(server: *mut uv_udp_t) -> c_int;
    pub fn uv_udp_set_multicast_loop(handle: *mut uv_udp_t, on: c_int) -> c_int;
    pub fn uv_udp_set_multicast_ttl(handle: *mut uv_udp_t, ttl: c_int) -> c_int;
    pub fn uv_udp_set_ttl(handle: *mut uv_udp_t, ttl: c_int) -> c_int;
    pub fn uv_udp_set_broadcast(handle: *mut uv_udp_t, on: c_int) -> c_int;
    pub fn uv_udp_getsockname(h: *const uv_udp_t, name: *mut sockaddr,
                              len: *mut c_int) -> c_int;

    // timer bindings
    pub fn uv_timer_init(l: *mut uv_loop_t, t: *mut uv_timer_t) -> c_int;
    pub fn uv_timer_start(t: *mut uv_timer_t, cb: uv_timer_cb,
                          timeout: libc::uint64_t,
                          repeat: libc::uint64_t) -> c_int;
    pub fn uv_timer_stop(handle: *mut uv_timer_t) -> c_int;

    // fs operations
    pub fn uv_fs_open(loop_ptr: *mut uv_loop_t, req: *mut uv_fs_t,
                      path: *const c_char, flags: c_int, mode: c_int,
                      cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_unlink(loop_ptr: *mut uv_loop_t, req: *mut uv_fs_t,
                        path: *const c_char, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_write(l: *mut uv_loop_t, req: *mut uv_fs_t, fd: c_int,
                       bufs: *const uv_buf_t, nbufs: c_uint,
                       offset: i64, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_read(l: *mut uv_loop_t, req: *mut uv_fs_t, fd: c_int,
                      bufs: *mut uv_buf_t, nbufs: c_uint,
                      offset: i64, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_close(l: *mut uv_loop_t, req: *mut uv_fs_t, fd: c_int,
                       cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_stat(l: *mut uv_loop_t, req: *mut uv_fs_t, path: *const c_char,
                      cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_fstat(l: *mut uv_loop_t, req: *mut uv_fs_t, fd: c_int,
                       cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_mkdir(l: *mut uv_loop_t, req: *mut uv_fs_t, path: *const c_char,
                       mode: c_int, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_rmdir(l: *mut uv_loop_t, req: *mut uv_fs_t, path: *const c_char,
                       cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_readdir(l: *mut uv_loop_t, req: *mut uv_fs_t,
                         path: *const c_char, flags: c_int,
                         cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_req_cleanup(req: *mut uv_fs_t);
    pub fn uv_fs_fsync(handle: *mut uv_loop_t, req: *mut uv_fs_t, file: c_int,
                       cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_fdatasync(handle: *mut uv_loop_t, req: *mut uv_fs_t, file: c_int,
                           cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_ftruncate(handle: *mut uv_loop_t, req: *mut uv_fs_t, file: c_int,
                           offset: i64, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_readlink(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                          file: *const c_char, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_symlink(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                         src: *const c_char, dst: *const c_char, flags: c_int,
                         cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_rename(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                        src: *const c_char, dst: *const c_char,
                        cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_utime(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                       path: *const c_char, atime: c_double, mtime: c_double,
                       cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_link(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                      src: *const c_char, dst: *const c_char,
                      cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_chown(handle: *mut uv_loop_t, req: *mut uv_fs_t, src: *const c_char,
                       uid: uv_uid_t, gid: uv_gid_t, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_chmod(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                       path: *const c_char, mode: c_int, cb: uv_fs_cb) -> c_int;
    pub fn uv_fs_lstat(handle: *mut uv_loop_t, req: *mut uv_fs_t,
                       file: *const c_char, cb: uv_fs_cb) -> c_int;

    // poll bindings
    pub fn uv_poll_init_socket(l: *mut uv_loop_t, h: *mut uv_poll_t, s: uv_os_socket_t) -> c_int;
    pub fn uv_poll_start(h: *mut uv_poll_t, events: c_int, cb: uv_poll_cb) -> c_int;
    pub fn uv_poll_stop(h: *mut uv_poll_t) -> c_int;

    // getaddrinfo
    pub fn uv_getaddrinfo(loop_: *mut uv_loop_t, req: *mut uv_getaddrinfo_t,
                          getaddrinfo_cb: uv_getaddrinfo_cb,
                          node: *const c_char, service: *const c_char,
                          hints: *const addrinfo) -> c_int;
    pub fn uv_freeaddrinfo(ai: *mut addrinfo);

    // process spawning
    pub fn uv_spawn(loop_ptr: *mut uv_loop_t, outptr: *mut uv_process_t,
                    options: *mut uv_process_options_t) -> c_int;
    pub fn uv_process_kill(p: *mut uv_process_t, signum: c_int) -> c_int;
    pub fn uv_kill(pid: c_int, signum: c_int) -> c_int;

    // pipes
    pub fn uv_pipe_init(l: *mut uv_loop_t, p: *mut uv_pipe_t,
                        ipc: c_int) -> c_int;
    pub fn uv_pipe_open(pipe: *mut uv_pipe_t, file: c_int) -> c_int;
    pub fn uv_pipe_bind(pipe: *mut uv_pipe_t, name: *const c_char) -> c_int;
    pub fn uv_pipe_connect(req: *mut uv_connect_t, handle: *mut uv_pipe_t,
                           name: *const c_char, cb: uv_connect_cb);

    // tty
    pub fn uv_tty_init(l: *mut uv_loop_t, tty: *mut uv_tty_t, fd: c_int,
                       readable: c_int) -> c_int;
    pub fn uv_tty_set_mode(tty: *mut uv_tty_t, mode: c_int) -> c_int;
    pub fn uv_tty_get_winsize(tty: *mut uv_tty_t,
                              width: *mut c_int,
                              height: *mut c_int) -> c_int;

    // signals
    pub fn uv_signal_init(loop_: *mut uv_loop_t,
                          handle: *mut uv_signal_t) -> c_int;
    pub fn uv_signal_start(h: *mut uv_signal_t, cb: uv_signal_cb,
                           signum: c_int) -> c_int;
    pub fn uv_signal_stop(handle: *mut uv_signal_t) -> c_int;
}

// libuv requires other native libraries on various platforms. These are all
// listed here (for each platform)

// libuv doesn't use pthread on windows
// android libc (bionic) provides pthread, so no additional link is required
#[cfg(not(windows), not(target_os = "android"))]
#[link(name = "pthread")]
extern {}

#[cfg(target_os = "linux")]
#[cfg(target_os = "dragonfly")]
#[link(name = "rt")]
extern {}

#[cfg(target_os = "win32")]
#[link(name = "ws2_32")]
#[link(name = "psapi")]
#[link(name = "iphlpapi")]
extern {}

#[cfg(target_os = "freebsd")]
#[cfg(target_os = "dragonfly")]
#[link(name = "kvm")]
extern {}
