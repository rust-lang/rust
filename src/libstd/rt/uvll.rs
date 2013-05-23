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
 * As new functionality, existant in uv.h, is added to the rust stdlib,
 * the mappings should be added in this module.
 */

#[allow(non_camel_case_types)]; // C types

use libc::{size_t, c_int, c_uint, c_void, c_char, uintptr_t};
use libc::{malloc, free};
use prelude::*;

pub struct uv_err_t {
    code: c_int,
    sys_errno_: c_int
}

pub struct uv_buf_t {
    base: *u8,
    len: libc::size_t,
}

pub type uv_handle_t = c_void;
pub type uv_loop_t = c_void;
pub type uv_idle_t = c_void;
pub type uv_tcp_t = c_void;
pub type uv_connect_t = c_void;
pub type uv_write_t = c_void;
pub type uv_async_t = c_void;
pub type uv_timer_t = c_void;
pub type uv_stream_t = c_void;
pub type uv_fs_t = c_void;

pub type uv_idle_cb = *u8;

pub type sockaddr_in = c_void;
pub type sockaddr_in6 = c_void;

#[deriving(Eq)]
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

#[deriving(Eq)]
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
    UV_REQ_TYPE_MAX
}

pub unsafe fn malloc_handle(handle: uv_handle_type) -> *c_void {
    assert!(handle != UV_UNKNOWN_HANDLE && handle != UV_HANDLE_TYPE_MAX);
    let size = rust_uv_handle_size(handle as uint);
    let p = malloc(size);
    assert!(p.is_not_null());
    return p;
}

pub unsafe fn free_handle(v: *c_void) {
    free(v)
}

pub unsafe fn malloc_req(req: uv_req_type) -> *c_void {
    assert!(req != UV_UNKNOWN_REQ && req != UV_REQ_TYPE_MAX);
    let size = rust_uv_req_size(req as uint);
    let p = malloc(size);
    assert!(p.is_not_null());
    return p;
}

pub unsafe fn free_req(v: *c_void) {
    free(v)
}

#[test]
fn handle_sanity_check() {
    unsafe {
        assert!(UV_HANDLE_TYPE_MAX as uint == rust_uv_handle_type_max());
    }
}

#[test]
fn request_sanity_check() {
    unsafe {
        assert!(UV_REQ_TYPE_MAX as uint == rust_uv_req_type_max());
    }
}

pub unsafe fn loop_new() -> *c_void {
    return rust_uv_loop_new();
}

pub unsafe fn loop_delete(loop_handle: *c_void) {
    rust_uv_loop_delete(loop_handle);
}

pub unsafe fn run(loop_handle: *c_void) {
    rust_uv_run(loop_handle);
}

pub unsafe fn close<T>(handle: *T, cb: *u8) {
    rust_uv_close(handle as *c_void, cb);
}

pub unsafe fn walk(loop_handle: *c_void, cb: *u8, arg: *c_void) {
    rust_uv_walk(loop_handle, cb, arg);
}

pub unsafe fn idle_new() -> *uv_idle_t {
    rust_uv_idle_new()
}

pub unsafe fn idle_delete(handle: *uv_idle_t) {
    rust_uv_idle_delete(handle)
}

pub unsafe fn idle_init(loop_handle: *uv_loop_t, handle: *uv_idle_t) -> c_int {
    rust_uv_idle_init(loop_handle, handle)
}

pub unsafe fn idle_start(handle: *uv_idle_t, cb: uv_idle_cb) -> c_int {
    rust_uv_idle_start(handle, cb)
}

pub unsafe fn idle_stop(handle: *uv_idle_t) -> c_int {
    rust_uv_idle_stop(handle)
}

pub unsafe fn tcp_init(loop_handle: *c_void, handle: *uv_tcp_t) -> c_int {
    return rust_uv_tcp_init(loop_handle, handle);
}

// FIXME ref #2064
pub unsafe fn tcp_connect(connect_ptr: *uv_connect_t,
                          tcp_handle_ptr: *uv_tcp_t,
                          addr_ptr: *sockaddr_in,
                          after_connect_cb: *u8) -> c_int {
    return rust_uv_tcp_connect(connect_ptr, tcp_handle_ptr,
                                       after_connect_cb, addr_ptr);
}
// FIXME ref #2064
pub unsafe fn tcp_connect6(connect_ptr: *uv_connect_t,
                           tcp_handle_ptr: *uv_tcp_t,
                           addr_ptr: *sockaddr_in6,
                           after_connect_cb: *u8) -> c_int {
    return rust_uv_tcp_connect6(connect_ptr, tcp_handle_ptr,
                                        after_connect_cb, addr_ptr);
}
// FIXME ref #2064
pub unsafe fn tcp_bind(tcp_server_ptr: *uv_tcp_t, addr_ptr: *sockaddr_in) -> c_int {
    return rust_uv_tcp_bind(tcp_server_ptr, addr_ptr);
}
// FIXME ref #2064
pub unsafe fn tcp_bind6(tcp_server_ptr: *uv_tcp_t, addr_ptr: *sockaddr_in6) -> c_int {
    return rust_uv_tcp_bind6(tcp_server_ptr, addr_ptr);
}

pub unsafe fn tcp_getpeername(tcp_handle_ptr: *uv_tcp_t, name: *sockaddr_in) -> c_int {
    return rust_uv_tcp_getpeername(tcp_handle_ptr, name);
}

pub unsafe fn tcp_getpeername6(tcp_handle_ptr: *uv_tcp_t, name: *sockaddr_in6) ->c_int {
    return rust_uv_tcp_getpeername6(tcp_handle_ptr, name);
}

pub unsafe fn listen<T>(stream: *T, backlog: c_int, cb: *u8) -> c_int {
    return rust_uv_listen(stream as *c_void, backlog, cb);
}

pub unsafe fn accept(server: *c_void, client: *c_void) -> c_int {
    return rust_uv_accept(server as *c_void, client as *c_void);
}

pub unsafe fn write<T>(req: *uv_write_t, stream: *T, buf_in: &[uv_buf_t], cb: *u8) -> c_int {
    let buf_ptr = vec::raw::to_ptr(buf_in);
    let buf_cnt = buf_in.len() as i32;
    return rust_uv_write(req as *c_void, stream as *c_void, buf_ptr, buf_cnt, cb);
}
pub unsafe fn read_start(stream: *uv_stream_t, on_alloc: *u8, on_read: *u8) -> c_int {
    return rust_uv_read_start(stream as *c_void, on_alloc, on_read);
}

pub unsafe fn read_stop(stream: *uv_stream_t) -> c_int {
    return rust_uv_read_stop(stream as *c_void);
}

pub unsafe fn last_error(loop_handle: *c_void) -> uv_err_t {
    return rust_uv_last_error(loop_handle);
}

pub unsafe fn strerror(err: *uv_err_t) -> *c_char {
    return rust_uv_strerror(err);
}
pub unsafe fn err_name(err: *uv_err_t) -> *c_char {
    return rust_uv_err_name(err);
}

pub unsafe fn async_init(loop_handle: *c_void, async_handle: *uv_async_t, cb: *u8) -> c_int {
    return rust_uv_async_init(loop_handle, async_handle, cb);
}

pub unsafe fn async_send(async_handle: *uv_async_t) {
    return rust_uv_async_send(async_handle);
}
pub unsafe fn buf_init(input: *u8, len: uint) -> uv_buf_t {
    let out_buf = uv_buf_t { base: ptr::null(), len: 0 as size_t };
    let out_buf_ptr = ptr::to_unsafe_ptr(&out_buf);
    rust_uv_buf_init(out_buf_ptr, input, len as size_t);
    return out_buf;
}

pub unsafe fn timer_init(loop_ptr: *c_void, timer_ptr: *uv_timer_t) -> c_int {
    return rust_uv_timer_init(loop_ptr, timer_ptr);
}
pub unsafe fn timer_start(timer_ptr: *uv_timer_t, cb: *u8, timeout: uint,
                          repeat: uint) -> c_int {
    return rust_uv_timer_start(timer_ptr, cb, timeout as c_uint, repeat as c_uint);
}
pub unsafe fn timer_stop(timer_ptr: *uv_timer_t) -> c_int {
    return rust_uv_timer_stop(timer_ptr);
}

pub unsafe fn malloc_ip4_addr(ip: &str, port: int) -> *sockaddr_in {
    do str::as_c_str(ip) |ip_buf| {
        rust_uv_ip4_addrp(ip_buf as *u8, port as libc::c_int)
    }
}
pub unsafe fn malloc_ip6_addr(ip: &str, port: int) -> *sockaddr_in6 {
    do str::as_c_str(ip) |ip_buf| {
        rust_uv_ip6_addrp(ip_buf as *u8, port as libc::c_int)
    }
}

pub unsafe fn free_ip4_addr(addr: *sockaddr_in) {
    rust_uv_free_ip4_addr(addr);
}

pub unsafe fn free_ip6_addr(addr: *sockaddr_in6) {
    rust_uv_free_ip6_addr(addr);
}

// data access helpers
pub unsafe fn get_loop_for_uv_handle<T>(handle: *T) -> *c_void {
    return rust_uv_get_loop_for_uv_handle(handle as *c_void);
}
pub unsafe fn get_stream_handle_from_connect_req(connect: *uv_connect_t) -> *uv_stream_t {
    return rust_uv_get_stream_handle_from_connect_req(connect);
}
pub unsafe fn get_stream_handle_from_write_req(write_req: *uv_write_t) -> *uv_stream_t {
    return rust_uv_get_stream_handle_from_write_req(write_req);
}
pub unsafe fn get_data_for_uv_loop(loop_ptr: *c_void) -> *c_void {
    rust_uv_get_data_for_uv_loop(loop_ptr)
}
pub unsafe fn set_data_for_uv_loop(loop_ptr: *c_void, data: *c_void) {
    rust_uv_set_data_for_uv_loop(loop_ptr, data);
}
pub unsafe fn get_data_for_uv_handle<T>(handle: *T) -> *c_void {
    return rust_uv_get_data_for_uv_handle(handle as *c_void);
}
pub unsafe fn set_data_for_uv_handle<T, U>(handle: *T, data: *U) {
    rust_uv_set_data_for_uv_handle(handle as *c_void, data as *c_void);
}
pub unsafe fn get_data_for_req<T>(req: *T) -> *c_void {
    return rust_uv_get_data_for_req(req as *c_void);
}
pub unsafe fn set_data_for_req<T, U>(req: *T, data: *U) {
    rust_uv_set_data_for_req(req as *c_void, data as *c_void);
}
pub unsafe fn get_base_from_buf(buf: uv_buf_t) -> *u8 {
    return rust_uv_get_base_from_buf(buf);
}
pub unsafe fn get_len_from_buf(buf: uv_buf_t) -> size_t {
    return rust_uv_get_len_from_buf(buf);
}
pub unsafe fn malloc_buf_base_of(suggested_size: size_t) -> *u8 {
    return rust_uv_malloc_buf_base_of(suggested_size);
}
pub unsafe fn free_base_of_buf(buf: uv_buf_t) {
    rust_uv_free_base_of_buf(buf);
}

pub unsafe fn get_last_err_info(uv_loop: *c_void) -> ~str {
    let err = last_error(uv_loop);
    let err_ptr = ptr::to_unsafe_ptr(&err);
    let err_name = str::raw::from_c_str(err_name(err_ptr));
    let err_msg = str::raw::from_c_str(strerror(err_ptr));
    return fmt!("LIBUV ERROR: name: %s msg: %s",
                    err_name, err_msg);
}

pub unsafe fn get_last_err_data(uv_loop: *c_void) -> uv_err_data {
    let err = last_error(uv_loop);
    let err_ptr = ptr::to_unsafe_ptr(&err);
    let err_name = str::raw::from_c_str(err_name(err_ptr));
    let err_msg = str::raw::from_c_str(strerror(err_ptr));
    uv_err_data { err_name: err_name, err_msg: err_msg }
}

pub struct uv_err_data {
    err_name: ~str,
    err_msg: ~str,
}

extern {

    fn rust_uv_handle_size(type_: uintptr_t) -> size_t;
    fn rust_uv_req_size(type_: uintptr_t) -> size_t;
    fn rust_uv_handle_type_max() -> uintptr_t;
    fn rust_uv_req_type_max() -> uintptr_t;

    // libuv public API
    fn rust_uv_loop_new() -> *c_void;
    fn rust_uv_loop_delete(lp: *c_void);
    fn rust_uv_run(loop_handle: *c_void);
    fn rust_uv_close(handle: *c_void, cb: *u8);
    fn rust_uv_walk(loop_handle: *c_void, cb: *u8, arg: *c_void);

    fn rust_uv_idle_new() -> *uv_idle_t;
    fn rust_uv_idle_delete(handle: *uv_idle_t);
    fn rust_uv_idle_init(loop_handle: *uv_loop_t, handle: *uv_idle_t) -> c_int;
    fn rust_uv_idle_start(handle: *uv_idle_t, cb: uv_idle_cb) -> c_int;
    fn rust_uv_idle_stop(handle: *uv_idle_t) -> c_int;

    fn rust_uv_async_send(handle: *uv_async_t);
    fn rust_uv_async_init(loop_handle: *c_void,
                          async_handle: *uv_async_t,
                          cb: *u8) -> c_int;
    fn rust_uv_tcp_init(loop_handle: *c_void, handle_ptr: *uv_tcp_t) -> c_int;
    // FIXME ref #2604 .. ?
    fn rust_uv_buf_init(out_buf: *uv_buf_t, base: *u8, len: size_t);
    fn rust_uv_last_error(loop_handle: *c_void) -> uv_err_t;
    // FIXME ref #2064
    fn rust_uv_strerror(err: *uv_err_t) -> *c_char;
    // FIXME ref #2064
    fn rust_uv_err_name(err: *uv_err_t) -> *c_char;
    fn rust_uv_ip4_addrp(ip: *u8, port: c_int) -> *sockaddr_in;
    fn rust_uv_ip6_addrp(ip: *u8, port: c_int) -> *sockaddr_in6;
    fn rust_uv_free_ip4_addr(addr: *sockaddr_in);
    fn rust_uv_free_ip6_addr(addr: *sockaddr_in6);
    fn rust_uv_ip4_name(src: *sockaddr_in, dst: *u8, size: size_t) -> c_int;
    fn rust_uv_ip6_name(src: *sockaddr_in6, dst: *u8, size: size_t) -> c_int;
    fn rust_uv_ip4_port(src: *sockaddr_in) -> c_uint;
    fn rust_uv_ip6_port(src: *sockaddr_in6) -> c_uint;
    // FIXME ref #2064
    fn rust_uv_tcp_connect(connect_ptr: *uv_connect_t,
                           tcp_handle_ptr: *uv_tcp_t,
                           after_cb: *u8,
                           addr: *sockaddr_in) -> c_int;
    // FIXME ref #2064
    fn rust_uv_tcp_bind(tcp_server: *uv_tcp_t, addr: *sockaddr_in) -> c_int;
    // FIXME ref #2064
    fn rust_uv_tcp_connect6(connect_ptr: *uv_connect_t,
                            tcp_handle_ptr: *uv_tcp_t,
                            after_cb: *u8,
                            addr: *sockaddr_in6) -> c_int;
    // FIXME ref #2064
    fn rust_uv_tcp_bind6(tcp_server: *uv_tcp_t, addr: *sockaddr_in6) -> c_int;
    fn rust_uv_tcp_getpeername(tcp_handle_ptr: *uv_tcp_t,
                               name: *sockaddr_in) -> c_int;
    fn rust_uv_tcp_getpeername6(tcp_handle_ptr: *uv_tcp_t,
                                name: *sockaddr_in6) ->c_int;
    fn rust_uv_listen(stream: *c_void, backlog: c_int, cb: *u8) -> c_int;
    fn rust_uv_accept(server: *c_void, client: *c_void) -> c_int;
    fn rust_uv_write(req: *c_void,
                     stream: *c_void,
                     buf_in: *uv_buf_t,
                     buf_cnt: c_int,
                     cb: *u8) -> c_int;
    fn rust_uv_read_start(stream: *c_void,
                          on_alloc: *u8,
                          on_read: *u8) -> c_int;
    fn rust_uv_read_stop(stream: *c_void) -> c_int;
    fn rust_uv_timer_init(loop_handle: *c_void,
                          timer_handle: *uv_timer_t) -> c_int;
    fn rust_uv_timer_start(timer_handle: *uv_timer_t,
                           cb: *u8,
                           timeout: c_uint,
                           repeat: c_uint) -> c_int;
    fn rust_uv_timer_stop(handle: *uv_timer_t) -> c_int;

    fn rust_uv_malloc_buf_base_of(sug_size: size_t) -> *u8;
    fn rust_uv_free_base_of_buf(buf: uv_buf_t);
    fn rust_uv_get_stream_handle_from_connect_req(connect_req: *uv_connect_t) -> *uv_stream_t;
    fn rust_uv_get_stream_handle_from_write_req(write_req: *uv_write_t) -> *uv_stream_t;
    fn rust_uv_get_loop_for_uv_handle(handle: *c_void) -> *c_void;
    fn rust_uv_get_data_for_uv_loop(loop_ptr: *c_void) -> *c_void;
    fn rust_uv_set_data_for_uv_loop(loop_ptr: *c_void, data: *c_void);
    fn rust_uv_get_data_for_uv_handle(handle: *c_void) -> *c_void;
    fn rust_uv_set_data_for_uv_handle(handle: *c_void, data: *c_void);
    fn rust_uv_get_data_for_req(req: *c_void) -> *c_void;
    fn rust_uv_set_data_for_req(req: *c_void, data: *c_void);
    fn rust_uv_get_base_from_buf(buf: uv_buf_t) -> *u8;
    fn rust_uv_get_len_from_buf(buf: uv_buf_t) -> size_t;
}
