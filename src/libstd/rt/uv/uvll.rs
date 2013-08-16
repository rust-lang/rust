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

use c_str::ToCStr;
use libc::{size_t, c_int, c_uint, c_void, c_char, uintptr_t};
use libc::{malloc, free};
use libc;
use prelude::*;
use ptr;
use str;
use vec;

pub static UNKNOWN: c_int = -1;
pub static OK: c_int = 0;
pub static EOF: c_int = 1;
pub static EADDRINFO: c_int = 2;
pub static EACCES: c_int = 3;
pub static ECONNREFUSED: c_int = 12;
pub static ECONNRESET: c_int = 13;
pub static EPIPE: c_int = 36;

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
pub type uv_udp_t = c_void;
pub type uv_connect_t = c_void;
pub type uv_write_t = c_void;
pub type uv_async_t = c_void;
pub type uv_timer_t = c_void;
pub type uv_stream_t = c_void;
pub type uv_fs_t = c_void;
pub type uv_udp_send_t = c_void;

pub type uv_idle_cb = *u8;
pub type uv_alloc_cb = *u8;
pub type uv_udp_send_cb = *u8;
pub type uv_udp_recv_cb = *u8;

pub type sockaddr = c_void;
pub type sockaddr_in = c_void;
pub type sockaddr_in6 = c_void;
pub type sockaddr_storage = c_void;

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

#[deriving(Eq)]
pub enum uv_membership {
    UV_LEAVE_GROUP,
    UV_JOIN_GROUP
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
        assert_eq!(UV_HANDLE_TYPE_MAX as uint, rust_uv_handle_type_max());
    }
}

#[test]
fn request_sanity_check() {
    unsafe {
        assert_eq!(UV_REQ_TYPE_MAX as uint, rust_uv_req_type_max());
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

pub unsafe fn udp_init(loop_handle: *uv_loop_t, handle: *uv_udp_t) -> c_int {
    return rust_uv_udp_init(loop_handle, handle);
}

pub unsafe fn udp_bind(server: *uv_udp_t, addr: *sockaddr_in, flags: c_uint) -> c_int {
    return rust_uv_udp_bind(server, addr, flags);
}

pub unsafe fn udp_bind6(server: *uv_udp_t, addr: *sockaddr_in6, flags: c_uint) -> c_int {
    return rust_uv_udp_bind6(server, addr, flags);
}

pub unsafe fn udp_send<T>(req: *uv_udp_send_t, handle: *T, buf_in: &[uv_buf_t],
                          addr: *sockaddr_in, cb: uv_udp_send_cb) -> c_int {
    let buf_ptr = vec::raw::to_ptr(buf_in);
    let buf_cnt = buf_in.len() as i32;
    return rust_uv_udp_send(req, handle as *c_void, buf_ptr, buf_cnt, addr, cb);
}

pub unsafe fn udp_send6<T>(req: *uv_udp_send_t, handle: *T, buf_in: &[uv_buf_t],
                          addr: *sockaddr_in6, cb: uv_udp_send_cb) -> c_int {
    let buf_ptr = vec::raw::to_ptr(buf_in);
    let buf_cnt = buf_in.len() as i32;
    return rust_uv_udp_send6(req, handle as *c_void, buf_ptr, buf_cnt, addr, cb);
}

pub unsafe fn udp_recv_start(server: *uv_udp_t, on_alloc: uv_alloc_cb,
                             on_recv: uv_udp_recv_cb) -> c_int {
    return rust_uv_udp_recv_start(server, on_alloc, on_recv);
}

pub unsafe fn udp_recv_stop(server: *uv_udp_t) -> c_int {
    return rust_uv_udp_recv_stop(server);
}

pub unsafe fn get_udp_handle_from_send_req(send_req: *uv_udp_send_t) -> *uv_udp_t {
    return rust_uv_get_udp_handle_from_send_req(send_req);
}

pub unsafe fn udp_get_sockname(handle: *uv_udp_t, name: *sockaddr_storage) -> c_int {
    return rust_uv_udp_getsockname(handle, name);
}

pub unsafe fn udp_set_membership(handle: *uv_udp_t, multicast_addr: *c_char,
                                 interface_addr: *c_char, membership: uv_membership) -> c_int {
    return rust_uv_udp_set_membership(handle, multicast_addr, interface_addr, membership as c_int);
}

pub unsafe fn udp_set_multicast_loop(handle: *uv_udp_t, on: c_int) -> c_int {
    return rust_uv_udp_set_multicast_loop(handle, on);
}

pub unsafe fn udp_set_multicast_ttl(handle: *uv_udp_t, ttl: c_int) -> c_int {
    return rust_uv_udp_set_multicast_ttl(handle, ttl);
}

pub unsafe fn udp_set_ttl(handle: *uv_udp_t, ttl: c_int) -> c_int {
    return rust_uv_udp_set_ttl(handle, ttl);
}

pub unsafe fn udp_set_broadcast(handle: *uv_udp_t, on: c_int) -> c_int {
    return rust_uv_udp_set_broadcast(handle, on);
}

pub unsafe fn tcp_init(loop_handle: *c_void, handle: *uv_tcp_t) -> c_int {
    return rust_uv_tcp_init(loop_handle, handle);
}

pub unsafe fn tcp_connect(connect_ptr: *uv_connect_t, tcp_handle_ptr: *uv_tcp_t,
                          addr_ptr: *sockaddr_in, after_connect_cb: *u8) -> c_int {
    return rust_uv_tcp_connect(connect_ptr, tcp_handle_ptr, after_connect_cb, addr_ptr);
}

pub unsafe fn tcp_connect6(connect_ptr: *uv_connect_t, tcp_handle_ptr: *uv_tcp_t,
                           addr_ptr: *sockaddr_in6, after_connect_cb: *u8) -> c_int {
    return rust_uv_tcp_connect6(connect_ptr, tcp_handle_ptr, after_connect_cb, addr_ptr);
}

pub unsafe fn tcp_bind(tcp_server_ptr: *uv_tcp_t, addr_ptr: *sockaddr_in) -> c_int {
    return rust_uv_tcp_bind(tcp_server_ptr, addr_ptr);
}

pub unsafe fn tcp_bind6(tcp_server_ptr: *uv_tcp_t, addr_ptr: *sockaddr_in6) -> c_int {
    return rust_uv_tcp_bind6(tcp_server_ptr, addr_ptr);
}

pub unsafe fn tcp_getpeername(tcp_handle_ptr: *uv_tcp_t, name: *sockaddr_storage) -> c_int {
    return rust_uv_tcp_getpeername(tcp_handle_ptr, name);
}

pub unsafe fn tcp_getsockname(handle: *uv_tcp_t, name: *sockaddr_storage) -> c_int {
    return rust_uv_tcp_getsockname(handle, name);
}

pub unsafe fn tcp_nodelay(handle: *uv_tcp_t, enable: c_int) -> c_int {
    return rust_uv_tcp_nodelay(handle, enable);
}

pub unsafe fn tcp_keepalive(handle: *uv_tcp_t, enable: c_int, delay: c_uint) -> c_int {
    return rust_uv_tcp_keepalive(handle, enable, delay);
}

pub unsafe fn tcp_simultaneous_accepts(handle: *uv_tcp_t, enable: c_int) -> c_int {
    return rust_uv_tcp_simultaneous_accepts(handle, enable);
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
pub unsafe fn read_start(stream: *uv_stream_t, on_alloc: uv_alloc_cb, on_read: *u8) -> c_int {
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
pub unsafe fn timer_start(timer_ptr: *uv_timer_t, cb: *u8, timeout: u64,
                          repeat: u64) -> c_int {
    return rust_uv_timer_start(timer_ptr, cb, timeout, repeat);
}
pub unsafe fn timer_stop(timer_ptr: *uv_timer_t) -> c_int {
    return rust_uv_timer_stop(timer_ptr);
}

pub unsafe fn is_ip4_addr(addr: *sockaddr) -> bool {
    match rust_uv_is_ipv4_sockaddr(addr) { 0 => false, _ => true }
}

pub unsafe fn is_ip6_addr(addr: *sockaddr) -> bool {
    match rust_uv_is_ipv6_sockaddr(addr) { 0 => false, _ => true }
}

pub unsafe fn malloc_ip4_addr(ip: &str, port: int) -> *sockaddr_in {
    do ip.with_c_str |ip_buf| {
        rust_uv_ip4_addrp(ip_buf as *u8, port as libc::c_int)
    }
}
pub unsafe fn malloc_ip6_addr(ip: &str, port: int) -> *sockaddr_in6 {
    do ip.with_c_str |ip_buf| {
        rust_uv_ip6_addrp(ip_buf as *u8, port as libc::c_int)
    }
}

pub unsafe fn malloc_sockaddr_storage() -> *sockaddr_storage {
    rust_uv_malloc_sockaddr_storage()
}

pub unsafe fn free_sockaddr_storage(ss: *sockaddr_storage) {
    rust_uv_free_sockaddr_storage(ss);
}

pub unsafe fn free_ip4_addr(addr: *sockaddr_in) {
    rust_uv_free_ip4_addr(addr);
}

pub unsafe fn free_ip6_addr(addr: *sockaddr_in6) {
    rust_uv_free_ip6_addr(addr);
}

pub unsafe fn ip4_name(addr: *sockaddr_in, dst: *u8, size: size_t) -> c_int {
    return rust_uv_ip4_name(addr, dst, size);
}

pub unsafe fn ip6_name(addr: *sockaddr_in6, dst: *u8, size: size_t) -> c_int {
    return rust_uv_ip6_name(addr, dst, size);
}

pub unsafe fn ip4_port(addr: *sockaddr_in) -> c_uint {
   return rust_uv_ip4_port(addr);
}

pub unsafe fn ip6_port(addr: *sockaddr_in6) -> c_uint {
    return rust_uv_ip6_port(addr);
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
    fn rust_uv_async_init(loop_handle: *c_void, async_handle: *uv_async_t, cb: *u8) -> c_int;
    fn rust_uv_tcp_init(loop_handle: *c_void, handle_ptr: *uv_tcp_t) -> c_int;
    fn rust_uv_buf_init(out_buf: *uv_buf_t, base: *u8, len: size_t);
    fn rust_uv_last_error(loop_handle: *c_void) -> uv_err_t;
    fn rust_uv_strerror(err: *uv_err_t) -> *c_char;
    fn rust_uv_err_name(err: *uv_err_t) -> *c_char;
    fn rust_uv_ip4_addrp(ip: *u8, port: c_int) -> *sockaddr_in;
    fn rust_uv_ip6_addrp(ip: *u8, port: c_int) -> *sockaddr_in6;
    fn rust_uv_free_ip4_addr(addr: *sockaddr_in);
    fn rust_uv_free_ip6_addr(addr: *sockaddr_in6);
    fn rust_uv_ip4_name(src: *sockaddr_in, dst: *u8, size: size_t) -> c_int;
    fn rust_uv_ip6_name(src: *sockaddr_in6, dst: *u8, size: size_t) -> c_int;
    fn rust_uv_ip4_port(src: *sockaddr_in) -> c_uint;
    fn rust_uv_ip6_port(src: *sockaddr_in6) -> c_uint;
    fn rust_uv_tcp_connect(req: *uv_connect_t, handle: *uv_tcp_t, cb: *u8,
                           addr: *sockaddr_in) -> c_int;
    fn rust_uv_tcp_bind(tcp_server: *uv_tcp_t, addr: *sockaddr_in) -> c_int;
    fn rust_uv_tcp_connect6(req: *uv_connect_t, handle: *uv_tcp_t, cb: *u8,
                            addr: *sockaddr_in6) -> c_int;
    fn rust_uv_tcp_bind6(tcp_server: *uv_tcp_t, addr: *sockaddr_in6) -> c_int;
    fn rust_uv_tcp_getpeername(tcp_handle_ptr: *uv_tcp_t, name: *sockaddr_storage) -> c_int;
    fn rust_uv_tcp_getsockname(handle: *uv_tcp_t, name: *sockaddr_storage) -> c_int;
    fn rust_uv_tcp_nodelay(handle: *uv_tcp_t, enable: c_int) -> c_int;
    fn rust_uv_tcp_keepalive(handle: *uv_tcp_t, enable: c_int, delay: c_uint) -> c_int;
    fn rust_uv_tcp_simultaneous_accepts(handle: *uv_tcp_t, enable: c_int) -> c_int;

    fn rust_uv_udp_init(loop_handle: *uv_loop_t, handle_ptr: *uv_udp_t) -> c_int;
    fn rust_uv_udp_bind(server: *uv_udp_t, addr: *sockaddr_in, flags: c_uint) -> c_int;
    fn rust_uv_udp_bind6(server: *uv_udp_t, addr: *sockaddr_in6, flags: c_uint) -> c_int;
    fn rust_uv_udp_send(req: *uv_udp_send_t, handle: *uv_udp_t, buf_in: *uv_buf_t,
                        buf_cnt: c_int, addr: *sockaddr_in, cb: *u8) -> c_int;
    fn rust_uv_udp_send6(req: *uv_udp_send_t, handle: *uv_udp_t, buf_in: *uv_buf_t,
                         buf_cnt: c_int, addr: *sockaddr_in6, cb: *u8) -> c_int;
    fn rust_uv_udp_recv_start(server: *uv_udp_t, on_alloc: *u8, on_recv: *u8) -> c_int;
    fn rust_uv_udp_recv_stop(server: *uv_udp_t) -> c_int;
    fn rust_uv_get_udp_handle_from_send_req(req: *uv_udp_send_t) -> *uv_udp_t;
    fn rust_uv_udp_getsockname(handle: *uv_udp_t, name: *sockaddr_storage) -> c_int;
    fn rust_uv_udp_set_membership(handle: *uv_udp_t, multicast_addr: *c_char,
                                  interface_addr: *c_char, membership: c_int) -> c_int;
    fn rust_uv_udp_set_multicast_loop(handle: *uv_udp_t, on: c_int) -> c_int;
    fn rust_uv_udp_set_multicast_ttl(handle: *uv_udp_t, ttl: c_int) -> c_int;
    fn rust_uv_udp_set_ttl(handle: *uv_udp_t, ttl: c_int) -> c_int;
    fn rust_uv_udp_set_broadcast(handle: *uv_udp_t, on: c_int) -> c_int;

    fn rust_uv_is_ipv4_sockaddr(addr: *sockaddr) -> c_int;
    fn rust_uv_is_ipv6_sockaddr(addr: *sockaddr) -> c_int;
    fn rust_uv_malloc_sockaddr_storage() -> *sockaddr_storage;
    fn rust_uv_free_sockaddr_storage(ss: *sockaddr_storage);

    fn rust_uv_listen(stream: *c_void, backlog: c_int, cb: *u8) -> c_int;
    fn rust_uv_accept(server: *c_void, client: *c_void) -> c_int;
    fn rust_uv_write(req: *c_void, stream: *c_void, buf_in: *uv_buf_t, buf_cnt: c_int,
                     cb: *u8) -> c_int;
    fn rust_uv_read_start(stream: *c_void, on_alloc: *u8, on_read: *u8) -> c_int;
    fn rust_uv_read_stop(stream: *c_void) -> c_int;
    fn rust_uv_timer_init(loop_handle: *c_void, timer_handle: *uv_timer_t) -> c_int;
    fn rust_uv_timer_start(timer_handle: *uv_timer_t, cb: *u8, timeout: libc::uint64_t,
                           repeat: libc::uint64_t) -> c_int;
    fn rust_uv_timer_stop(handle: *uv_timer_t) -> c_int;

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
