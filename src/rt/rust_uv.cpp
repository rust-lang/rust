// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifdef __WIN32__
// For alloca
#include <malloc.h>
#endif

#include "rust_globals.h"
#include "rust_task.h"
#include "rust_log.h"
#include "uv.h"

// extern fn pointers
typedef void (*extern_async_op_cb)(uv_loop_t* loop, void* data,
        uv_async_t* op_handle);
typedef void (*extern_simple_cb)(uint8_t* id_buf, void* loop_data);
typedef void (*extern_close_cb)(uint8_t* id_buf, void* handle,
        void* data);

// data types
#define RUST_UV_HANDLE_LEN 16

struct handle_data {
    uint8_t id_buf[RUST_UV_HANDLE_LEN];
    extern_simple_cb cb;
    extern_close_cb close_cb;
};

// helpers
static void*
current_kernel_malloc(size_t size, const char* tag) {
  void* ptr = rust_get_current_task()->kernel->malloc(size, tag);
  return ptr;
}

static void
current_kernel_free(void* ptr) {
  rust_get_current_task()->kernel->free(ptr);
}

static handle_data*
new_handle_data_from(uint8_t* buf, extern_simple_cb cb) {
    handle_data* data = (handle_data*)current_kernel_malloc(
            sizeof(handle_data),
            "handle_data");
    memcpy(data->id_buf, buf, RUST_UV_HANDLE_LEN);
    data->cb = cb;
    return data;
}

// libuv callback impls
static void
foreign_extern_async_op_cb(uv_async_t* handle, int status) {
    extern_async_op_cb cb = (extern_async_op_cb)handle->data;
    void* loop_data = handle->loop->data;
    cb(handle->loop, loop_data, handle);
}

static void
foreign_async_cb(uv_async_t* handle, int status) {
    handle_data* handle_d = (handle_data*)handle->data;
    void* loop_data = handle->loop->data;
    handle_d->cb(handle_d->id_buf, loop_data);
}

static void
foreign_timer_cb(uv_timer_t* handle, int status) {
    handle_data* handle_d = (handle_data*)handle->data;
    void* loop_data = handle->loop->data;
    handle_d->cb(handle_d->id_buf, loop_data);
}

static void
foreign_close_cb(uv_handle_t* handle) {
    handle_data* data = (handle_data*)handle->data;
    data->close_cb(data->id_buf, handle, handle->loop->data);
}

static void
foreign_close_op_cb(uv_handle_t* op_handle) {
    current_kernel_free(op_handle);
    // uv_run() should return after this..
}

// foreign fns bound in rust
extern "C" void
rust_uv_free(void* ptr) {
    current_kernel_free(ptr);
}
extern "C" void*
rust_uv_loop_new() {
    return (void*)uv_loop_new();
}

extern "C" void
rust_uv_loop_delete(uv_loop_t* loop) {
    // FIXME: This is a workaround for #1815. libev uses realloc(0) to
    // free the loop, which valgrind doesn't like. We have suppressions
    // to make valgrind ignore them.
    //
    // Valgrind also has a sanity check when collecting allocation backtraces
    // that the stack pointer must be at least 512 bytes into the stack (at
    // least 512 bytes of frames must have come before). When this is not
    // the case it doesn't collect the backtrace.
    //
    // Unfortunately, with our spaghetti stacks that valgrind check triggers
    // sometimes and we don't get the backtrace for the realloc(0), it
    // fails to be suppressed, and it gets reported as 0 bytes lost
    // from a malloc with no backtrace.
    //
    // This pads our stack with some extra space before deleting the loop
    alloca(512);
    uv_loop_delete(loop);
}

extern "C" void
rust_uv_loop_set_data(uv_loop_t* loop, void* data) {
    loop->data = data;
}

extern "C" void*
rust_uv_bind_op_cb(uv_loop_t* loop, extern_async_op_cb cb) {
    uv_async_t* async = (uv_async_t*)current_kernel_malloc(
            sizeof(uv_async_t),
            "uv_async_t");
    uv_async_init(loop, async, foreign_extern_async_op_cb);
    async->data = (void*)cb;
    // decrement the ref count, so that our async bind
    // doesn't count towards keeping the loop alive
    //uv_unref(loop);
    return async;
}

extern "C" void
rust_uv_stop_op_cb(uv_handle_t* op_handle) {
    uv_close(op_handle, foreign_close_op_cb);
}

extern "C" void
rust_uv_run(uv_loop_t* loop) {
    uv_run(loop, UV_RUN_DEFAULT);
}

extern "C" void
rust_uv_close(uv_handle_t* handle, uv_close_cb cb) {
    uv_close(handle, cb);
}

extern "C" void
rust_uv_walk(uv_loop_t* loop, uv_walk_cb cb, void* arg) {
    uv_walk(loop, cb, arg);
}

extern "C" void
rust_uv_hilvl_close(uv_handle_t* handle, extern_close_cb cb) {
    handle_data* data = (handle_data*)handle->data;
    data->close_cb = cb;
    uv_close(handle, foreign_close_cb);
}

extern "C" void
rust_uv_hilvl_close_async(uv_async_t* handle) {
    current_kernel_free(handle->data);
    current_kernel_free(handle);
}

extern "C" void
rust_uv_hilvl_close_timer(uv_async_t* handle) {
    current_kernel_free(handle->data);
    current_kernel_free(handle);
}

extern "C" void
rust_uv_async_send(uv_async_t* handle) {
    uv_async_send(handle);
}

extern "C" int
rust_uv_async_init(uv_loop_t* loop_handle,
        uv_async_t* async_handle,
        uv_async_cb cb) {
    return uv_async_init(loop_handle, async_handle, cb);
}

extern "C" void*
rust_uv_hilvl_async_init(uv_loop_t* loop, extern_simple_cb cb,
        uint8_t* buf) {
    uv_async_t* async = (uv_async_t*)current_kernel_malloc(
            sizeof(uv_async_t),
            "uv_async_t");
    uv_async_init(loop, async, foreign_async_cb);
    handle_data* data = new_handle_data_from(buf, cb);
    async->data = data;

    return async;
}

extern "C" void*
rust_uv_hilvl_timer_init(uv_loop_t* loop, extern_simple_cb cb,
        uint8_t* buf) {
    uv_timer_t* new_timer = (uv_timer_t*)current_kernel_malloc(
            sizeof(uv_timer_t),
            "uv_timer_t");
    uv_timer_init(loop, new_timer);
    handle_data* data = new_handle_data_from(buf, cb);
    new_timer->data = data;

    return new_timer;
}

extern "C" void
rust_uv_hilvl_timer_start(uv_timer_t* the_timer, uint32_t timeout,
        uint32_t repeat) {
    uv_timer_start(the_timer, foreign_timer_cb, timeout, repeat);
}

extern "C" int
rust_uv_timer_init(uv_loop_t* loop, uv_timer_t* timer) {
    return uv_timer_init(loop, timer);
}

extern "C" int
rust_uv_timer_start(uv_timer_t* the_timer, uv_timer_cb cb,
                    int64_t timeout, int64_t repeat) {
    return uv_timer_start(the_timer, cb, timeout, repeat);
}

extern "C" int
rust_uv_timer_stop(uv_timer_t* the_timer) {
    return uv_timer_stop(the_timer);
}

extern "C" int
rust_uv_tcp_init(uv_loop_t* loop, uv_tcp_t* handle) {
    return uv_tcp_init(loop, handle);
}

extern "C" int
rust_uv_tcp_connect(uv_connect_t* connect_ptr,
        uv_tcp_t* tcp_ptr,
        uv_connect_cb cb,
        sockaddr_in* addr_ptr) {
    // FIXME ref #2064
    sockaddr_in addr = *addr_ptr;
    int result = uv_tcp_connect(connect_ptr, tcp_ptr, addr, cb);
    return result;
}

extern "C" int
rust_uv_tcp_bind(uv_tcp_t* tcp_server, sockaddr_in* addr_ptr) {
    // FIXME ref #2064
    sockaddr_in addr = *addr_ptr;
    return uv_tcp_bind(tcp_server, addr);
}
extern "C" int
rust_uv_tcp_connect6(uv_connect_t* connect_ptr,
        uv_tcp_t* tcp_ptr,
        uv_connect_cb cb,
        sockaddr_in6* addr_ptr) {
    // FIXME ref #2064
    sockaddr_in6 addr = *addr_ptr;
    int result = uv_tcp_connect6(connect_ptr, tcp_ptr, addr, cb);
    return result;
}

extern "C" int
rust_uv_tcp_bind6
(uv_tcp_t* tcp_server, sockaddr_in6* addr_ptr) {
    // FIXME ref #2064
    sockaddr_in6 addr = *addr_ptr;
    return uv_tcp_bind6(tcp_server, addr);
}

extern "C" int
rust_uv_tcp_getpeername
(uv_tcp_t* handle, sockaddr_in* name) {
    int namelen = sizeof(sockaddr_in);
    return uv_tcp_getpeername(handle, (sockaddr*)name, &namelen);
}

extern "C" int
rust_uv_tcp_getpeername6
(uv_tcp_t* handle, sockaddr_in6* name) {
    int namelen = sizeof(sockaddr_in6);
    return uv_tcp_getpeername(handle, (sockaddr*)name, &namelen);
}

extern "C" int
rust_uv_listen(uv_stream_t* stream, int backlog,
        uv_connection_cb cb) {
    return uv_listen(stream, backlog, cb);
}

extern "C" int
rust_uv_accept(uv_stream_t* server, uv_stream_t* client) {
    return uv_accept(server, client);
}

extern "C" size_t
rust_uv_helper_uv_tcp_t_size() {
    return sizeof(uv_tcp_t);
}
extern "C" size_t
rust_uv_helper_uv_connect_t_size() {
    return sizeof(uv_connect_t);
}
extern "C" size_t
rust_uv_helper_uv_buf_t_size() {
    return sizeof(uv_buf_t);
}
extern "C" size_t
rust_uv_helper_uv_write_t_size() {
    return sizeof(uv_write_t);
}
extern "C" size_t
rust_uv_helper_uv_err_t_size() {
    return sizeof(uv_err_t);
}
extern "C" size_t
rust_uv_helper_sockaddr_in_size() {
    return sizeof(sockaddr_in);
}
extern "C" size_t
rust_uv_helper_sockaddr_in6_size() {
    return sizeof(sockaddr_in6);
}
extern "C" size_t
rust_uv_helper_uv_async_t_size() {
    return sizeof(uv_async_t);
}
extern "C" size_t
rust_uv_helper_uv_timer_t_size() {
    return sizeof(uv_timer_t);
}
extern "C" size_t
rust_uv_helper_addr_in_size() {
    return sizeof(sockaddr_in6);
}
extern "C" size_t
rust_uv_helper_uv_getaddrinfo_t_size() {
    return sizeof(uv_getaddrinfo_t);
}
extern "C" size_t
rust_uv_helper_addrinfo_size() {
    return sizeof(addrinfo);
}
extern "C" unsigned int
rust_uv_helper_get_INADDR_NONE() {
    return INADDR_NONE;
}
extern "C" uv_stream_t*
rust_uv_get_stream_handle_from_connect_req(uv_connect_t* connect) {
    return connect->handle;
}
extern "C" uv_stream_t*
rust_uv_get_stream_handle_from_write_req(uv_write_t* write_req) {
    return write_req->handle;
}

extern "C" uv_buf_t
current_kernel_malloc_alloc_cb(uv_handle_t* handle,
        size_t suggested_size) {
    char* base_ptr = (char*)current_kernel_malloc(sizeof(char)
            * suggested_size,
            "uv_buf_t_base_val");
    return uv_buf_init(base_ptr, suggested_size);
}

extern "C" void
rust_uv_buf_init(uv_buf_t* out_buf, char* base, size_t len) {
    *out_buf = uv_buf_init(base, len);
}

extern "C" uv_loop_t*
rust_uv_get_loop_for_uv_handle(uv_handle_t* handle) {
    return handle->loop;
}

extern "C" void*
rust_uv_get_data_for_uv_loop(uv_loop_t* loop) {
    return loop->data;
}

extern "C" void
rust_uv_set_data_for_uv_loop(uv_loop_t* loop,
        void* data) {
    loop->data = data;
}

extern "C" void*
rust_uv_get_data_for_uv_handle(uv_handle_t* handle) {
    return handle->data;
}

extern "C" void
rust_uv_set_data_for_uv_handle(uv_handle_t* handle, void* data) {
    handle->data = data;
}

extern "C" void*
rust_uv_get_data_for_req(uv_req_t* req) {
    return req->data;
}

extern "C" void
rust_uv_set_data_for_req(uv_req_t* req, void* data) {
    req->data = data;
}

extern "C" char*
rust_uv_get_base_from_buf(uv_buf_t buf) {
    return buf.base;
}

extern "C" size_t
rust_uv_get_len_from_buf(uv_buf_t buf) {
    return buf.len;
}

extern "C" uv_err_t
rust_uv_last_error(uv_loop_t* loop) {
    return uv_last_error(loop);
}

extern "C" const char*
rust_uv_strerror(uv_err_t* err_ptr) {
    uv_err_t err = *err_ptr;
    return uv_strerror(err);
}

extern "C" const char*
rust_uv_err_name(uv_err_t* err_ptr) {
    uv_err_t err = *err_ptr;
    return uv_err_name(err);
}

extern "C" int
rust_uv_write(uv_write_t* req, uv_stream_t* handle,
        uv_buf_t* bufs, int buf_cnt,
        uv_write_cb cb) {
    return uv_write(req, handle, bufs, buf_cnt, cb);
}
extern "C" int
rust_uv_read_start(uv_stream_t* stream, uv_alloc_cb on_alloc,
        uv_read_cb on_read) {
    return uv_read_start(stream, on_alloc, on_read);
}

extern "C" int
rust_uv_read_stop(uv_stream_t* stream) {
    return uv_read_stop(stream);
}

extern "C" char*
rust_uv_malloc_buf_base_of(size_t suggested_size) {
    return (char*) current_kernel_malloc(sizeof(char)*suggested_size,
            "uv_buf_t base");
}
extern "C" void
rust_uv_free_base_of_buf(uv_buf_t buf) {
    current_kernel_free(buf.base);
}

extern "C" struct sockaddr_in
rust_uv_ip4_addr(const char* ip, int port) {
    struct sockaddr_in addr = uv_ip4_addr(ip, port);
    return addr;
}
extern "C" struct sockaddr_in6
rust_uv_ip6_addr(const char* ip, int port) {
    return uv_ip6_addr(ip, port);
}

extern "C" struct sockaddr_in*
rust_uv_ip4_addrp(const char* ip, int port) {
  struct sockaddr_in addr = uv_ip4_addr(ip, port);
  struct sockaddr_in *addrp = (sockaddr_in*)malloc(sizeof(struct sockaddr_in));
  assert(addrp);
  memcpy(addrp, &addr, sizeof(struct sockaddr_in));
  return addrp;
}
extern "C" struct sockaddr_in6*
rust_uv_ip6_addrp(const char* ip, int port) {
  struct sockaddr_in6 addr = uv_ip6_addr(ip, port);
  struct sockaddr_in6 *addrp = (sockaddr_in6*)malloc(sizeof(struct sockaddr_in6));
  assert(addrp);
  memcpy(addrp, &addr, sizeof(struct sockaddr_in6));
  return addrp;
}

extern "C" void
rust_uv_free_ip4_addr(sockaddr_in *addrp) {
  free(addrp);
}

extern "C" void
rust_uv_free_ip6_addr(sockaddr_in6 *addrp) {
  free(addrp);
}

extern "C" int
rust_uv_ip4_name(struct sockaddr_in* src, char* dst, size_t size) {
    return uv_ip4_name(src, dst, size);
}
extern "C" int
rust_uv_ip6_name(struct sockaddr_in6* src, char* dst, size_t size) {
    int result = uv_ip6_name(src, dst, size);
    return result;
}
extern "C" unsigned int
rust_uv_ip4_port(struct sockaddr_in* src) {
    return ntohs(src->sin_port);
}
extern "C" unsigned int
rust_uv_ip6_port(struct sockaddr_in6* src) {
    return ntohs(src->sin6_port);
}

extern "C" void*
rust_uv_current_kernel_malloc(size_t size) {
    return current_kernel_malloc(size, "rust_uv_current_kernel_malloc");
}

extern "C" void
rust_uv_current_kernel_free(void* mem) {
    current_kernel_free(mem);
}

extern  "C" int
rust_uv_getaddrinfo(uv_loop_t* loop, uv_getaddrinfo_t* handle,
                    uv_getaddrinfo_cb cb,
                    char* node, char* service,
                    addrinfo* hints) {
    return uv_getaddrinfo(loop, handle, cb, node, service, hints);
}
extern "C" void
rust_uv_freeaddrinfo(addrinfo* res) {
    uv_freeaddrinfo(res);
}
extern "C" bool
rust_uv_is_ipv4_addrinfo(addrinfo* input) {
    return input->ai_family == AF_INET;
}
extern "C" bool
rust_uv_is_ipv6_addrinfo(addrinfo* input) {
    return input->ai_family == AF_INET6;
}
extern "C" addrinfo*
rust_uv_get_next_addrinfo(addrinfo* input) {
    return input->ai_next;
}
extern "C" sockaddr_in*
rust_uv_addrinfo_as_sockaddr_in(addrinfo* input) {
    return (sockaddr_in*)input->ai_addr;
}
extern "C" sockaddr_in6*
rust_uv_addrinfo_as_sockaddr_in6(addrinfo* input) {
    return (sockaddr_in6*)input->ai_addr;
}

extern "C" uv_idle_t*
rust_uv_idle_new() {
  return new uv_idle_t;
}

extern "C" void
rust_uv_idle_delete(uv_idle_t* handle) {
  delete handle;
}

extern "C" int
rust_uv_idle_init(uv_loop_t* loop, uv_idle_t* idle) {
  return uv_idle_init(loop, idle);
}

extern "C" int
rust_uv_idle_start(uv_idle_t* idle, uv_idle_cb cb) {
  return uv_idle_start(idle, cb);
}

extern "C" int
rust_uv_idle_stop(uv_idle_t* idle) {
  return uv_idle_stop(idle);
}

extern "C" size_t
rust_uv_handle_size(uintptr_t type) {
  return uv_handle_size((uv_handle_type)type);
}

extern "C" size_t
rust_uv_req_size(uintptr_t type) {
  return uv_req_size((uv_req_type)type);
}

extern "C" uintptr_t
rust_uv_handle_type_max() {
  return UV_HANDLE_TYPE_MAX;
}

extern "C" uintptr_t
rust_uv_req_type_max() {
  return UV_REQ_TYPE_MAX;
}
