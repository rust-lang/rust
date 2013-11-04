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

#ifndef __WIN32__
// for signal
#include <signal.h>
#endif

#include "uv.h"

#include "rust_globals.h"

extern "C" void*
rust_uv_loop_new() {
// XXX libuv doesn't always ignore SIGPIPE even though we don't need it.
#ifndef __WIN32__
    signal(SIGPIPE, SIG_IGN);
#endif
    return (void*)uv_loop_new();
}

extern "C" void
rust_uv_loop_set_data(uv_loop_t* loop, void* data) {
    loop->data = data;
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
(uv_tcp_t* handle, sockaddr_storage* name) {
    // sockaddr_storage is big enough to hold either
    // sockaddr_in or sockaddr_in6
    int namelen = sizeof(sockaddr_in);
    return uv_tcp_getpeername(handle, (sockaddr*)name, &namelen);
}

extern "C" int
rust_uv_tcp_getsockname
(uv_tcp_t* handle, sockaddr_storage* name) {
    // sockaddr_storage is big enough to hold either
    // sockaddr_in or sockaddr_in6
    int namelen = sizeof(sockaddr_storage);
    return uv_tcp_getsockname(handle, (sockaddr*)name, &namelen);
}

extern "C" int
rust_uv_udp_bind(uv_udp_t* server, sockaddr_in* addr_ptr, unsigned flags) {
    return uv_udp_bind(server, *addr_ptr, flags);
}

extern "C" int
rust_uv_udp_bind6(uv_udp_t* server, sockaddr_in6* addr_ptr, unsigned flags) {
    return uv_udp_bind6(server, *addr_ptr, flags);
}

extern "C" int
rust_uv_udp_send(uv_udp_send_t* req, uv_udp_t* handle, uv_buf_t* buf_in,
                 int buf_cnt, sockaddr_in* addr_ptr, uv_udp_send_cb cb) {
    return uv_udp_send(req, handle, buf_in, buf_cnt, *addr_ptr, cb);
}

extern "C" int
rust_uv_udp_send6(uv_udp_send_t* req, uv_udp_t* handle, uv_buf_t* buf_in,
                  int buf_cnt, sockaddr_in6* addr_ptr, uv_udp_send_cb cb) {
    return uv_udp_send6(req, handle, buf_in, buf_cnt, *addr_ptr, cb);
}

extern "C" uv_udp_t*
rust_uv_get_udp_handle_from_send_req(uv_udp_send_t* send_req) {
    return send_req->handle;
}

extern "C" int
rust_uv_udp_getsockname
(uv_udp_t* handle, sockaddr_storage* name) {
    // sockaddr_storage is big enough to hold either
    // sockaddr_in or sockaddr_in6
    int namelen = sizeof(sockaddr_storage);
    return uv_udp_getsockname(handle, (sockaddr*)name, &namelen);
}

extern "C" uv_stream_t*
rust_uv_get_stream_handle_from_connect_req(uv_connect_t* connect) {
    return connect->handle;
}
extern "C" uv_stream_t*
rust_uv_get_stream_handle_from_write_req(uv_write_t* write_req) {
    return write_req->handle;
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

extern "C" struct sockaddr_storage *
rust_uv_malloc_sockaddr_storage() {
    struct sockaddr_storage *ss = (sockaddr_storage *)malloc(sizeof(struct sockaddr_storage));
    return ss;
}

extern "C" void
rust_uv_free_sockaddr_storage(struct sockaddr_storage *ss) {
    free(ss);
}

extern "C" void
rust_uv_free_ip4_addr(sockaddr_in *addrp) {
  free(addrp);
}

extern "C" void
rust_uv_free_ip6_addr(sockaddr_in6 *addrp) {
  free(addrp);
}
extern "C" unsigned int
rust_uv_ip4_port(struct sockaddr_in* src) {
    return ntohs(src->sin_port);
}
extern "C" unsigned int
rust_uv_ip6_port(struct sockaddr_in6* src) {
    return ntohs(src->sin6_port);
}

extern "C" int
rust_uv_is_ipv4_sockaddr(sockaddr* addr) {
    return addr->sa_family == AF_INET;
}

extern "C" int
rust_uv_is_ipv6_sockaddr(sockaddr* addr) {
    return addr->sa_family == AF_INET6;
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

extern "C" uintptr_t
rust_uv_handle_type_max() {
  return UV_HANDLE_TYPE_MAX;
}

extern "C" uintptr_t
rust_uv_req_type_max() {
  return UV_REQ_TYPE_MAX;
}

extern "C" int
rust_uv_get_result_from_fs_req(uv_fs_t* req) {
  return req->result;
}
extern "C" void*
rust_uv_get_ptr_from_fs_req(uv_fs_t* req) {
  return req->ptr;
}
extern "C" uv_loop_t*
rust_uv_get_loop_from_fs_req(uv_fs_t* req) {
  return req->loop;
}

extern "C" uv_loop_t*
rust_uv_get_loop_from_getaddrinfo_req(uv_getaddrinfo_t* req) {
  return req->loop;
}

extern "C" void
rust_uv_populate_uv_stat(uv_fs_t* req_in, uv_stat_t* stat_out) {
  stat_out->st_dev = req_in->statbuf.st_dev;
  stat_out->st_mode = req_in->statbuf.st_mode;
  stat_out->st_nlink = req_in->statbuf.st_nlink;
  stat_out->st_uid = req_in->statbuf.st_uid;
  stat_out->st_gid = req_in->statbuf.st_gid;
  stat_out->st_rdev = req_in->statbuf.st_rdev;
  stat_out->st_ino = req_in->statbuf.st_ino;
  stat_out->st_size = req_in->statbuf.st_size;
  stat_out->st_blksize = req_in->statbuf.st_blksize;
  stat_out->st_blocks = req_in->statbuf.st_blocks;
  stat_out->st_flags = req_in->statbuf.st_flags;
  stat_out->st_gen = req_in->statbuf.st_gen;
  stat_out->st_atim.tv_sec = req_in->statbuf.st_atim.tv_sec;
  stat_out->st_atim.tv_nsec = req_in->statbuf.st_atim.tv_nsec;
  stat_out->st_mtim.tv_sec = req_in->statbuf.st_mtim.tv_sec;
  stat_out->st_mtim.tv_nsec = req_in->statbuf.st_mtim.tv_nsec;
  stat_out->st_ctim.tv_sec = req_in->statbuf.st_ctim.tv_sec;
  stat_out->st_ctim.tv_nsec = req_in->statbuf.st_ctim.tv_nsec;
  stat_out->st_birthtim.tv_sec = req_in->statbuf.st_birthtim.tv_sec;
  stat_out->st_birthtim.tv_nsec = req_in->statbuf.st_birthtim.tv_nsec;
}

extern "C" void
rust_set_stdio_container_flags(uv_stdio_container_t *c, int flags) {
  c->flags = (uv_stdio_flags) flags;
}

extern "C" void
rust_set_stdio_container_fd(uv_stdio_container_t *c, int fd) {
  c->data.fd = fd;
}

extern "C" void
rust_set_stdio_container_stream(uv_stdio_container_t *c, uv_stream_t *stream) {
  c->data.stream = stream;
}

extern "C" int
rust_uv_process_pid(uv_process_t* p) {
  return p->pid;
}
