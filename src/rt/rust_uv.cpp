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

extern "C" uv_udp_t*
rust_uv_get_udp_handle_from_send_req(uv_udp_send_t* send_req) {
    return send_req->handle;
}

extern "C" uv_stream_t*
rust_uv_get_stream_handle_from_connect_req(uv_connect_t* connect) {
    return connect->handle;
}
extern "C" uv_stream_t*
rust_uv_get_stream_handle_from_write_req(uv_write_t* write_req) {
    return write_req->handle;
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

extern "C" int
rust_sockaddr_size() {
    return sizeof(struct sockaddr_storage);
}

extern "C" struct sockaddr*
rust_malloc_ip4_addr(char *name, int port) {
    struct sockaddr_in *addr = (struct sockaddr_in*) calloc(1, rust_sockaddr_size());
    assert(addr != NULL);
    addr->sin_port = htons(port);
    assert(uv_inet_pton(AF_INET, name, &addr->sin_addr) == 0);
    addr->sin_family = AF_INET;
    return (struct sockaddr*) addr;
}

extern "C" struct sockaddr*
rust_malloc_ip6_addr(char *name, int port) {
    struct sockaddr_in6 *addr = (struct sockaddr_in6*) calloc(1, rust_sockaddr_size());
    assert(addr != NULL);
    addr->sin6_port = htons(port);
    assert(uv_inet_pton(AF_INET6, name, &addr->sin6_addr) == 0);
    addr->sin6_family = AF_INET6;
    return (struct sockaddr*) addr;
}

extern "C" unsigned int
rust_ip4_port(struct sockaddr_in* src) {
    return ntohs(src->sin_port);
}
extern "C" unsigned int
rust_ip6_port(struct sockaddr_in6* src) {
    return ntohs(src->sin6_port);
}

extern "C" int
rust_is_ipv4_sockaddr(sockaddr* addr) {
    return addr->sa_family == AF_INET;
}

extern "C" int
rust_is_ipv6_sockaddr(sockaddr* addr) {
    return addr->sa_family == AF_INET6;
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
extern "C" const char*
rust_uv_get_path_from_fs_req(uv_fs_t* req) {
  return req->path;
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

extern "C" int
rust_uv_guess_handle(int fd) {
  return uv_guess_handle(fd);
}
