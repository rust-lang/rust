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
rust_uv_async_send(uv_async_t* handle) {
    uv_async_send(handle);
}

extern "C" int
rust_uv_async_init(uv_loop_t* loop_handle,
        uv_async_t* async_handle,
        uv_async_cb cb) {
    return uv_async_init(loop_handle, async_handle, cb);
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
rust_uv_tcp_nodelay
(uv_tcp_t* handle, int enable) {
    return uv_tcp_nodelay(handle, enable);
}

extern "C" int
rust_uv_tcp_keepalive
(uv_tcp_t* handle, int enable, unsigned int delay) {
    return uv_tcp_keepalive(handle, enable, delay);
}

extern "C" int
rust_uv_tcp_simultaneous_accepts
(uv_tcp_t* handle, int enable) {
    return uv_tcp_simultaneous_accepts(handle, enable);
}

extern "C" int
rust_uv_udp_init(uv_loop_t* loop, uv_udp_t* handle) {
    return uv_udp_init(loop, handle);
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

extern "C" int
rust_uv_udp_recv_start(uv_udp_t* server, uv_alloc_cb on_alloc, uv_udp_recv_cb on_read) {
    return uv_udp_recv_start(server, on_alloc, on_read);
}

extern "C" int
rust_uv_udp_recv_stop(uv_udp_t* server) {
    return uv_udp_recv_stop(server);
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

extern "C" int
rust_uv_udp_set_membership
(uv_udp_t* handle, const char* m_addr, const char* i_addr, uv_membership membership) {
    return uv_udp_set_membership(handle, m_addr, i_addr, membership);
}

extern "C" int
rust_uv_udp_set_multicast_loop
(uv_udp_t* handle, int on) {
    return uv_udp_set_multicast_loop(handle, on);
}

extern "C" int
rust_uv_udp_set_multicast_ttl
(uv_udp_t* handle, int ttl) {
    return uv_udp_set_multicast_ttl(handle, ttl);
}

extern "C" int
rust_uv_udp_set_ttl
(uv_udp_t* handle, int ttl) {
    return uv_udp_set_ttl(handle, ttl);
}

extern "C" int
rust_uv_udp_set_broadcast
(uv_udp_t* handle, int on) {
    return uv_udp_set_broadcast(handle, on);
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

extern "C" char*
rust_uv_get_base_from_buf(uv_buf_t buf) {
    return buf.base;
}

extern "C" size_t
rust_uv_get_len_from_buf(uv_buf_t buf) {
    return buf.len;
}

extern "C" const char*
rust_uv_strerror(int err) {
    return uv_strerror(err);
}

extern "C" const char*
rust_uv_err_name(int err) {
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

extern "C" int
rust_uv_fs_open(uv_loop_t* loop, uv_fs_t* req, const char* path, int flags,
                int mode, uv_fs_cb cb) {
  return uv_fs_open(loop, req, path, flags, mode, cb);
}
extern "C" int
rust_uv_fs_unlink(uv_loop_t* loop, uv_fs_t* req, const char* path, uv_fs_cb cb) {
  return uv_fs_unlink(loop, req, path, cb);
}
extern "C" int
rust_uv_fs_write(uv_loop_t* loop, uv_fs_t* req, uv_file fd, void* buf,
                 size_t len, int64_t offset, uv_fs_cb cb) {
  return uv_fs_write(loop, req, fd, buf, len, offset, cb);
}
extern "C" int
rust_uv_fs_read(uv_loop_t* loop, uv_fs_t* req, uv_file fd, void* buf,
                 size_t len, int64_t offset, uv_fs_cb cb) {
  return uv_fs_read(loop, req, fd, buf, len, offset, cb);
}
extern "C" int
rust_uv_fs_close(uv_loop_t* loop, uv_fs_t* req, uv_file fd, uv_fs_cb cb) {
  return uv_fs_close(loop, req, fd, cb);
}
extern "C" void
rust_uv_fs_req_cleanup(uv_fs_t* req) {
  uv_fs_req_cleanup(req);
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

extern "C" int
rust_uv_fs_stat(uv_loop_t* loop, uv_fs_t* req, const char* path, uv_fs_cb cb) {
  return uv_fs_stat(loop, req, path, cb);
}
extern "C" int
rust_uv_fs_fstat(uv_loop_t* loop, uv_fs_t* req, uv_file file, uv_fs_cb cb) {
  return uv_fs_fstat(loop, req, file, cb);
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

extern "C" int
rust_uv_fs_mkdir(uv_loop_t* loop, uv_fs_t* req, const char* path, int mode, uv_fs_cb cb) {
  return uv_fs_mkdir(loop, req, path, mode, cb);
}
extern "C" int
rust_uv_fs_rmdir(uv_loop_t* loop, uv_fs_t* req, const char* path, uv_fs_cb cb) {
  return uv_fs_rmdir(loop, req, path, cb);
}

extern "C" int
rust_uv_fs_readdir(uv_loop_t* loop, uv_fs_t* req, const char* path, int flags, uv_fs_cb cb) {
  return uv_fs_readdir(loop, req, path, flags, cb);
}
extern "C" int
rust_uv_fs_rename(uv_loop_t *loop, uv_fs_t* req, const char *path,
                  const char *to, uv_fs_cb cb) {
    return uv_fs_rename(loop, req, path, to, cb);
}
extern "C" int
rust_uv_fs_chmod(uv_loop_t* loop, uv_fs_t* req, const char* path, int mode, uv_fs_cb cb) {
  return uv_fs_chmod(loop, req, path, mode, cb);
}

extern "C" int
rust_uv_spawn(uv_loop_t *loop, uv_process_t *p, uv_process_options_t options) {
  return uv_spawn(loop, p, options);
}

extern "C" int
rust_uv_process_kill(uv_process_t *p, int signum) {
  return uv_process_kill(p, signum);
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
rust_uv_pipe_init(uv_loop_t *loop, uv_pipe_t* p, int ipc) {
  return uv_pipe_init(loop, p, ipc);
}

extern "C" int
rust_uv_pipe_open(uv_pipe_t *pipe, int file) {
    return uv_pipe_open(pipe, file);
}

extern "C" int
rust_uv_pipe_bind(uv_pipe_t *pipe, char *name) {
    return uv_pipe_bind(pipe, name);
}

extern "C" void
rust_uv_pipe_connect(uv_connect_t *req, uv_pipe_t *handle,
                     char *name, uv_connect_cb cb) {
    uv_pipe_connect(req, handle, name, cb);
}

extern "C" int
rust_uv_tty_init(uv_loop_t *loop, uv_tty_t *tty, int fd, int readable) {
    return uv_tty_init(loop, tty, fd, readable);
}

extern "C" int
rust_uv_tty_set_mode(uv_tty_t *tty, int mode) {
    return uv_tty_set_mode(tty, mode);
}

extern "C" int
rust_uv_tty_get_winsize(uv_tty_t *tty, int *width, int *height) {
    return uv_tty_get_winsize(tty, width, height);
}

extern "C" int
rust_uv_guess_handle(int fd) {
    return uv_guess_handle(fd);
}

extern "C" int
rust_uv_signal_init(uv_loop_t* loop, uv_signal_t* handle) {
  return uv_signal_init(loop, handle);
}

extern "C" int
rust_uv_signal_start(uv_signal_t* handle, uv_signal_cb signal_cb, int signum) {
  return uv_signal_start(handle, signal_cb, signum);
}

extern "C" int
rust_uv_signal_stop(uv_signal_t* handle) {
  return uv_signal_stop(handle);
}
