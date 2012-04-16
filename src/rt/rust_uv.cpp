#ifdef __WIN32__
// For alloca
#include <malloc.h>
#endif

#include "rust_globals.h"
#include "rust_task.h"
#include "rust_log.h"
#include "uv.h"

// crust fn pointers
typedef void (*crust_async_op_cb)(uv_loop_t* loop, void* data,
        uv_async_t* op_handle);
typedef void (*crust_simple_cb)(uint8_t* id_buf, void* loop_data);
typedef void (*crust_close_cb)(uint8_t* id_buf, void* handle,
        void* data);

// data types
#define RUST_UV_HANDLE_LEN 16

struct handle_data {
    uint8_t id_buf[RUST_UV_HANDLE_LEN];
    crust_simple_cb cb;
    crust_close_cb close_cb;
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
new_handle_data_from(uint8_t* buf, crust_simple_cb cb) {
    handle_data* data = (handle_data*)current_kernel_malloc(
            sizeof(handle_data),
            "handle_data");
    memcpy(data->id_buf, buf, RUST_UV_HANDLE_LEN);
    data->cb = cb;
    return data;
}

// libuv callback impls
static void
native_crust_async_op_cb(uv_async_t* handle, int status) {
    crust_async_op_cb cb = (crust_async_op_cb)handle->data;
    void* loop_data = handle->loop->data;
    cb(handle->loop, loop_data, handle);
}

static void
native_async_cb(uv_async_t* handle, int status) {
    handle_data* handle_d = (handle_data*)handle->data;
    void* loop_data = handle->loop->data;
    handle_d->cb(handle_d->id_buf, loop_data);
}

static void
native_timer_cb(uv_timer_t* handle, int status) {
    handle_data* handle_d = (handle_data*)handle->data;
    void* loop_data = handle->loop->data;
    handle_d->cb(handle_d->id_buf, loop_data);
}

static void
native_close_cb(uv_handle_t* handle) {
    handle_data* data = (handle_data*)handle->data;
    data->close_cb(data->id_buf, handle, handle->loop->data);
}

static void
native_close_op_cb(uv_handle_t* op_handle) {
    current_kernel_free(op_handle);
    // uv_run() should return after this..
}

// native fns bound in rust
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
rust_uv_bind_op_cb(uv_loop_t* loop, crust_async_op_cb cb) {
    uv_async_t* async = (uv_async_t*)current_kernel_malloc(
            sizeof(uv_async_t),
            "uv_async_t");
    uv_async_init(loop, async, native_crust_async_op_cb);
    async->data = (void*)cb;
    // decrement the ref count, so that our async bind
    // doesn't count towards keeping the loop alive
    //uv_unref(loop);
    return async;
}

extern "C" void
rust_uv_stop_op_cb(uv_handle_t* op_handle) {
    uv_close(op_handle, native_close_op_cb);
}

extern "C" void
rust_uv_run(uv_loop_t* loop) {
    uv_run(loop);
}

extern "C" void
rust_uv_close(uv_handle_t* handle, uv_close_cb cb) {
    uv_close(handle, cb);
}

extern "C" void
rust_uv_hilvl_close(uv_handle_t* handle, crust_close_cb cb) {
    handle_data* data = (handle_data*)handle->data;
    data->close_cb = cb;
    uv_close(handle, native_close_cb);
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
rust_uv_hilvl_async_init(uv_loop_t* loop, crust_simple_cb cb,
        uint8_t* buf) {
    uv_async_t* async = (uv_async_t*)current_kernel_malloc(
            sizeof(uv_async_t),
            "uv_async_t");
    uv_async_init(loop, async, native_async_cb);
    handle_data* data = new_handle_data_from(buf, cb);
    async->data = data;

    return async;
}

extern "C" void*
rust_uv_hilvl_timer_init(uv_loop_t* loop, crust_simple_cb cb,
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
    uv_timer_start(the_timer, native_timer_cb, timeout, repeat);
}

extern "C" int
rust_uv_timer_init(uv_loop_t* loop, uv_timer_t* timer) {
    return uv_timer_init(loop, timer);
}

extern "C" int
rust_uv_timer_start(uv_timer_t* the_timer, uv_timer_cb cb,
                        uint32_t timeout, uint32_t repeat) {
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
    rust_task* task = rust_get_current_task();
    LOG(task, stdlib, "inside rust_uv_tcp_connect");
    // FIXME ref #2064
    sockaddr_in addr = *addr_ptr;
    LOG(task, stdlib, "before tcp_connect .. port: %d",
        addr.sin_port);
    LOG(task, stdlib, "before tcp_connect.. tcp stream:" \
        "%lu cb ptr: %lu",
        (unsigned long int)tcp_ptr, (unsigned long int)cb);
    int result = uv_tcp_connect(connect_ptr, tcp_ptr, addr, cb);
    LOG(task, stdlib, "leaving rust_uv_tcp_connect.." \
        "and result: %d",
            result);
    return result;
}

extern "C" int
rust_uv_tcp_bind(uv_tcp_t* tcp_server, sockaddr_in* addr_ptr) {
    // FIXME ref #2064
    rust_task* task = rust_get_current_task();
    sockaddr_in addr = *addr_ptr;
    LOG(task, stdlib, "before uv_tcp_bind .. tcp_server:" \
        "%lu port: %d",
            (unsigned long int)tcp_server, addr.sin_port);
    return uv_tcp_bind(tcp_server, addr);
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
rust_uv_helper_uv_async_t_size() {
    return sizeof(uv_async_t);
}
extern "C" size_t
rust_uv_helper_uv_timer_t_size() {
    return sizeof(uv_timer_t);
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
    rust_task* task = rust_get_current_task();
    LOG(task, stdlib,"rust_uv_buf_init: base: %lu" \
        "len: %lu",
        (unsigned long int)base,
        (unsigned long int)len);
    *out_buf = uv_buf_init(base, len);
    LOG(task, stdlib, "rust_uv_buf_init: after: "
        "result->base: %" PRIxPTR " len: %" PRIxPTR,
           (unsigned long int)(*out_buf).base,
           (unsigned long int)(*out_buf).len);
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
rust_uv_set_data_for_uv_handle(uv_handle_t* handle,
        void* data) {
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
    rust_task* task = rust_get_current_task();
    LOG(task, stdlib, "before creating addr_ptr.. ip %s" \
        "port %d", ip, port);
    struct sockaddr_in addr = uv_ip4_addr(ip, port);
    LOG(task, stdlib, "after creating .. port: %d", addr.sin_port);
    return addr;
}

extern "C" uintptr_t*
rust_uv_get_kernel_global_chan_ptr() {
    uintptr_t* result = rust_get_current_task()->kernel->get_global_loop();
    rust_task* task = rust_get_current_task();
    LOG(task, stdlib, "global loop: %lu", (unsigned long int)result);
    LOG(task, stdlib,"global loop val: %lu", (unsigned long int)*result);
    return result;
}

extern "C" void**
rust_uv_get_kernel_global_async_handle() {
    return rust_get_current_task()->kernel->get_global_async_handle();
}
extern "C" void
rust_uv_set_kernel_global_async_handle(uv_async_t* handle) {
    rust_get_current_task()->kernel->set_global_async_handle((void*)handle);
}
extern "C" void
rust_uv_free_kernel_global_async_handle() {
    free((void*)rust_get_current_task()->kernel->get_global_async_handle());
}
