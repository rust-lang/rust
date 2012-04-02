#include "rust_internal.h"
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
extern "C" void*
rust_uv_loop_new() {
    return (void*)uv_loop_new();
}

extern "C" void
rust_uv_loop_delete(uv_loop_t* loop) {
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
rust_uv_close(uv_handle_t* handle, crust_close_cb cb) {
        handle_data* data = (handle_data*)handle->data;
        data->close_cb = cb;
        uv_close(handle, native_close_cb);
}

extern "C" void
rust_uv_close_async(uv_async_t* handle) {
  current_kernel_free(handle->data);
  current_kernel_free(handle);
}

extern "C" void
rust_uv_close_timer(uv_async_t* handle) {
  current_kernel_free(handle->data);
  current_kernel_free(handle);
}

extern "C" void
rust_uv_async_send(uv_async_t* handle) {
    uv_async_send(handle);
}

extern "C" void*
rust_uv_async_init(uv_loop_t* loop, crust_simple_cb cb,
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
rust_uv_timer_init(uv_loop_t* loop, crust_simple_cb cb,
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
rust_uv_timer_start(uv_timer_t* the_timer, uint32_t timeout,
                                                  uint32_t repeat) {
        uv_timer_start(the_timer, native_timer_cb, timeout, repeat);
}

extern "C" void
rust_uv_timer_stop(uv_timer_t* the_timer) {
  uv_timer_stop(the_timer);
}

