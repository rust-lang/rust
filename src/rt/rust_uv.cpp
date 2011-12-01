#include "rust_internal.h"
#include "uv.h"

/*
  Wrappers of uv_* functions. These can be eliminated by figuring
  out how to build static uv with externs, or by just using dynamic libuv
 */

extern "C" CDECL uv_loop_t*
rust_uv_default_loop() {
  return uv_default_loop();
}

extern "C" CDECL uv_loop_t*
rust_uv_loop_new() {
  return uv_loop_new();
}

extern "C" CDECL void
rust_uv_loop_delete(uv_loop_t *loop) {
  return uv_loop_delete(loop);
}

extern "C" CDECL int
rust_uv_run(uv_loop_t *loop) {
  return uv_run(loop);
}

extern "C" CDECL void
rust_uv_unref(uv_loop_t *loop) {
  return uv_unref(loop);
}

extern "C" CDECL int
rust_uv_idle_init(uv_loop_t* loop, uv_idle_t* idle) {
  return uv_idle_init(loop, idle);
}

extern "C" CDECL int
rust_uv_idle_start(uv_idle_t* idle, uv_idle_cb cb) {
  return uv_idle_start(idle, cb);
}




extern "C" CDECL size_t
rust_uv_size_of_idle_t() {
  return sizeof(uv_idle_t);
}
