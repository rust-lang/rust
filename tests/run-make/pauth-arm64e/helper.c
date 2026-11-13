#include <stdint.h>

#ifndef __arm64e__
#error This test must compile its C half for arm64e
#endif

typedef uintptr_t (*callback_t)(uintptr_t);

_Thread_local uintptr_t c_tls = 37;

uintptr_t c_target(uintptr_t value) { return value + 3; }

callback_t c_get_callback(void) { return c_target; }

uintptr_t c_call(callback_t callback, uintptr_t value) {
  return callback(value);
}

extern uintptr_t rust_c_target(uintptr_t);

uintptr_t c_call_rust(uintptr_t value) { return rust_c_target(value); }
