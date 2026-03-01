#include <stddef.h>
#include <stdint.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

EXPORT void call_fn_ptr(void f(void)) {
    if (f != NULL) {
        f();
    }
}
