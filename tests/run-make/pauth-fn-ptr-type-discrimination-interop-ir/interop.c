// C side of the interop test. Built with Clang's function pointer type discrimination: a code
// address is signed according to the C type of the slot holding it, and C casts resign.

#include <stddef.h>
#include <string.h>

typedef int (*binop_t)(int, int);
typedef void (*generic_cb_t)(void);

static int c_sub(int a, int b) { return a - b; }
static int c_mul(int a, int b) { return a * b; }

// Correctly typed callback: C calls a Rust function through its real type.
int apply(binop_t op, int a, int b) { return op(a, b); }

// Correctly typed function pointer returned to Rust.
binop_t get_sub(void) { return c_sub; }

// `dlsym`-like lookup: the code address is handed out as a `void *` (discriminator 0).
void *lookup(const char *name) {
  if (strcmp(name, "mul") == 0)
    return (void *)c_mul;
  return NULL;
}

// Generic callback slot (GLib's `GCallback` style): the callback is stored with the generic type
// and cast back to its real type before the call.
static generic_cb_t g_slot;
void register_callback(generic_cb_t cb) { g_slot = cb; }
int invoke_callback(int a, int b) { return ((binop_t)g_slot)(a, b); }

// Generic callback handed out by C.
generic_cb_t get_proc(const char *name) {
  if (strcmp(name, "sub") == 0)
    return (generic_cb_t)c_sub;
  return NULL;
}

// A code address passed in as `void *` user data, cast back before the call.
int call_opaque(void *fn, int a, int b) { return ((binop_t)fn)(a, b); }
