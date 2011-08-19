// Rust garbage collection.

// TODO: Windows
#include <utility>
#include <stdint.h>

#include "rust_internal.h"

#ifdef __WIN32__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace gc {

struct root {
    intptr_t frame_offset;
    uintptr_t dynamic;  // 0 = static, 1 = dynamic
    type_desc *tydesc;
};

struct safe_point {
    uintptr_t n_roots;
    root roots[0];
};

class safe_point_map {
    uintptr_t n_safe_points;
    const std::pair<void *,const safe_point *> *index;
    const safe_point *safe_points;

public:
    safe_point_map() {
        const uintptr_t *data;
#ifdef __WIN32__
        data = (const uintptr_t *)GetProcAddress(GetModuleHandle(NULL),
                                                 "rust_gc_safe_points");
#else
        data = (const uintptr_t *)dlsym(RTLD_DEFAULT, "rust_gc_safe_points");
#endif
        n_safe_points = *data++;
        index = (const std::pair<void *,const safe_point *> *)data;
        data += n_safe_points * 2;
        safe_points = (const safe_point *)data;
    }
};

void
gc() {
    safe_point_map map;

    // TODO
}

void
maybe_gc() {
    // FIXME: We ought to lock this.
    static int zeal = -1;
    if (zeal == -1) {
        char *ev = getenv("RUST_GC_ZEAL");
        zeal = ev[0] != '\0' && ev[0] != '0';
    }

    if (zeal)
        gc();
}

}

