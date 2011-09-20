// Rust cycle collector. Temporary, but will probably stick around for some
// time until LLVM's GC infrastructure is more mature.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include "rust_gc.h"
#include "rust_internal.h"
#include "rust_shape.h"
#include "rust_task.h"

#undef DPRINT
#define DPRINT(fmt,...)     fprintf(stderr, fmt, ##__VA_ARGS__)

namespace cc {

void
do_cc(rust_task *task) {
    std::map<void *,type_desc *>::iterator begin(task->local_allocs.begin());
    std::map<void *,type_desc *>::iterator end(task->local_allocs.end());
    while (begin != end) {
        void *p = begin->first;
        type_desc *tydesc = begin->second;

        DPRINT("marking allocation: %p, tydesc=%p\n", p, tydesc);

        // Prevents warnings for now
        (void)p;
        (void)tydesc;
#if 0
        shape::arena arena;
        shape::type_param *params =
            shape::type_param::from_tydesc(tydesc, arena);
        mark mark(task, true, tydesc->shape, params, tydesc->shape_tables, p);
        mark.walk();
#endif

        ++begin;
    }
}

void
maybe_cc(rust_task *task) {
    // FIXME: We ought to lock this.
    static int zeal = -1;
    if (zeal == -1) {
        char *ev = getenv("RUST_CC_ZEAL");
        zeal = ev && ev[0] != '\0' && ev[0] != '0';
    }

    if (zeal)
        do_cc(task);
}

}   // end namespace cc

