// Rust garbage collection.

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <stdint.h>

#include "rust_abi.h"
#include "rust_debug.h"
#include "rust_gc.h"
#include "rust_internal.h"
#include "rust_shape.h"

#ifdef __WIN32__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace stack_walk;

namespace gc {

weak_symbol<const uintptr_t> safe_point_data("rust_gc_safe_points");

struct root_info {
    intptr_t frame_offset;
    uintptr_t dynamic;  // 0 = static, 1 = dynamic
    const type_desc *tydesc;
};

struct root {
    const type_desc *tydesc;
    uint8_t *data;

    root(const root_info &info, const frame &frame)
    : tydesc(info.tydesc),
      data((uint8_t *)frame.bp + info.frame_offset) {}
};

struct safe_point {
    uintptr_t n_roots;
    root_info roots[0];
};

struct safe_point_index_entry {
    void (*ra)();                   // The return address.
    const struct safe_point *safe_point;   // The safe point.

    struct cmp {
        bool operator()(const safe_point_index_entry &entry, void (*ra)())
                const {
            return entry.ra < ra;
        }
        bool operator()(void (*ra)(), const safe_point_index_entry &entry)
                const {
            return ra < entry.ra;
        }
    };
};

class safe_point_map {
    uintptr_t n_safe_points;
    const safe_point_index_entry *index;
    const safe_point *safe_points;

public:
    safe_point_map() {
        const uintptr_t *data = *safe_point_data;
        n_safe_points = *data++;
        index = (const safe_point_index_entry *)data;
        data += n_safe_points * 2;
        safe_points = (const safe_point *)data;
    }

    const safe_point *get_safe_point(void (*addr)());
};

class gc {
private:
    rust_task *task;

    void mark(std::vector<root> &roots);
    void sweep();

public:
    gc(rust_task *in_task) : task(in_task) {}
    void run();
};

const safe_point *
safe_point_map::get_safe_point(void (*addr)()) {
    safe_point_index_entry::cmp cmp;
    const safe_point_index_entry *entry =
        std::lower_bound(index, index + n_safe_points, addr, cmp);
    return (entry && entry->ra == addr) ? entry->safe_point : NULL;
}

void
gc::mark(std::vector<root> &roots) {
    std::vector<root>::iterator ri = roots.begin(), rend = roots.end();
    while (ri < rend) {
        DPRINT("root: %p\n", ri->data);

        shape::arena arena;
        shape::type_param *params =
            shape::type_param::from_tydesc_and_data(ri->tydesc, ri->data,
                                                    arena);
        shape::log log(task, true, ri->tydesc->shape, params,
                       ri->tydesc->shape_tables, ri->data, std::cerr);
        log.walk();
        DPRINT("\n");

        ++ri;
    }
    // TODO
}

void
gc::sweep() {
    // TODO
}

void
gc::run() {
    safe_point_map map;

    // Find roots.
    std::vector<root> roots;
    std::vector<frame> call_stack = backtrace();
    for (unsigned i = 0; i < call_stack.size(); i++) {
        frame f = call_stack[i];
        const safe_point *sp = map.get_safe_point(f.ra);
        if (!sp)
            continue;

        DPRINT("%u: ra %p, ebp %p\n", i, call_stack[i].ra, call_stack[i].bp);
        for (unsigned j = 0; j < sp->n_roots; j++) {
            root r(sp->roots[j], f);
            roots.push_back(r);
        }
    }

    // Mark and sweep.
    mark(roots);
    sweep();
}

void
maybe_gc(rust_task *task) {
    if (*safe_point_data == NULL)
        return;

    static debug::flag zeal("RUST_GC_ZEAL");

    if (*zeal) {
        gc gc(task);
        gc.run();
    }
}

}

