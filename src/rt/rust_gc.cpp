// Rust garbage collection.

#include <algorithm>
#include <utility>
#include <vector>
#include <stdint.h>

#include "rust_gc.h"
#include "rust_internal.h"

#ifdef __WIN32__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DPRINT(fmt,...)     fprintf(stderr, fmt, ##__VA_ARGS__)

#define END_OF_STACK_RA     (void (*)())0xdeadbeef

namespace gc {

struct frame {
    uint8_t *bp;    // The frame pointer.
    void (*ra)();   // The return address.

    frame(void *in_bp) : bp((uint8_t *)in_bp) {}

    inline void read_ra() {
        ra = *(void (**)())(bp + sizeof(void *));
    }

    inline void next() {
        bp = *(uint8_t **)bp;
    }
};

struct root_info {
    intptr_t frame_offset;
    uintptr_t dynamic;  // 0 = static, 1 = dynamic
    type_desc *tydesc;
};

struct root {
    type_desc *tydesc;
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
        const uintptr_t *data = get_safe_point_data();
        n_safe_points = *data++;
        index = (const safe_point_index_entry *)data;
        data += n_safe_points * 2;
        safe_points = (const safe_point *)data;
    }

    const safe_point *get_safe_point(void (*addr)());

    static const uintptr_t *get_safe_point_data() {
        static bool init = false;
        static const uintptr_t *data;
        if (!init) {
#ifdef __WIN32__
            data = (const uintptr_t *)GetProcAddress(GetModuleHandle(NULL),
                                                     "rust_gc_safe_points");
#else
            data = (const uintptr_t *)dlsym(RTLD_DEFAULT,
                                            "rust_gc_safe_points");
#endif
            init = true;
        }
        return data;
    }
};

class gc {
private:
    void mark(std::vector<root> &roots);
    void sweep();

public:
    void run(rust_task *task);
    std::vector<frame> backtrace();
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
        ++ri;
    }
    // TODO
}

void
gc::sweep() {
    // TODO
}

std::vector<frame>
gc::backtrace() {
    std::vector<frame> frames;
    frame f(__builtin_frame_address(0));
    while (f.ra != END_OF_STACK_RA) {
        f.read_ra();
        frames.push_back(f);
        f.next();
    }
    return frames;
}

void
gc::run(rust_task *task) {
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
    if (safe_point_map::get_safe_point_data() == NULL)
        return;

    // FIXME: We ought to lock this.
    static int zeal = -1;
    if (zeal == -1) {
        char *ev = getenv("RUST_GC_ZEAL");
        zeal = ev && ev[0] != '\0' && ev[0] != '0';
    }

    if (zeal) {
        gc gc;
        gc.run(task);
    }
}

}

