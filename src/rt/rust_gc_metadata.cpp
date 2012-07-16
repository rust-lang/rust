#include "rust_gc_metadata.h"
#include "rust_crate_map.h"
#include "rust_globals.h"

#include <algorithm>
#include <vector>

struct safe_point {
    size_t safe_point_loc;
    size_t safe_point_meta;
    size_t function_meta;
};

struct update_gc_entry_args {
    std::vector<safe_point> *safe_points;
};

static void
update_gc_entry(const mod_entry* entry, void *cookie) {
    update_gc_entry_args *args = (update_gc_entry_args *)cookie;
    if (!strcmp(entry->name, "_gc_module_metadata")) {
        size_t *next = entry->state;
        uint32_t num_safe_points = *(uint32_t *)next;
        next++;

        for (uint32_t i = 0; i < num_safe_points; i++) {
            safe_point sp = { next[0], next[1], next[2] };
            next += 3;

            args->safe_points->push_back(sp);
        }
    }
}

static bool
cmp_safe_point(safe_point a, safe_point b) {
    return a.safe_point_loc < b.safe_point_loc;
}

size_t *global_safe_points = 0;

void
update_gc_metadata(const void* map) {
    std::vector<safe_point> safe_points;
    update_gc_entry_args args = { &safe_points };

    // Extract list of safe points from each module.
    iter_crate_map((const cratemap *)map, update_gc_entry, (void *)&args);
    std::sort(safe_points.begin(), safe_points.end(), cmp_safe_point);

    // Serialize safe point list into format expected by runtime.
    global_safe_points =
        (size_t *)malloc((safe_points.size()*3 + 1)*sizeof(size_t));
    if (!global_safe_points) return;

    size_t *next = global_safe_points;
    *(uint32_t *)next = safe_points.size();
    next++;
    for (uint32_t i = 0; i < safe_points.size(); i++) {
        next[0] = safe_points[i].safe_point_loc;
        next[1] = safe_points[i].safe_point_meta;
        next[2] = safe_points[i].function_meta;
        next += 3;
    }
}

extern "C" CDECL void *
rust_gc_metadata() {
    return (void *)global_safe_points;
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
