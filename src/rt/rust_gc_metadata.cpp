// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rust_gc_metadata.h"
#include "rust_crate_map.h"
#include "rust_globals.h"

#include <algorithm>
#include <vector>

struct safe_point {
    uintptr_t safe_point_loc;
    uintptr_t safe_point_meta;
    uintptr_t function_meta;
};

struct update_gc_entry_args {
    std::vector<safe_point> *safe_points;
};

static void
update_gc_entry(const mod_entry *entry, void *cookie) {
    update_gc_entry_args *args = (update_gc_entry_args *)cookie;
    if (!strcmp(entry->name, "_gc_module_metadata")) {
        uintptr_t *next = (uintptr_t *)entry->state;
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

uintptr_t *global_safe_points = 0;

void
update_gc_metadata(const void* map) {
    std::vector<safe_point> safe_points;
    update_gc_entry_args args = { &safe_points };

    // Extract list of safe points from each module.
    iter_crate_map((const cratemap *)map, update_gc_entry, (void *)&args);
    std::sort(safe_points.begin(), safe_points.end(), cmp_safe_point);

    // Serialize safe point list into format expected by runtime.
    global_safe_points =
        (uintptr_t *)malloc((safe_points.size()*3 + 1)*sizeof(uintptr_t));
    if (!global_safe_points) return;

    uintptr_t *next = global_safe_points;
    *next = safe_points.size();
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

extern "C" CDECL void
rust_update_gc_metadata(const void* map) {
    update_gc_metadata(map);
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
