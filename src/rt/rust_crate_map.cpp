// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rust_crate_map.h"
#include <set>

void iter_module_map(const mod_entry* map,
                    void (*fn)(void* fptr, void* env, const mod_entry *entry),
                    void* fptr,
                    void* env
                    ) {
    for (const mod_entry* cur = map; cur->name; cur++) {
        fn(fptr, env, cur);
    }
}

void iter_crate_map(const cratemap* map,
                    void (*fn)(void* fptr, void* env, const mod_entry *entry),
                    void *fptr,
                    void *env,
                    std::set<const cratemap*>& visited) {
    if (visited.find(map) == visited.end()) {
        // Mark this crate visited
        visited.insert(map);
        // First iterate this crate
        iter_module_map(map->entries(), fn, fptr, env);
        // Then recurse on linked crates
        for (cratemap::iterator i = map->begin(),
                e = map->end(); i != e; ++i) {
            iter_crate_map(*i, fn, fptr, env, visited);
        }
    }
}

void iter_crate_map(const cratemap* map,
                    void (*fn)(void* fptr, void* env, const mod_entry *entry),
                    void *fptr,
                    void *env
                    ) {
    std::set<const cratemap*> visited;
    iter_crate_map(map, fn, fptr, env, visited);
}

extern "C" CDECL void
rust_iter_crate_map(const cratemap* map,
                    void (*fn)(void* fptr, void* env, const mod_entry *entry),
                    void *fptr,
                    void *env
                    ) {
    return iter_crate_map(map, fn, fptr, env);
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
