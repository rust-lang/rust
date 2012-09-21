#include "rust_crate_map.h"

void iter_module_map(const mod_entry* map,
                     void (*fn)(const mod_entry* entry, void *cookie),
                     void *cookie) {
    for (const mod_entry* cur = map; cur->name; cur++) {
        fn(cur, cookie);
    }
}

void iter_crate_map(const cratemap* map,
                    void (*fn)(const mod_entry* map, void *cookie),
                    void *cookie) {
    // First iterate this crate
    iter_module_map(map->entries(), fn, cookie);
    // Then recurse on linked crates
    // FIXME (#2673) this does double work in diamond-shaped deps. could
    //   keep a set of visited addresses, if it turns out to be actually
    //   slow
    for (cratemap::iterator i = map->begin(), e = map->end(); i != e; ++i) {
        iter_crate_map(*i, fn, cookie);
    }
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
