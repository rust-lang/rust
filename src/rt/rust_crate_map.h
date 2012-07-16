#ifndef RUST_CRATE_MAP_H
#define RUST_CRATE_MAP_H

#include "rust_log.h"

struct mod_entry {
    const char* name;
    uint32_t* state;
};

struct cratemap {
    const mod_entry* entries;
    const cratemap* children[1];
};

void iter_module_map(const mod_entry* map,
                     void (*fn)(const mod_entry* entry, void *cookie),
                     void *cookie);

void iter_crate_map(const cratemap* map,
                    void (*fn)(const mod_entry* entry, void *cookie),
                    void *cookie);

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

#endif /* RUST_CRATE_MAP_H */
