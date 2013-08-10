// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_CRATE_MAP_H
#define RUST_CRATE_MAP_H

#include "rust_globals.h"
#include <stdint.h>

struct mod_entry {
    const char* name;
    uint32_t* state;
};

class cratemap;

class cratemap_v0 {
    friend class cratemap;
    const mod_entry *m_entries;
    const cratemap* m_children[1];
};

class cratemap {
private:
    int32_t m_version;
    const void *m_annihilate_fn;
    const mod_entry* m_entries;
    const cratemap* m_children[1];

    inline int32_t version() const {
        switch (m_version) {
        case 1:     return 1;
        default:    return 0;
        }
    }

public:
    typedef const cratemap *const *iterator;

    inline const void *annihilate_fn() const {
        switch (version()) {
        case 0: return NULL;
        case 1: return m_annihilate_fn;
        default: assert(false && "Unknown crate map version!");
            return NULL; // Appease -Werror=return-type
        }
    }

    inline const mod_entry *entries() const {
        switch (version()) {
        case 0: return reinterpret_cast<const cratemap_v0 *>(this)->m_entries;
        case 1: return m_entries;
        default: assert(false && "Unknown crate map version!");
            return NULL; // Appease -Werror=return-type
        }
    }

    inline const iterator begin() const {
        switch (version()) {
        case 0:
            return &reinterpret_cast<const cratemap_v0 *>(this)->
                m_children[0];
        case 1:
            return &m_children[0];
        default: assert(false && "Unknown crate map version!");
            return NULL; // Appease -Werror=return-type
        }
    }

    inline const iterator end() const {
        iterator i = begin();
        while (*i)
            i++;
        return i;
    }
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
