// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * The Rust runtime uses memory regions to provide a primitive level of
 * memory management and isolation between tasks, and domains.
 *
 * FIXME (#2686): Implement a custom lock-free malloc / free instead of
 *       relying solely on the standard malloc / free.
 */

#ifndef MEMORY_REGION_H
#define MEMORY_REGION_H

#include "rust_globals.h"
#include "sync/lock_and_signal.h"
#include "util/array_list.h"

// There are three levels of debugging:
//
// 0 --- no headers, no debugging support
// 1 --- support poison, but do not track allocations
// 2 --- track allocations in detail
// 3 --- record backtraces of every allocation
//
// NB: please do not commit code with level 2. It's
// hugely expensive and should only be used as a last resort.
#define RUSTRT_TRACK_ALLOCATIONS 0

struct rust_env;

class memory_region {
private:
    struct alloc_header {
#       if RUSTRT_TRACK_ALLOCATIONS > 0
        uint32_t magic;
        int index;
        const char *tag;
        uint32_t size;
#       if RUSTRT_TRACK_ALLOCATIONS >= 3
        void *bt[32];
        int btframes;
#       endif
#       endif
    };

    inline alloc_header *get_header(void *mem);
    inline void *get_data(alloc_header *);

    memory_region *_parent;
    int _live_allocations;
    array_list<alloc_header *> _allocation_list;
    const bool _detailed_leaks;
    const bool _poison_on_free;
    lock_and_signal _lock;

    void add_alloc();
    void dec_alloc();
    void maybe_poison(void *mem);

    void release_alloc(void *mem);
    void claim_alloc(void *mem);

    void maybe_print_backtrace(const alloc_header *) const;

private:
    // private and undefined to disable copying
    memory_region(const memory_region& rhs);
    memory_region& operator=(const memory_region& rhs);

public:
    memory_region(bool detailed_leaks, bool poison_on_free);
    memory_region(memory_region *parent);
    void *malloc(size_t size, const char *tag);
    void *realloc(void *mem, size_t size);
    void free(void *mem);
    ~memory_region();
 };

inline void *operator new(size_t size, memory_region &region,
                          const char *tag) {
    return region.malloc(size, tag);
}

inline void *operator new(size_t size, memory_region *region,
                          const char *tag) {
    return region->malloc(size, tag);
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

#endif /* MEMORY_REGION_H */
