/*
 * The Rust runtime uses memory regions to provide a primitive level of
 * memory management and isolation between tasks, and domains.
 *
 * FIXME: Implement a custom lock-free malloc / free instead of relying solely
 *       on the standard malloc / free.
 */

#ifndef MEMORY_REGION_H
#define MEMORY_REGION_H

#include "sync/lock_and_signal.h"

// There are three levels of debugging:
//
// 0 --- no headers, no debugging support
// 1 --- support poison, but do not track allocations
// 2 --- track allocations in detail
//
// NB: please do not commit code with level 2. It's
// hugely expensive and should only be used as a last resort.
#define RUSTRT_TRACK_ALLOCATIONS 0

class rust_srv;

class memory_region {
private:
    struct alloc_header {
#       if RUSTRT_TRACK_ALLOCATIONS > 0
        uint32_t magic;
        int index;
        const char *tag;
        uint32_t size;
#       endif
    };

    inline alloc_header *get_header(void *mem);
    inline void *get_data(alloc_header *);
    

    rust_srv *_srv;
    memory_region *_parent;
    int _live_allocations;
    array_list<alloc_header *> _allocation_list;
    const bool _detailed_leaks;
    const bool _synchronized;
    lock_and_signal _lock;

    void add_alloc();
    void dec_alloc();
    void maybe_poison(void *mem);

    void release_alloc(void *mem);
    void claim_alloc(void *mem);

public:
    memory_region(rust_srv *srv, bool synchronized);
    memory_region(memory_region *parent);
    void *malloc(size_t size, const char *tag, bool zero = true);
    void *calloc(size_t size, const char *tag);
    void *realloc(void *mem, size_t size);
    void free(void *mem);
    virtual ~memory_region();
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
