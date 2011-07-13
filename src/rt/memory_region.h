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

class rust_srv;

class memory_region {
private:
    rust_srv *_srv;
    memory_region *_parent;
    size_t _live_allocations;
    array_list<void *> _allocation_list;
    const bool _detailed_leaks;
    const bool _synchronized;
    lock_and_signal _lock;

    void add_alloc();
    void dec_alloc();
public:
    memory_region(rust_srv *srv, bool synchronized);
    memory_region(memory_region *parent);
    void *malloc(size_t size);
    void *calloc(size_t size);
    void *realloc(void *mem, size_t size);
    void free(void *mem);
    virtual ~memory_region();
};

inline void *operator new(size_t size, memory_region &region) {
    return region.malloc(size);
}

inline void *operator new(size_t size, memory_region *region) {
    return region->malloc(size);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

#endif /* MEMORY_REGION_H */
