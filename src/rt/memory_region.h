/*
 * The Rust runtime uses memory regions to provide a primitive level of
 * memory management and isolation between tasks, and domains.
 *
 * TODO: Implement a custom lock-free malloc / free instead of relying solely
 *       on the standard malloc / free.
 */

#ifndef MEMORY_REGION_H
#define MEMORY_REGION_H

#include "sync/spin_lock.h"

class rust_srv;

class memory_region {
private:
    rust_srv *_srv;
    memory_region *_parent;
    size_t _live_allocations;
    array_list<void *> _allocation_list;
    const bool _synchronized;
    spin_lock _lock;
public:
    enum memory_region_type {
        LOCAL = 0x1, SYNCHRONIZED = 0x2
    };
    memory_region(rust_srv *srv, bool synchronized);
    memory_region(memory_region *parent);
    void *malloc(size_t size);
    void *calloc(size_t size);
    void *realloc(void *mem, size_t size);
    void free(void *mem);
    virtual ~memory_region();
};

#endif /* MEMORY_REGION_H */
