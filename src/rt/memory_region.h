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
    struct alloc_header {
        uint32_t magic;
        int index;
        const char *tag;
        uint32_t size;
        char data[];
    };

    alloc_header *get_header(void *mem);

    rust_srv *_srv;
    memory_region *_parent;
    int _live_allocations;
    array_list<alloc_header *> _allocation_list;
    const bool _detailed_leaks;
    const bool _synchronized;
    lock_and_signal _lock;
    bool _hack_allow_leaks;

    void add_alloc();
    void dec_alloc();
    void maybe_poison(void *mem);

public:
    memory_region(rust_srv *srv, bool synchronized);
    memory_region(memory_region *parent);
    void *malloc(size_t size, const char *tag, bool zero = true);
    void *calloc(size_t size, const char *tag);
    void *realloc(void *mem, size_t size);
    void free(void *mem);
    virtual ~memory_region();
    // FIXME (236: This is a temporary hack to allow failing tasks that leak
    // to not kill the entire process, which the test runner needs. Please
    // kill with prejudice once unwinding works.
    void hack_allow_leaks();

    void release_alloc(void *mem);
    void claim_alloc(void *mem);
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

#endif /* MEMORY_REGION_H */
