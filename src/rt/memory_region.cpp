#include "rust_internal.h"
#include "memory_region.h"

// NB: please do not commit code with this uncommented. It's
// hugely expensive and should only be used as a last resort.
//
// #define TRACK_ALLOCATIONS

#define MAGIC 0xbadc0ffe

memory_region::alloc_header *memory_region::get_header(void *mem) {
    return (alloc_header *)((char *)mem - sizeof(alloc_header));
}

memory_region::memory_region(rust_srv *srv, bool synchronized) :
    _srv(srv), _parent(NULL), _live_allocations(0),
    _detailed_leaks(getenv("RUST_DETAILED_LEAKS") != NULL),
    _synchronized(synchronized), _hack_allow_leaks(false) {
}

memory_region::memory_region(memory_region *parent) :
    _srv(parent->_srv), _parent(parent), _live_allocations(0),
    _detailed_leaks(parent->_detailed_leaks),
    _synchronized(parent->_synchronized), _hack_allow_leaks(false) {
}

void memory_region::add_alloc() {
    _live_allocations++;
    //sync::increment(_live_allocations);
}

void memory_region::dec_alloc() {
    _live_allocations--;
    //sync::decrement(_live_allocations);
}

void memory_region::free(void *mem) {
    // printf("free: ptr 0x%" PRIxPTR" region=%p\n", (uintptr_t) mem, this);
    if (!mem) { return; }
    if (_synchronized) { _lock.lock(); }
    alloc_header *alloc = get_header(mem);
    assert(alloc->magic == MAGIC);
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list[alloc->index] != alloc) {
        printf("free: ptr 0x%" PRIxPTR " is not in allocation_list\n",
               (uintptr_t) mem);
        _srv->fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
    else {
        // printf("freed index %d\n", index);
        _allocation_list[alloc->index] = NULL;
    }
#endif
    if (_live_allocations < 1) {
        _srv->fatal("live_allocs < 1", __FILE__, __LINE__, "");
    }
    dec_alloc();
    _srv->free(alloc);
    if (_synchronized) { _lock.unlock(); }
}

void *
memory_region::realloc(void *mem, size_t size) {
    if (_synchronized) { _lock.lock(); }
    if (!mem) {
        add_alloc();
    }
    size += sizeof(alloc_header);
    alloc_header *alloc = get_header(mem);
    assert(alloc->magic == MAGIC);
    alloc_header *newMem = (alloc_header *)_srv->realloc(alloc, size);
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list[newMem->index] != alloc) {
        printf("at index %d, found %p, expected %p\n",
               alloc->index, _allocation_list[alloc->index], alloc);
        printf("realloc: ptr 0x%" PRIxPTR " is not in allocation_list\n",
            (uintptr_t) mem);
        _srv->fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
    else {
        _allocation_list[newMem->index] = newMem;
        // printf("realloc: stored %p at index %d, replacing %p\n",
        //        newMem, index, mem);
    }
#endif
    if (_synchronized) { _lock.unlock(); }
    return newMem->data;
}

void *
memory_region::malloc(size_t size, const char *tag, bool zero) {
    if (_synchronized) { _lock.lock(); }
    add_alloc();
    size_t old_size = size;
    size += sizeof(alloc_header);
    alloc_header *mem = (alloc_header *)_srv->malloc(size);
    mem->magic = MAGIC;
    mem->tag = tag;
#ifdef TRACK_ALLOCATIONS
    mem->index = _allocation_list.append(mem);
    // printf("malloc: stored %p at index %d\n", mem, index);
#endif
    // printf("malloc: ptr 0x%" PRIxPTR " region=%p\n",
    //        (uintptr_t) mem, this);

    if(zero) {
        memset(mem->data, 0, old_size);
    }

    if (_synchronized) { _lock.unlock(); }
    return mem->data;
}

void *
memory_region::calloc(size_t size, const char *tag) {
    return malloc(size, tag, true);
}

memory_region::~memory_region() {
    if (_synchronized) { _lock.lock(); }
    if (_live_allocations == 0) {
        if (_synchronized) { _lock.unlock(); }
        return;
    }
    char msg[128];
    snprintf(msg, sizeof(msg),
             "leaked memory in rust main loop (%" PRIuPTR " objects)",
             _live_allocations);
#ifdef TRACK_ALLOCATIONS
    if (_detailed_leaks) {
        for (size_t i = 0; i < _allocation_list.size(); i++) {
            if (_allocation_list[i] != NULL) {
                printf("allocation 0x%" PRIxPTR " was not freed\n",
                       (uintptr_t) _allocation_list[i]);
            }
        }
    }
#endif
    if (!_hack_allow_leaks) {
        _srv->fatal(msg, __FILE__, __LINE__,
                    "%d objects", _live_allocations);
    } else {
        _srv->warning(msg, __FILE__, __LINE__,
                      "%d objects", _live_allocations);
    }
    if (_synchronized) { _lock.unlock(); }
}

void
memory_region::hack_allow_leaks() {
    _hack_allow_leaks = true;
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
