#include "rust_internal.h"
#include "memory_region.h"

// NB: please do not commit code with this uncommented. It's
// hugely expensive and should only be used as a last resort.
//
// #define TRACK_ALLOCATIONS

memory_region::memory_region(rust_srv *srv, bool synchronized) :
    _srv(srv), _parent(NULL), _live_allocations(0),
    _detailed_leaks(getenv("RUST_DETAILED_LEAKS") != NULL),
    _synchronized(synchronized) {
}

memory_region::memory_region(memory_region *parent) :
    _srv(parent->_srv), _parent(parent), _live_allocations(0),
    _detailed_leaks(parent->_detailed_leaks),
    _synchronized(parent->_synchronized) {
    // Nop.
}

void memory_region::free(void *mem) {
    // printf("free: ptr 0x%" PRIxPTR"\n", (uintptr_t) mem);
    if (!mem) { return; }
    if (_synchronized) { _lock.lock(); }
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list.replace(mem, NULL) == false) {
        printf("free: ptr 0x%" PRIxPTR " is not in allocation_list\n",
            (uintptr_t) mem);
        _srv->fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
#endif
    if (_live_allocations < 1) {
        _srv->fatal("live_allocs < 1", __FILE__, __LINE__, "");
    }
    _live_allocations--;
    _srv->free(mem);
    if (_synchronized) { _lock.unlock(); }
}

void *
memory_region::realloc(void *mem, size_t size) {
    if (_synchronized) { _lock.lock(); }
    if (!mem) {
        _live_allocations++;
    }
    void *newMem = _srv->realloc(mem, size);
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list.replace(mem, newMem) == false) {
        printf("realloc: ptr 0x%" PRIxPTR " is not in allocation_list\n",
            (uintptr_t) mem);
        _srv->fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
#endif
    if (_synchronized) { _lock.unlock(); }
    return newMem;
}

void *
memory_region::malloc(size_t size) {
    if (_synchronized) { _lock.lock(); }
    _live_allocations++;
    void *mem = _srv->malloc(size);
#ifdef TRACK_ALLOCATIONS
    _allocation_list.append(mem);
#endif
    // printf("malloc: ptr 0x%" PRIxPTR "\n", (uintptr_t) mem);
    if (_synchronized) { _lock.unlock(); }
    return mem;
}

void *
memory_region::calloc(size_t size) {
    if (_synchronized) { _lock.lock(); }
    _live_allocations++;
    void *mem = _srv->malloc(size);
    memset(mem, 0, size);
#ifdef TRACK_ALLOCATIONS
    _allocation_list.append(mem);
#endif
    if (_synchronized) { _lock.unlock(); }
    return mem;
}

memory_region::~memory_region() {
    if (_live_allocations == 0) {
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
    _srv->fatal(msg, __FILE__, __LINE__, "%d objects", _live_allocations);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
