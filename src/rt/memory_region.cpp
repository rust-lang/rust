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

void memory_region::add_alloc() {
    //_live_allocations++;
    sync::increment(_live_allocations);
}

void memory_region::dec_alloc() {
    //_live_allocations--;
    sync::decrement(_live_allocations);
}

void memory_region::free(void *mem) {
    // printf("free: ptr 0x%" PRIxPTR" region=%p\n", (uintptr_t) mem, this);
    if (!mem) { return; }
    if (_synchronized) { _lock.lock(); }
#ifdef TRACK_ALLOCATIONS
    int index = ((int  *)mem)[-1];
    if (_allocation_list[index] != (uint8_t *)mem - sizeof(int)) {
        printf("free: ptr 0x%" PRIxPTR " is not in allocation_list\n",
               (uintptr_t) mem);
        _srv->fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
    else {
        // printf("freed index %d\n", index);
        _allocation_list[index] = NULL;
    }
    mem = (void*)((uint8_t*)mem - sizeof(int));
#endif
    if (_live_allocations < 1) {
        _srv->fatal("live_allocs < 1", __FILE__, __LINE__, "");
    }
    dec_alloc();
    _srv->free(mem);
    if (_synchronized) { _lock.unlock(); }
}

void *
memory_region::realloc(void *mem, size_t size) {
    if (_synchronized) { _lock.lock(); }
    if (!mem) {
        add_alloc();
    }
#ifdef TRACK_ALLOCATIONS
    size += sizeof(int);
    mem = (void*)((uint8_t*)mem - sizeof(int));
    int index = *(int  *)mem;
#endif
    void *newMem = _srv->realloc(mem, size);
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list[index] != mem) {
        printf("at index %d, found %p, expected %p\n", 
               index, _allocation_list[index], mem);
        printf("realloc: ptr 0x%" PRIxPTR " is not in allocation_list\n",
            (uintptr_t) mem);
        _srv->fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
    else {
        _allocation_list[index] = newMem;
        (*(int*)newMem) = index;
        // printf("realloc: stored %p at index %d, replacing %p\n", 
        //        newMem, index, mem);
    }
#endif
    if (_synchronized) { _lock.unlock(); }
#ifdef TRACK_ALLOCATIONS
    newMem = (void *)((uint8_t*)newMem + sizeof(int));
#endif
    return newMem;
}

void *
memory_region::malloc(size_t size) {
    if (_synchronized) { _lock.lock(); }
    add_alloc();
#ifdef TRACK_ALLOCATIONS
    size += sizeof(int);
#endif
    void *mem = _srv->malloc(size);
#ifdef TRACK_ALLOCATIONS
    int index = _allocation_list.append(mem);
    int *p = (int *)mem;
    *p = index;
    // printf("malloc: stored %p at index %d\n", mem, index);
#endif
    // printf("malloc: ptr 0x%" PRIxPTR " region=%p\n", 
    //        (uintptr_t) mem, this);
    if (_synchronized) { _lock.unlock(); }
#ifdef TRACK_ALLOCATIONS
    mem = (void*)((uint8_t*)mem + sizeof(int));
#endif
    return mem;
}

void *
memory_region::calloc(size_t size) {
    if (_synchronized) { _lock.lock(); }
    add_alloc();
#ifdef TRACK_ALLOCATIONS
    size += sizeof(int);
#endif
    void *mem = _srv->malloc(size);
    memset(mem, 0, size);
#ifdef TRACK_ALLOCATIONS
    int index = _allocation_list.append(mem);
    int *p = (int *)mem;
    *p = index;
    // printf("calloc: stored %p at index %d\n", mem, index);
#endif
    if (_synchronized) { _lock.unlock(); }
#ifdef TRACK_ALLOCATIONS
    mem = (void*)((uint8_t*)mem + sizeof(int));
#endif
    return mem;
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
    _srv->fatal(msg, __FILE__, __LINE__, "%d objects", _live_allocations);
    if (_synchronized) { _lock.unlock(); }
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
