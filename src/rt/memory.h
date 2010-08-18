/*
 *
 */

#ifndef MEMORY_H
#define MEMORY_H


inline void *operator new(size_t size, void *mem) {
    return mem;
}

inline void *operator new(size_t size, rust_dom *dom) {
    return dom->malloc(size, memory_region::LOCAL);
}

inline void *operator new[](size_t size, rust_dom *dom) {
    return dom->malloc(size, memory_region::LOCAL);
}

inline void *operator new(size_t size, rust_dom &dom) {
    return dom.malloc(size, memory_region::LOCAL);
}

inline void *operator new[](size_t size, rust_dom &dom) {
    return dom.malloc(size, memory_region::LOCAL);
}

inline void *operator new(size_t size, rust_dom *dom,
    memory_region::memory_region_type type) {
    return dom->malloc(size, type);
}

inline void *operator new[](size_t size, rust_dom *dom,
    memory_region::memory_region_type type) {
    return dom->malloc(size, type);
}

inline void *operator new(size_t size, rust_dom &dom,
    memory_region::memory_region_type type) {
    return dom.malloc(size, type);
}

inline void *operator new[](size_t size, rust_dom &dom,
    memory_region::memory_region_type type) {
    return dom.malloc(size, type);
}

inline void operator delete(void *mem, rust_dom *dom) {
    dom->free(mem, memory_region::LOCAL);
    return;
}

inline void operator delete(void *mem, rust_dom *dom,
    memory_region::memory_region_type type) {
    dom->free(mem, type);
    return;
}

#endif /* MEMORY_H */
