// Object stacks, used in lieu of dynamically-sized frames.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdint.h>

#include "rust_internal.h"
#include "rust_obstack.h"
#include "rust_shape.h"
#include "rust_task.h"

// ISAAC, let go of max()!
#ifdef max
#undef max
#endif

//const size_t DEFAULT_CHUNK_SIZE = 4096;
const size_t DEFAULT_CHUNK_SIZE = 500000;
const size_t DEFAULT_ALIGNMENT = 16;

// A single type-tagged allocation in a chunk.
struct rust_obstack_alloc {
    size_t len;
    const type_desc *tydesc;
    uint8_t data[];

    rust_obstack_alloc(size_t in_len, const type_desc *in_tydesc)
    : len(in_len), tydesc(in_tydesc) {}
};

// A contiguous set of allocations.
struct rust_obstack_chunk {
    rust_obstack_chunk *prev;
    size_t size;
    size_t alen;
    size_t pad;
    uint8_t data[];

    rust_obstack_chunk(rust_obstack_chunk *in_prev, size_t in_size)
    : prev(in_prev), size(in_size), alen(0) {}

    void *alloc(size_t len, type_desc *tydesc);
    bool free(void *ptr);
    void *mark();
};

void *
rust_obstack_chunk::alloc(size_t len, type_desc *tydesc) {
    alen = align_to(alen, DEFAULT_ALIGNMENT);

    if (sizeof(rust_obstack_alloc) + len > size - alen) {
        DPRINT("Not enough space, len=%lu!\n", len);
        assert(0);      // FIXME
        return NULL;    // Not enough space.
    }

    rust_obstack_alloc *a = new(data + alen) rust_obstack_alloc(len, tydesc);
    alen += sizeof(*a) + len;
    return &a->data;
}

bool
rust_obstack_chunk::free(void *ptr) {
    uint8_t *p = (uint8_t *)ptr;
    if (p < data || p > data + size)
        return false;
    assert(p <= data + alen);
    alen = (size_t)(p - data);
    return true;
}

void *
rust_obstack_chunk::mark() {
    return data + alen;
}

// Allocates the given number of bytes in a new chunk.
void *
rust_obstack::alloc_new(size_t len, type_desc *tydesc) {
    size_t chunk_size = std::max(sizeof(rust_obstack_alloc) + len,
                                 DEFAULT_CHUNK_SIZE);
    void *ptr = task->malloc(sizeof(chunk) + chunk_size, "obstack");
    DPRINT("making new chunk at %p, len %lu\n", ptr, chunk_size);
    chunk = new(ptr) rust_obstack_chunk(chunk, chunk_size);
    return chunk->alloc(len, tydesc);
}

rust_obstack::~rust_obstack() {
    while (chunk) {
        rust_obstack_chunk *prev = chunk->prev;
        task->free(chunk);
        chunk = prev;
    }
}

void *
rust_obstack::alloc(size_t len, type_desc *tydesc) {
    if (!chunk)
        return alloc_new(len, tydesc);

    DPRINT("alloc sz %u", (uint32_t)len);

    void *ptr = chunk->alloc(len, tydesc);
    ptr = ptr ? ptr : alloc_new(len, tydesc);

    return ptr;
}

void
rust_obstack::free(void *ptr) {
    if (!ptr)
        return;

    DPRINT("free ptr %p\n", ptr);

    assert(chunk);
    while (!chunk->free(ptr)) {
        DPRINT("deleting chunk at %p\n", chunk);
        rust_obstack_chunk *prev = chunk->prev;
        task->free(chunk);
        chunk = prev;
        assert(chunk);
    }
}

void *
rust_obstack::mark() {
    return chunk ? chunk->mark() : NULL;
}

