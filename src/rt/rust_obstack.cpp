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

#undef DPRINT
#define DPRINT(fmt, ...)

const size_t DEFAULT_CHUNK_SIZE = 128;
const size_t MAX_CHUNK_SIZE = (1024*64);
const size_t DEFAULT_ALIGNMENT = 16;

// A single type-tagged allocation in a chunk.
struct rust_obstack_alloc {
    size_t len;
    const type_desc *tydesc;
    uint32_t pad0;  // FIXME: x86-specific
    uint32_t pad1;
    uint8_t data[];

    rust_obstack_alloc(size_t in_len, const type_desc *in_tydesc)
    : len(in_len), tydesc(in_tydesc) {}
};

void *
rust_obstack_chunk::alloc(size_t len, type_desc *tydesc) {
    alen = align_to(alen, DEFAULT_ALIGNMENT);

    if (sizeof(rust_obstack_alloc) + len > size - alen) {
        DPRINT("Not enough space, len=%lu!\n", len);
        return NULL;    // Not enough space.
    }

    rust_obstack_alloc *a = new(data + alen) rust_obstack_alloc(len, tydesc);
    alen += sizeof(*a) + len;
    memset(a->data, '\0', len); // FIXME: For GC.
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
    size_t default_chunk_size = DEFAULT_CHUNK_SIZE;
    if (chunk) {
	default_chunk_size = std::min(chunk->size * 2, MAX_CHUNK_SIZE);
    }

    size_t chunk_size = std::max(sizeof(rust_obstack_alloc) + len,
                                 default_chunk_size);
    void *ptr = task->malloc(sizeof(rust_obstack_chunk) + chunk_size,
			     "obstack");
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

    void *ptr = chunk->alloc(len, tydesc);
    ptr = ptr ? ptr : alloc_new(len, tydesc);

    return ptr;
}

void
rust_obstack::free(void *ptr) {
    if (!ptr)
        return;

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


// Iteration over self-describing obstacks

std::pair<const type_desc *,void *>
rust_obstack::iterator::operator*() const {
    return std::make_pair(alloc->tydesc, alloc->data);
}

rust_obstack::iterator &
rust_obstack::iterator::operator++() {
    uint8_t *adata = align_to(alloc->data + alloc->len, DEFAULT_ALIGNMENT);
    alloc = reinterpret_cast<rust_obstack_alloc *>(adata);
    if (reinterpret_cast<uint8_t *>(alloc) >= chunk->data + chunk->alen) {
        // We reached the end of this chunk; go on to the next one.
        chunk = chunk->prev;
        if (chunk)
            alloc = reinterpret_cast<rust_obstack_alloc *>(chunk->data);
        else
            alloc = NULL;
    }
    return *this;
}

bool
rust_obstack::iterator::operator==(const rust_obstack::iterator &other)
        const {
    return chunk == other.chunk && alloc == other.alloc;
}

bool
rust_obstack::iterator::operator!=(const rust_obstack::iterator &other)
        const {
    return !(*this == other);
}


// Debugging

void
rust_obstack::dump() const {
    iterator b = begin(), e = end();
    while (b != e) {
        std::pair<const type_desc *,void *> data = *b;
        uint8_t *dp = reinterpret_cast<uint8_t *>(data.second);

        shape::arena arena;
        shape::type_param *params =
            shape::type_param::from_tydesc_and_data(data.first, dp, arena);
        shape::log log(task, true, data.first->shape, params,
                       data.first->shape_tables, dp, std::cerr);
        log.walk();
        std::cerr << "\n";

        ++b;
    }

    std::cerr << "end of dynastack dump\n";
}

