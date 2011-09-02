// Object stacks, used in lieu of dynamically-sized frames.

#ifndef RUST_OBSTACK_H
#define RUST_OBSTACK_H

#include <utility>

struct rust_obstack_alloc;
struct rust_task;
struct type_desc;

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

class rust_obstack {
    rust_obstack_chunk *chunk;
    rust_task *task;

    // Allocates the given number of bytes in a new chunk.
    void *alloc_new(size_t len, type_desc *tydesc);

public:
    class iterator {
        rust_obstack_chunk *chunk;
        rust_obstack_alloc *alloc;

    public:
        iterator(rust_obstack_chunk *in_chunk)
        : chunk(in_chunk),
          alloc(in_chunk
                ? reinterpret_cast<rust_obstack_alloc *>(in_chunk->data)
                : NULL) {}

        std::pair<const type_desc *,void *> operator*() const;
        iterator &operator++();
        bool operator==(const iterator &other) const;
        bool operator!=(const iterator &other) const;
    };

    rust_obstack(rust_task *in_task) : chunk(NULL), task(in_task) {}
    ~rust_obstack();

    inline iterator begin() const {
        iterator it(chunk);
        return it;
    }

    inline iterator end() const {
        iterator it(NULL);
        return it;
    }

    void *alloc(size_t len, type_desc *tydesc);
    void free(void *ptr);
    void *mark();

    /** Debugging tool: dumps the contents of this obstack to stderr. */
    void dump() const;
};

#endif

