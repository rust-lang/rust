// Object stacks, used in lieu of dynamically-sized frames.

#ifndef RUST_OBSTACK_H
#define RUST_OBSTACK_H

struct rust_obstack_chunk;
struct rust_task;
struct type_desc;

class rust_obstack {
    rust_obstack_chunk *chunk;
    rust_task *task;

    // Allocates the given number of bytes in a new chunk.
    void *alloc_new(size_t len, type_desc *tydesc);

public:
    rust_obstack(rust_task *in_task) : chunk(NULL), task(in_task) {}
    ~rust_obstack();

    void *alloc(size_t len, type_desc *tydesc);
    void free(void *ptr);
    void *mark();
};

#endif

