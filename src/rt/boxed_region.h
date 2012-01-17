#ifndef BOXED_REGION_H
#define BOXED_REGION_H

#include <stdlib.h>

struct type_desc;
class memory_region;
struct rust_opaque_box;

/* Tracks the data allocated by a particular task in the '@' region.
 * Currently still relies on the standard malloc as a backing allocator, but
 * this could be improved someday if necessary. Every allocation must provide
 * a type descr which describes the payload (what follows the header). */
class boxed_region {
private:
    memory_region *backing_region;
    rust_opaque_box *live_allocs;

    size_t align_to(size_t v, size_t align) {
        size_t alignm1 = align - 1;
        v += alignm1;
        v &= ~alignm1;
        return v;
    }

public:
    boxed_region(memory_region *br)
        : backing_region(br)
        , live_allocs(NULL)
    {}

    rust_opaque_box *first_live_alloc() { return live_allocs; }

    rust_opaque_box *malloc(type_desc *td);
    rust_opaque_box *calloc(type_desc *td);
    void free(rust_opaque_box *box);
};

#endif /* BOXED_REGION_H */
