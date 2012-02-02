#include <assert.h>
#include "boxed_region.h"
#include "rust_internal.h"

// #define DUMP_BOXED_REGION

rust_opaque_box *boxed_region::malloc(type_desc *td) {
    size_t header_size = sizeof(rust_opaque_box);
    size_t body_size = td->size;
    size_t body_align = td->align;
    size_t total_size = align_to(header_size, body_align) + body_size;
    rust_opaque_box *box =
      (rust_opaque_box*)backing_region->malloc(total_size, "@");
    box->td = td;
    box->ref_count = 1;
    box->prev = NULL;
    box->next = live_allocs;
    if (live_allocs) live_allocs->prev = box;
    live_allocs = box;

#   ifdef DUMP_BOXED_REGION
    fprintf(stderr, "Allocated box %p with td %p,"
            " size %lu==%lu+%lu, align %lu, prev %p, next %p\n",
            box, td, total_size, header_size, body_size, body_align,
            box->prev, box->next);
#   endif

    return box;
}

rust_opaque_box *boxed_region::calloc(type_desc *td) {
    rust_opaque_box *box = malloc(td);
    memset(box_body(box), 0, td->size);
    return box;
}

void boxed_region::free(rust_opaque_box *box) {
    // This turns out to not be true in various situations,
    // like when we are unwinding after a failure.
    //
    // assert(box->ref_count == 0);

    // This however should always be true.  Helps to detect
    // double frees (kind of).
    assert(box->td != NULL);

#   ifdef DUMP_BOXED_REGION
    fprintf(stderr, "Freed box %p with td %p, prev %p, next %p\n",
            box, box->td, box->prev, box->next);
#   endif

    if (box->prev) box->prev->next = box->next;
    if (box->next) box->next->prev = box->prev;
    if (live_allocs == box) live_allocs = box->next;
    box->prev = NULL;
    box->next = NULL;
    box->td = NULL;
    backing_region->free(box);
}
