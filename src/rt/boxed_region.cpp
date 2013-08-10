// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "memory_region.h"
#include "boxed_region.h"
#include "rust_globals.h"
#include "rust_env.h"
#include "rust_util.h"

// #define DUMP_BOXED_REGION

rust_opaque_box *boxed_region::malloc(type_desc *td, size_t body_size) {
    size_t total_size = get_box_size(body_size, td->align);
    rust_opaque_box *box =
      (rust_opaque_box*)backing_region->malloc(total_size, "@");
    box->td = td;
    box->ref_count = 1;
    box->prev = NULL;
    box->next = live_allocs;
    if (live_allocs) live_allocs->prev = box;
    live_allocs = box;

    /*LOG(rust_get_current_task(), box,
        "@malloc()=%p with td %p, size %lu==%lu+%lu, "
        "align %lu, prev %p, next %p\n",
        box, td, total_size, sizeof(rust_opaque_box), body_size,
        td->align, box->prev, box->next);*/

    return box;
}

rust_opaque_box *boxed_region::realloc(rust_opaque_box *box,
                                       size_t new_size) {

    size_t total_size = new_size + sizeof(rust_opaque_box);
    rust_opaque_box *new_box =
        (rust_opaque_box*)backing_region->realloc(box, total_size);
    if (new_box->prev) new_box->prev->next = new_box;
    if (new_box->next) new_box->next->prev = new_box;
    if (live_allocs == box) live_allocs = new_box;

    /*LOG(rust_get_current_task(), box,
        "@realloc()=%p with orig=%p, size %lu==%lu+%lu",
        new_box, box, total_size, sizeof(rust_opaque_box), new_size);*/

    return new_box;
}


rust_opaque_box *boxed_region::calloc(type_desc *td, size_t body_size) {
    rust_opaque_box *box = malloc(td, body_size);
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

    /*LOG(rust_get_current_task(), box,
        "@free(%p) with td %p, prev %p, next %p\n",
        box, box->td, box->prev, box->next);*/

    if (box->prev) box->prev->next = box->next;
    if (box->next) box->next->prev = box->prev;
    if (live_allocs == box) live_allocs = box->next;

    if (poison_on_free) {
        memset(box_body(box), 0xab, box->td->size);
    }

    box->prev = NULL;
    box->next = NULL;
    box->td = NULL;

    backing_region->free(box);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
