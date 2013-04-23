// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef BOXED_REGION_H
#define BOXED_REGION_H

#include <stdlib.h>

struct type_desc;
class memory_region;
struct rust_opaque_box;
struct rust_env;

/* Tracks the data allocated by a particular task in the '@' region.
 * Currently still relies on the standard malloc as a backing allocator, but
 * this could be improved someday if necessary. Every allocation must provide
 * a type descr which describes the payload (what follows the header). */
class boxed_region {
private:
    bool poison_on_free;
    memory_region *backing_region;
    rust_opaque_box *live_allocs;

    size_t align_to(size_t v, size_t align) {
        size_t alignm1 = align - 1;
        v += alignm1;
        v &= ~alignm1;
        return v;
    }

private:
    // private and undefined to disable copying
    boxed_region(const boxed_region& rhs);
    boxed_region& operator=(const boxed_region& rhs);

public:
    boxed_region(memory_region *br, bool poison_on_free)
        : poison_on_free(poison_on_free)
        , backing_region(br)
        , live_allocs(NULL)
    {}

    rust_opaque_box *first_live_alloc() { return live_allocs; }

    rust_opaque_box *malloc(type_desc *td, size_t body_size);
    rust_opaque_box *calloc(type_desc *td, size_t body_size);
    rust_opaque_box *realloc(rust_opaque_box *box, size_t new_size);
    void free(rust_opaque_box *box);
};

#endif /* BOXED_REGION_H */

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
