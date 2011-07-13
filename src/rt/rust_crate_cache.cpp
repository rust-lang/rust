
#include "rust_internal.h"

type_desc *
rust_crate_cache::get_type_desc(size_t size,
                                size_t align,
                                size_t n_descs,
                                type_desc const **descs)
{
    I(sched, n_descs > 1);
    type_desc *td = NULL;
    size_t keysz = n_descs * sizeof(type_desc*);
    HASH_FIND(hh, this->type_descs, descs, keysz, td);
    if (td) {
        DLOG(sched, cache, "rust_crate_cache::get_type_desc hit");
        return td;
    }
    DLOG(sched, cache, "rust_crate_cache::get_type_desc miss");
    td = (type_desc*) sched->kernel->malloc(sizeof(type_desc) + keysz);
    if (!td)
        return NULL;
    // By convention, desc 0 is the root descriptor.
    // but we ignore the size and alignment of it and use the
    // passed-in, computed values.
    memcpy(td, descs[0], sizeof(type_desc));
    td->first_param = &td->descs[1];
    td->size = size;
    td->align = align;
    for (size_t i = 0; i < n_descs; ++i) {
        DLOG(sched, cache,
                 "rust_crate_cache::descs[%" PRIdPTR "] = 0x%" PRIxPTR,
                 i, descs[i]);
        td->descs[i] = descs[i];
        // FIXME (issue #136):  Below is a miscalculation.
        td->is_stateful |= descs[i]->is_stateful;
    }
    HASH_ADD(hh, this->type_descs, descs, keysz, td);
    return td;
}

rust_crate_cache::rust_crate_cache(rust_scheduler *sched)
    : type_descs(NULL),
      sched(sched),
      idx(0)
{
}

void
rust_crate_cache::flush() {
    DLOG(sched, cache, "rust_crate_cache::flush()");

    while (type_descs) {
        type_desc *d = type_descs;
        HASH_DEL(type_descs, d);
        DLOG(sched, mem, "rust_crate_cache::flush() tydesc %" PRIxPTR, d);
        sched->kernel->free(d);
    }
}

rust_crate_cache::~rust_crate_cache()
{
    flush();
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
