
#include "rust_internal.h"
#include <algorithm>

type_desc *
rust_crate_cache::get_type_desc(size_t size,
                                size_t align,
                                size_t n_descs,
                                type_desc const **descs,
                                uintptr_t n_obj_params)
{
    I(sched, n_descs > 1);
    type_desc *td = NULL;
    size_t keysz = n_descs * sizeof(type_desc*);
    HASH_FIND(hh, this->type_descs, descs, keysz, td);
    if (td) {
        DLOG(sched, cache, "rust_crate_cache::get_type_desc hit");

        // FIXME: This is a gross hack.
        td->n_obj_params = std::max(td->n_obj_params, n_obj_params);

        return td;
    }
    DLOG(sched, cache, "rust_crate_cache::get_type_desc miss");
    td = (type_desc*) sched->kernel->malloc(sizeof(type_desc) + keysz,
                                            "crate cache typedesc");
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
    }
    td->n_obj_params = n_obj_params;
    td->n_params = n_descs - 1;
    HASH_ADD(hh, this->type_descs, descs, keysz, td);
    return td;
}

void**
rust_crate_cache::get_dict(size_t n_fields, void** dict) {
    rust_hashable_dict *found = NULL;
    uintptr_t key = 0;
    for (size_t i = 0; i < n_fields; ++i) key ^= (uintptr_t)dict[i];
    size_t keysz = sizeof(uintptr_t);
    HASH_FIND(hh, this->dicts, &key, keysz, found);
    if (found) { printf("found!\n"); return &(found->fields[0]); }
    printf("not found\n");
    size_t dictsz = n_fields * sizeof(void*);
    found = (rust_hashable_dict*)
        sched->kernel->malloc(keysz + sizeof(UT_hash_handle) + dictsz,
                              "crate cache dict");
    if (!found) return NULL;
    found->key = key;
    void** retptr = &(found->fields[0]);
    memcpy(retptr, dict, dictsz);
    HASH_ADD(hh, this->dicts, key, keysz, found);
    return retptr;
}

rust_crate_cache::rust_crate_cache(rust_scheduler *sched)
    : type_descs(NULL),
      dicts(NULL),
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
    while (dicts) {
        rust_hashable_dict *d = dicts;
        HASH_DEL(dicts, d);
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
