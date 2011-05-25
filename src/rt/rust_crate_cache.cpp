
#include "rust_internal.h"

rust_crate_cache::lib::lib(rust_dom *dom, char const *name)
    : handle(0),
      dom(dom)
{
#if defined(__WIN32__)
    handle = (uintptr_t)LoadLibrary(_T(name));
#else
    handle = (uintptr_t)dlopen(name, RTLD_GLOBAL|RTLD_LAZY);
#endif
    DLOG(dom, cache, "loaded library '%s' as 0x%"  PRIxPTR,
             name, handle);
}

rust_crate_cache::lib::~lib() {
    DLOG(dom, cache, "~rust_crate_cache::lib(0x%" PRIxPTR ")",
             handle);
    if (handle) {
#if defined(__WIN32__)
        FreeLibrary((HMODULE)handle);
#else
        dlclose((void*)handle);
#endif
    }
}

uintptr_t
rust_crate_cache::lib::get_handle() {
    return handle;
}

static inline void
adjust_disp(uintptr_t &disp, const void *oldp, const void *newp)
{
    if (disp) {
        disp += (uintptr_t)oldp;
        disp -= (uintptr_t)newp;
    }
}

type_desc *
rust_crate_cache::get_type_desc(size_t size,
                                size_t align,
                                size_t n_descs,
                                type_desc const **descs)
{
    I(dom, n_descs > 1);
    type_desc *td = NULL;
    size_t keysz = n_descs * sizeof(type_desc*);
    HASH_FIND(hh, this->type_descs, descs, keysz, td);
    if (td) {
        DLOG(dom, cache, "rust_crate_cache::get_type_desc hit");
        return td;
    }
    DLOG(dom, cache, "rust_crate_cache::get_type_desc miss");
    td = (type_desc*) dom->malloc(sizeof(type_desc) + keysz);
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
        DLOG(dom, cache,
                 "rust_crate_cache::descs[%" PRIdPTR "] = 0x%" PRIxPTR,
                 i, descs[i]);
        td->descs[i] = descs[i];
        // FIXME (issue #136):  Below is a miscalculation.
        td->is_stateful |= descs[i]->is_stateful;
    }
    HASH_ADD(hh, this->type_descs, descs, keysz, td);
    return td;
}

rust_crate_cache::rust_crate_cache(rust_dom *dom,
                                   rust_crate const *crate)
    : libs((lib**) dom->calloc(sizeof(lib*) * crate->n_libs)),
      type_descs(NULL),
      crate(crate),
      dom(dom),
      idx(0)
{
    I(dom, libs);
}

void
rust_crate_cache::flush() {
    DLOG(dom, cache, "rust_crate_cache::flush()");

    for (size_t i = 0; i < crate->n_libs; ++i) {
        lib *l = libs[i];
        if (l) {
            DLOG(dom, cache, "rust_crate_cache::flush() deref lib %"
                     PRIdPTR " (rc=%" PRIdPTR ")", i, l->ref_count);
            l->deref();
        }
        libs[i] = NULL;
    }

    while (type_descs) {
        type_desc *d = type_descs;
        HASH_DEL(type_descs, d);
        DLOG(dom, mem, "rust_crate_cache::flush() tydesc %" PRIxPTR, d);
        dom->free(d);
    }
}

rust_crate_cache::~rust_crate_cache()
{
    flush();
    dom->free(libs);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
