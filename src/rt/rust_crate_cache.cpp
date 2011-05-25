
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



rust_crate_cache::c_sym::c_sym(rust_dom *dom, lib *library, char const *name)
    : val(0),
      library(library),
      dom(dom)
{
    library->ref();
    uintptr_t handle = library->get_handle();
    if (handle) {
#if defined(__WIN32__)
        val = (uintptr_t)GetProcAddress((HMODULE)handle, _T(name));
#else
        val = (uintptr_t)dlsym((void*)handle, name);
#endif
        DLOG(dom, cache, "resolved symbol '%s' to 0x%"  PRIxPTR,
                 name, val);
    } else {
        DLOG_ERR(dom, cache, "unresolved symbol '%s', null lib handle\n"
                             "(did you omit a -L flag?)", name);
    }
}

rust_crate_cache::c_sym::~c_sym() {
    DLOG(dom, cache,
            "~rust_crate_cache::c_sym(0x%" PRIxPTR ")", val);
    library->deref();
}

uintptr_t
rust_crate_cache::c_sym::get_val() {
    return val;
}



rust_crate_cache::rust_sym::rust_sym(rust_dom *dom,
                                     rust_crate const *curr_crate,
                                     c_sym *crate_sym,
                                     char const **path)
    : val(0),
      crate_sym(crate_sym),
      dom(dom)
{
    crate_sym->ref();
    typedef rust_crate_reader::die die;
    rust_crate const *crate = (rust_crate*)crate_sym->get_val();
    if (!crate) {
        DLOG_ERR(dom, cache, "failed to resolve symbol, null crate symbol");
        return;
    }
    rust_crate_reader rdr(dom, crate);
    bool found_root = false;
    bool found_leaf = false;
    for (die d = rdr.dies.first_die();
         !(found_root || d.is_null());
         d = d.next_sibling()) {

        die t1 = d;
        die t2 = d;
        for (char const **c = crate_rel(curr_crate, path);
             (*c
              && !t1.is_null()
              && t1.find_child_by_name(crate_rel(curr_crate, *c), t2));
             ++c, t1=t2) {
            DLOG(dom, dwarf, "matched die <0x%"  PRIxPTR
                    ">, child '%s' = die<0x%" PRIxPTR ">",
                    t1.off, crate_rel(curr_crate, *c), t2.off);
            found_root = found_root || true;
            if (!*(c+1) && t2.find_num_attr(DW_AT_low_pc, val)) {
                DLOG(dom, dwarf, "found relative address: 0x%"  PRIxPTR, val);
                DLOG(dom, dwarf, "plus image-base 0x%"  PRIxPTR,
                     crate->get_image_base());
                val += crate->get_image_base();
                found_leaf = true;
                break;
            }
        }
        if (found_root || found_leaf)
            break;
    }
    if (found_leaf) {
        DLOG(dom, cache, "resolved symbol to 0x%"  PRIxPTR, val);
    } else {
        DLOG_ERR(dom, cache, "failed to resolve symbol");
    }
}

rust_crate_cache::rust_sym::~rust_sym() {
    DLOG(dom, cache,
             "~rust_crate_cache::rust_sym(0x%" PRIxPTR ")", val);
    crate_sym->deref();
}

uintptr_t
rust_crate_cache::rust_sym::get_val() {
    return val;
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
    : rust_syms((rust_sym**)
                dom->calloc(sizeof(rust_sym*) * crate->n_rust_syms)),
      c_syms((c_sym**) dom->calloc(sizeof(c_sym*) * crate->n_c_syms)),
      libs((lib**) dom->calloc(sizeof(lib*) * crate->n_libs)),
      type_descs(NULL),
      crate(crate),
      dom(dom),
      idx(0)
{
    I(dom, rust_syms);
    I(dom, c_syms);
    I(dom, libs);
}

void
rust_crate_cache::flush() {
    DLOG(dom, cache, "rust_crate_cache::flush()");
    for (size_t i = 0; i < crate->n_rust_syms; ++i) {
        rust_sym *s = rust_syms[i];
        if (s) {
            DLOG(dom, cache,
                     "rust_crate_cache::flush() deref rust_sym %"
                     PRIdPTR " (rc=%" PRIdPTR ")", i, s->ref_count);
            s->deref();
        }
        rust_syms[i] = NULL;
    }

    for (size_t i = 0; i < crate->n_c_syms; ++i) {
        c_sym *s = c_syms[i];
        if (s) {
            DLOG(dom, cache,
                     "rust_crate_cache::flush() deref c_sym %"
                     PRIdPTR " (rc=%" PRIdPTR ")", i, s->ref_count);
            s->deref();
        }
        c_syms[i] = NULL;
    }

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
    dom->free(rust_syms);
    dom->free(c_syms);
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
