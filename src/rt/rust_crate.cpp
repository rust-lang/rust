
#include "rust_internal.h"

uintptr_t
rust_crate::get_image_base() const {
  return ((uintptr_t)this + image_base_off);
}

ptrdiff_t
rust_crate::get_relocation_diff() const {
  return ((uintptr_t)this - self_addr);
}

uintptr_t
rust_crate::get_gc_glue() const {
  return ((uintptr_t)this + gc_glue_off);
}

rust_crate::mem_area::mem_area(rust_dom *dom, uintptr_t pos, size_t sz)
  : dom(dom),
    base(pos),
    lim(pos + sz)
{
  DLOG(dom, mem, "new mem_area [0x%" PRIxPTR ",0x%" PRIxPTR "]",
       base, lim);
}

rust_crate::mem_area
rust_crate::get_debug_info(rust_dom *dom) const {
    if (debug_info_off)
        return mem_area(dom,
                        ((uintptr_t)this + debug_info_off),
                        debug_info_sz);
    else
        return mem_area(dom, 0, 0);
}

rust_crate::mem_area
rust_crate::get_debug_abbrev(rust_dom *dom) const {
    if (debug_abbrev_off)
        return mem_area(dom,
                        ((uintptr_t)this + debug_abbrev_off),
                        debug_abbrev_sz);
    else
        return mem_area(dom, 0, 0);
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
