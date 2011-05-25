
#include "rust_internal.h"

rust_crate_reader::abbrev::abbrev(rust_dom *dom,
                                  uintptr_t body_off,
                                  size_t body_sz,
                                  uintptr_t tag,
                                  uint8_t has_children) :
  dom(dom),
  body_off(body_off),
  tag(tag),
  has_children(has_children),
  idx(0)
{}

bool
rust_crate_reader::attr::is_numeric() const
{
  switch (form) {
  case DW_FORM_ref_addr:
  case DW_FORM_addr:
  case DW_FORM_data4:
  case DW_FORM_data1:
  case DW_FORM_flag:
    return true;
  default:
    break;
  }
  return false;
}

bool
rust_crate_reader::attr::is_string() const
{
  return form == DW_FORM_string;
}

size_t
rust_crate_reader::attr::get_ssz(rust_dom *dom) const
{
  I(dom, is_string());
  return val.str.sz;
}

char const *
rust_crate_reader::attr::get_str(rust_dom *dom) const
{
  I(dom, is_string());
  return val.str.s;
}

uintptr_t
rust_crate_reader::attr::get_num(rust_dom *dom) const
{
  I(dom, is_numeric());
  return val.num;
}

bool
rust_crate_reader::attr::is_unknown() const {
  return !(is_numeric() || is_string());
}

rust_crate_reader::rust_crate_reader(rust_dom *dom)
  : dom(dom)
{
  DLOG(dom, mem, "crate_reader on crate: 0x%" PRIxPTR, this);
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
