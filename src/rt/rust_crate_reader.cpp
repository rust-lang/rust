
#include "rust_internal.h"

bool
rust_crate_reader::mem_reader::is_ok()
{
  return ok;
}

bool
rust_crate_reader::mem_reader::at_end()
{
  return pos == mem.lim;
}

void
rust_crate_reader::mem_reader::fail()
{
  ok = false;
}

void
rust_crate_reader::mem_reader::reset()
{
  pos = mem.base;
  ok = true;
}

rust_crate_reader::mem_reader::mem_reader(rust_crate::mem_area &m)
  : mem(m),
    ok(true),
    pos(m.base)
{}

size_t
rust_crate_reader::mem_reader::tell_abs()
{
  return pos;
}

size_t
rust_crate_reader::mem_reader::tell_off()
{
  return pos - mem.base;
}

void
rust_crate_reader::mem_reader::seek_abs(uintptr_t p)
{
  if (!ok || p < mem.base || p >= mem.lim)
    ok = false;
  else
    pos = p;
}

void
rust_crate_reader::mem_reader::seek_off(uintptr_t p)
{
  seek_abs(p + mem.base);
}


bool
rust_crate_reader::mem_reader::adv_zstr(size_t sz)
{
  sz = 0;
  while (ok) {
    char c = 0;
    get(c);
    ++sz;
    if (c == '\0')
      return true;
  }
  return false;
}

bool
rust_crate_reader::mem_reader::get_zstr(char const *&c, size_t &sz)
{
  if (!ok)
    return false;
  c = (char const *)(pos);
  return adv_zstr(sz);
}

void
rust_crate_reader::mem_reader::adv(size_t amt)
{
  if (pos < mem.base
      || pos >= mem.lim
      || pos + amt > mem.lim)
    ok = false;
  if (!ok)
    return;
  // mem.DLOG(dom, mem, "adv %d bytes", amt);
  pos += amt;
  ok &= !at_end();
  I(mem.dom, at_end() || (mem.base <= pos && pos < mem.lim));
}


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


rust_crate_reader::abbrev_reader::abbrev_reader
  (rust_crate::mem_area &abbrev_mem)
  : mem_reader(abbrev_mem),
    abbrevs(abbrev_mem.dom)
{
  rust_dom *dom = mem.dom;
  while (is_ok() && !at_end()) {

    // DLOG(dom, dwarf, "reading new abbrev at 0x%" PRIxPTR,
    //          tell_off());

    uintptr_t idx, tag;
    uint8_t has_children = 0;
    get_uleb(idx);
    get_uleb(tag);
    get(has_children);

    uintptr_t attr, form;
    size_t body_off = tell_off();
    while (is_ok() && step_attr_form_pair(attr, form));

    // DLOG(dom, dwarf,
    //         "finished scanning attr/form pairs, pos=0x%"
    //         PRIxPTR ", lim=0x%" PRIxPTR ", is_ok=%d, at_end=%d",
    //        pos, mem.lim, is_ok(), at_end());

    if (is_ok() || at_end()) {
      DLOG(dom, dwarf, "read abbrev: %" PRIdPTR, idx);
      I(dom, idx = abbrevs.length() + 1);
      abbrevs.push(new (dom) abbrev(dom, body_off,
                                    tell_off() - body_off,
                                    tag, has_children));
    }
  }
}

rust_crate_reader::abbrev *
rust_crate_reader::abbrev_reader::get_abbrev(size_t i) {
  i -= 1;
  if (i < abbrevs.length())
    return abbrevs[i];
  return NULL;
}

bool
rust_crate_reader::abbrev_reader::step_attr_form_pair(uintptr_t &attr,
                                                      uintptr_t &form)
{
  attr = 0;
  form = 0;
  // mem.DLOG(dom, dwarf, "reading attr/form pair at 0x%" PRIxPTR,
  //              tell_off());
  get_uleb(attr);
  get_uleb(form);
  // mem.DLOG(dom, dwarf, "attr 0x%" PRIxPTR ", form 0x%" PRIxPTR,
  //              attr, form);
  return ! (attr == 0 && form == 0);
}
rust_crate_reader::abbrev_reader::~abbrev_reader() {
  while (abbrevs.length()) {
    delete abbrevs.pop();
  }
}


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

rust_crate_reader::rust_crate_reader(rust_dom *dom,
                                     rust_crate const *crate)
  : dom(dom),
    abbrev_mem(crate->get_debug_abbrev(dom)),
    abbrevs(abbrev_mem),
    die_mem(crate->get_debug_info(dom))
{
  DLOG(dom, mem, "crate_reader on crate: 0x%" PRIxPTR, this);
  DLOG(dom, mem, "debug_abbrev: 0x%" PRIxPTR, abbrev_mem.base);
  DLOG(dom, mem, "debug_info: 0x%" PRIxPTR, die_mem.base);
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
