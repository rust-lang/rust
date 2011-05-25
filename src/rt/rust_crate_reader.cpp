
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

rust_crate_reader::rdr_sess::rdr_sess(die_reader *rdr) : rdr(rdr)
{
  I(rdr->mem.dom, !rdr->in_use);
  rdr->in_use = true;
}

rust_crate_reader::rdr_sess::~rdr_sess()
{
  rdr->in_use = false;
}

rust_crate_reader::die::die(die_reader *rdr, uintptr_t off)
  : rdr(rdr),
    off(off),
    using_rdr(false)
{
  rust_dom *dom = rdr->mem.dom;
  rdr_sess use(rdr);

  rdr->reset();
  rdr->seek_off(off);
  if (!rdr->is_ok()) {
    ab = NULL;
    return;
  }
  size_t ab_idx;
  rdr->get_uleb(ab_idx);
  if (!ab_idx) {
    ab = NULL;
    DLOG(dom, dwarf, "DIE <0x%" PRIxPTR "> (null)", off);
  } else {
    ab = rdr->abbrevs.get_abbrev(ab_idx);
    if (!ab) {
        DLOG(dom, dwarf, "  bad abbrev number: 0x%"
                 PRIxPTR, ab_idx);
        rdr->fail();
    } else {
        DLOG(dom, dwarf, "DIE <0x%" PRIxPTR "> abbrev 0x%"
                 PRIxPTR, off, ab_idx);
        DLOG(dom, dwarf, "  tag 0x%x, has children: %d",
                 ab->tag, ab->has_children);
    }
  }
}

bool
rust_crate_reader::die::is_null() const
{
  return ab == NULL;
}

bool
rust_crate_reader::die::has_children() const
{
  return (!is_null()) && ab->has_children;
}

dw_tag
rust_crate_reader::die::tag() const
{
  if (is_null())
    return (dw_tag) (-1);
  return (dw_tag) ab->tag;
}

bool
rust_crate_reader::die::start_attrs() const
{
  if (is_null())
    return false;
  rdr->reset();
  rdr->seek_off(off + 1);
  rdr->abbrevs.reset();
  rdr->abbrevs.seek_off(ab->body_off);
  return rdr->is_ok();
}

bool
rust_crate_reader::die::step_attr(attr &a) const
{
  uintptr_t ai, fi;
  if (rdr->abbrevs.step_attr_form_pair(ai, fi) && rdr->is_ok()) {
    a.at = (dw_at)ai;
    a.form = (dw_form)fi;

    uint32_t u32 = 0;
    uint8_t u8 = 0;

    switch (a.form) {
    case DW_FORM_string:
      return rdr->get_zstr(a.val.str.s, a.val.str.sz);
      break;

    case DW_FORM_ref_addr:
      I(rdr->mem.dom, sizeof(uintptr_t) == 4);
    case DW_FORM_addr:
    case DW_FORM_data4:
      rdr->get(u32);
      a.val.num = (uintptr_t)u32;
      return rdr->is_ok() || rdr->at_end();
      break;

    case DW_FORM_data1:
    case DW_FORM_flag:
      rdr->get(u8);
      a.val.num = u8;
      return rdr->is_ok() || rdr->at_end();
      break;

    case DW_FORM_block1:
      rdr->get(u8);
      rdr->adv(u8);
      return rdr->is_ok() || rdr->at_end();
      break;

    case DW_FORM_block4:
      rdr->get(u32);
      rdr->adv(u32);
      return rdr->is_ok() || rdr->at_end();
      break;

    case DW_FORM_udata:
      rdr->get_uleb(u32);
      return rdr->is_ok() || rdr->at_end();
      break;

    default:
      DLOG(rdr->mem.dom, dwarf, "  unknown dwarf form: 0x%"
                        PRIxPTR, a.form);
      rdr->fail();
      break;
    }
  }
  return false;
}

bool
rust_crate_reader::die::find_str_attr(dw_at at, char const *&c)
{
  rdr_sess use(rdr);
  if (is_null())
    return false;
  if (start_attrs()) {
    attr a;
    while (step_attr(a)) {
      if (a.at == at && a.is_string()) {
        c = a.get_str(rdr->mem.dom);
        return true;
      }
    }
  }
  return false;
}

bool
rust_crate_reader::die::find_num_attr(dw_at at, uintptr_t &n)
{
  rdr_sess use(rdr);
  if (is_null())
    return false;
  if (start_attrs()) {
    attr a;
    while (step_attr(a)) {
      if (a.at == at && a.is_numeric()) {
        n = a.get_num(rdr->mem.dom);
        return true;
      }
    }
  }
  return false;
}

bool
rust_crate_reader::die::is_transparent()
{
  // "semantically transparent" DIEs are those with
  // children that serve to structure the tree but have
  // tags that don't reflect anything in the rust-module
  // name hierarchy.
  switch (tag()) {
  case DW_TAG_compile_unit:
  case DW_TAG_lexical_block:
    return (has_children());
  default:
    break;
  }
  return false;
}

bool
rust_crate_reader::die::find_child_by_name(char const *c,
                                                       die &child,
                                                       bool exact)
{
  rust_dom *dom = rdr->mem.dom;
  I(dom, has_children());
  I(dom, !is_null());

  for (die ch = next(); !ch.is_null(); ch = ch.next_sibling()) {
    char const *ac;
    if (!exact && ch.is_transparent()) {
      if (ch.find_child_by_name(c, child, exact)) {
        return true;
      }
    }
    else if (ch.find_str_attr(DW_AT_name, ac)) {
      if (strcmp(ac, c) == 0) {
        child = ch;
        return true;
      }
    }
  }
  return false;
}

bool
rust_crate_reader::die::find_child_by_tag(dw_tag tag, die &child)
{
  rust_dom *dom = rdr->mem.dom;
  I(dom, has_children());
  I(dom, !is_null());

  for (child = next(); !child.is_null();
       child = child.next_sibling()) {
    if (child.tag() == tag)
      return true;
  }
  return false;
}

rust_crate_reader::die
rust_crate_reader::die::next() const
{
  rust_dom *dom = rdr->mem.dom;

  if (is_null()) {
    rdr->seek_off(off + 1);
    return die(rdr, rdr->tell_off());
  }

  {
    rdr_sess use(rdr);
    if (start_attrs()) {
        attr a;
        while (step_attr(a)) {
            I(dom, !(a.is_numeric() && a.is_string()));
            if (a.is_numeric())
                DLOG(dom, dwarf, "  attr num: 0x%"
                         PRIxPTR, a.get_num(dom));
            else if (a.is_string())
                DLOG(dom, dwarf, "  attr str: %s",
                         a.get_str(dom));
            else
                DLOG(dom, dwarf, "  attr ??:");
        }
    }
  }
  return die(rdr, rdr->tell_off());
}

rust_crate_reader::die
rust_crate_reader::die::next_sibling() const
{
  // FIXME: use DW_AT_sibling, when present.
  if (has_children()) {
    // DLOG(rdr->mem.dom, dwarf, "+++ children of die 0x%"
    //                   PRIxPTR, off);
    die child = next();
    while (!child.is_null())
      child = child.next_sibling();
    // DLOG(rdr->mem.dom, dwarf, "--- children of die 0x%"
    //                   PRIxPTR, off);
    return child.next();
  } else {
    return next();
  }
}


rust_crate_reader::die
rust_crate_reader::die_reader::first_die()
{
  reset();
  seek_off(cu_base
           + sizeof(dwarf_vers)
           + sizeof(cu_abbrev_off)
           + sizeof(sizeof_addr));
  return die(this, tell_off());
}

void
rust_crate_reader::die_reader::dump()
{
  rust_dom *dom = mem.dom;
  die d = first_die();
  while (!d.is_null())
    d = d.next_sibling();
  I(dom, d.is_null());
  I(dom, d.off == mem.lim - mem.base);
}


rust_crate_reader::die_reader::die_reader(rust_crate::mem_area &die_mem,
                              abbrev_reader &abbrevs)
  : mem_reader(die_mem),
    abbrevs(abbrevs),
    cu_unit_length(0),
    cu_base(0),
    dwarf_vers(0),
    cu_abbrev_off(0),
    sizeof_addr(0),
    in_use(false)
{
  rust_dom *dom = mem.dom;

  rdr_sess use(this);

  get(cu_unit_length);
  cu_base = tell_off();

  get(dwarf_vers);
  get(cu_abbrev_off);
  get(sizeof_addr);

  if (is_ok()) {
    DLOG(dom, dwarf, "new root CU at 0x%" PRIxPTR, die_mem.base);
    DLOG(dom, dwarf, "CU unit length: %" PRId32, cu_unit_length);
    DLOG(dom, dwarf, "dwarf version: %" PRId16, dwarf_vers);
    DLOG(dom, dwarf, "CU abbrev off: %" PRId32, cu_abbrev_off);
    DLOG(dom, dwarf, "size of address: %" PRId8, sizeof_addr);
    I(dom, sizeof_addr == sizeof(uintptr_t));
    I(dom, dwarf_vers >= 2);
    I(dom, cu_base + cu_unit_length == die_mem.lim - die_mem.base);
  } else {
    DLOG(dom, dwarf, "failed to read root CU header");
  }
}

rust_crate_reader::die_reader::~die_reader() {
}


rust_crate_reader::rust_crate_reader(rust_dom *dom,
                                     rust_crate const *crate)
  : dom(dom),
    abbrev_mem(crate->get_debug_abbrev(dom)),
    abbrevs(abbrev_mem),
    die_mem(crate->get_debug_info(dom)),
    dies(die_mem, abbrevs)
{
  DLOG(dom, mem, "crate_reader on crate: 0x%" PRIxPTR, this);
  DLOG(dom, mem, "debug_abbrev: 0x%" PRIxPTR, abbrev_mem.base);
  DLOG(dom, mem, "debug_info: 0x%" PRIxPTR, die_mem.base);
  // For now, perform diagnostics only.
  dies.dump();
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
