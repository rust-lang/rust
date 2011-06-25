import std::bitv;

export t;
export create_tritv;
export tritv_clone;
export tritv_set;
export to_vec;
export trit;
export dont_care;
export ttrue;
export tfalse;
export tritv_get;
export tritv_set_all;
export tritv_difference;
export tritv_union;
export tritv_intersect;
export tritv_copy;
export tritv_clear;
export tritv_doesntcare;
export to_str;

/* for a fixed index: 
   10 = "this constraint may or may not be true after execution"
   01 = "this constraint is definitely true"
   00 = "this constraint is definitely false"
   11 should never appear
 FIXME: typestate precondition (uncertain and val must
 have the same length; 11 should never appear in a given position)
*/

type t = rec(bitv::t uncertain, bitv::t val, uint nbits);
tag trit {
    ttrue;
    tfalse;
    dont_care;
}

fn create_tritv(uint len) -> t {
  ret rec(uncertain=bitv::create(len, true),
          val=bitv::create(len, false),
          nbits=len);
}


fn trit_minus(trit a, trit b) -> trit {
    /*   2 - anything = 2
         1 - 1 = 2
         1 - 0 is an error
         1 - 2 = 1
         0 - 1 is an error
         0 - anything else - 0
     */
  alt (a) {
    case (dont_care) { dont_care }
    case (ttrue) {
      alt (b) {
        case (ttrue)     { dont_care }
        case (tfalse)    { ttrue } /* internally contradictory, but
                                      I guess it'll get flagged? */
        case (dont_care) { ttrue }
      }
    }
    case (tfalse) {
      alt (b) {
        case (ttrue) { tfalse } /* see above comment */
        case (_)     { tfalse }
      }
    }
  }
}

fn trit_or(trit a, trit b) -> trit {
  alt (a) {
    case (dont_care) { b }
    case (ttrue)     { ttrue }
    case (tfalse)    {
      alt (b) {
        case (ttrue)  { dont_care } /* FIXME: ?????? */
        case (_)      { tfalse }
      }
    }
  }
}

fn trit_and(trit a, trit b) -> trit {
  alt (a) {
    case (dont_care) { dont_care }
    case (ttrue)     {
      alt (b) {
        case (dont_care) { dont_care }
        case (ttrue)     { ttrue }
        case (tfalse)    { dont_care } // ???
      }
    }
    case (tfalse) { tfalse }
  }
}

fn change(bool changed, trit old, trit new) -> bool { changed || new != old }

fn tritv_difference(&t p1, &t p2) -> bool {
    let uint i = 0u;
    assert (p1.nbits == p2.nbits);
    let uint sz = p1.nbits;
    auto changed = false;
    while (i < sz) {
      auto old = tritv_get(p1, i);
      auto new = trit_minus(old, tritv_get(p2, i));
      changed = change(changed, old, new);
      tritv_set(i, p1, new);
      i += 1u;
    }
    ret changed;
}

fn tritv_union(&t p1, &t p2) -> bool {
    let uint i = 0u;
    assert (p1.nbits == p2.nbits);
    let uint sz = p1.nbits;
    auto changed = false;
    while (i < sz) {
      auto old = tritv_get(p1, i);
      auto new = trit_or(old, tritv_get(p2, i));
      changed = change(changed, old, new);
      tritv_set(i, p1, new);
      i += 1u;
    }
    ret changed;
}

fn tritv_intersect(&t p1, &t p2) -> bool {
    let uint i = 0u;
    assert (p1.nbits == p2.nbits);
    let uint sz = p1.nbits;
    auto changed = false;
    while (i < sz) {
      auto old = tritv_get(p1, i);
      auto new = trit_and(old, tritv_get(p2, i));
      changed = change(changed, old, new);
      tritv_set(i, p1, new);
      i += 1u;
    }
    ret changed;
}

fn tritv_get(&t v, uint i) -> trit {
  auto b1 = bitv::get(v.uncertain, i);
  auto b2 = bitv::get(v.val, i);
  assert (! (b1 && b2));
  if (b1)      { dont_care }
  else if (b2) { ttrue }
  else         { tfalse}
}
 
fn tritv_set(uint i, &t v, trit t) -> bool {
  auto old = tritv_get(v, i);
  alt (t) {
    case (dont_care) {
      bitv::set(v.uncertain, i, true);
      bitv::set(v.val, i, false);
    }
    case (ttrue) {
      bitv::set(v.uncertain, i, false);
      bitv::set(v.val, i, true);
    }
    case (tfalse) {
      bitv::set(v.uncertain, i, false);
      bitv::set(v.val, i, false);
    }
  }
  ret change(false, old, t);
}

fn tritv_copy(&t target, &t source) -> bool {
  let uint i = 0u;
  assert (target.nbits == source.nbits);
  auto changed = false;
  auto oldunc;
  auto newunc;
  auto oldval;
  auto newval;
  while (i < target.nbits) {
    oldunc = bitv::get(target.uncertain, i);
    newunc = bitv::get(source.uncertain, i);
    oldval = bitv::get(target.val, i);
    newval = bitv::get(source.val, i);
    bitv::set(target.uncertain, i, newunc);
    changed = changed || (oldunc && !newunc);
    bitv::set(target.val, i, newval);
    changed = changed || (oldval && !newval);
    i += 1u;
  }
  ret changed;
}

fn tritv_set_all(&t v) {
  let uint i = 0u;
  while (i < v.nbits) {
    tritv_set(i, v, ttrue);
    i += 1u;
  }
}

fn tritv_clear(&t v) {
  let uint i = 0u;
  while (i < v.nbits) {
    tritv_set(i, v, dont_care);
    i += 1u;
  }
}

fn tritv_clone(&t v) -> t {
  ret rec(uncertain=bitv::clone(v.uncertain),
          val=bitv::clone(v.val),
          nbits=v.nbits);
}

fn tritv_doesntcare(&t v) -> bool {
  let uint i = 0u;
  while (i < v.nbits) {
    if (tritv_get(v, i) != dont_care) {
      ret false;
    }
    i += 1u;
  }
  ret true;
}

fn to_vec(&t v) -> vec[uint] {
  let uint i = 0u;
  let vec[uint] rslt = [];
  while (i < v.nbits) {
    rslt += [alt (tritv_get(v, i)) {
        case (dont_care) { 2u }
        case (ttrue)     { 1u }
        case (tfalse)    { 0u } }];
    i += 1u;
  }
  ret rslt;
}

fn to_str(&t v) -> str {
  let uint i = 0u;
  let str res = "";
  while (i < v.nbits) {
    res += alt (tritv_get(v, i)) {
        case (dont_care) { "?" }
        case (ttrue)     { "1" }
        case (tfalse)    { "0" } };
    i += 1u;
  }
  ret res;
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
