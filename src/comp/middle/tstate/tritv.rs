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

type t = {uncertain: bitv::t, val: bitv::t, nbits: uint};
tag trit { ttrue; tfalse; dont_care; }

fn create_tritv(len: uint) -> t {
    ret {uncertain: bitv::create(len, true),
         val: bitv::create(len, false),
         nbits: len};
}


fn trit_minus(a: trit, b: trit) -> trit {

    /*   2 - anything = 2
         1 - 1 = 2
         1 - 0 is an error
         1 - 2 = 1
         0 - 1 is an error
         0 - anything else - 0
     */
    alt a {
      dont_care. { dont_care }
      ttrue. {
        alt b {
          ttrue. { dont_care }
          tfalse. { ttrue }
           /* internally contradictory, but
              I guess it'll get flagged? */
           dont_care. {
            ttrue
          }
        }
      }
      tfalse. {
        alt b {
          ttrue. { tfalse }
           /* see above comment */
          _ {
            tfalse
          }
        }
      }
    }
}

fn trit_or(a: trit, b: trit) -> trit {
    alt a {
      dont_care. { b }
      ttrue. { ttrue }
      tfalse. {
        alt b {
          ttrue. { dont_care }
           /* FIXME: ?????? */
          _ {
            tfalse
          }
        }
      }
    }
}

// FIXME: This still seems kind of dodgy to me (that is,
// that 1 + ? = 1. But it might work out given that
// all variables start out in a 0 state. Probably I need
// to make it so that all constraints start out in a 0 state
// (we consider a constraint false until proven true), too.
fn trit_and(a: trit, b: trit) -> trit {
    alt a {
      dont_care. { b }
       // also seems wrong for case b = ttrue
      ttrue. {
        alt b {
          dont_care. { ttrue }
           // ??? Seems wrong
          ttrue. {
            ttrue
          }

          // false wins, since if something is uninit
          // on one path, we care
          // (Rationale: it's always safe to assume that
          // a var is uninitialized or that a constraint
          // needs to be re-established)
          tfalse. {
            tfalse
          }
        }
      }

      // Rationale: if it's uninit on one path,
      // we can consider it as uninit on all paths
      tfalse. {
        tfalse
      }
    }
    // if the result is dont_care, that means
    // a and b were both dont_care
}

fn change(changed: bool, old: trit, new: trit) -> bool {
    changed || new != old
}

fn tritv_difference(p1: &t, p2: &t) -> bool {
    let i: uint = 0u;
    assert (p1.nbits == p2.nbits);
    let sz: uint = p1.nbits;
    let changed = false;
    while i < sz {
        let old = tritv_get(p1, i);
        let new = trit_minus(old, tritv_get(p2, i));
        changed = change(changed, old, new);
        tritv_set(i, p1, new);
        i += 1u;
    }
    ret changed;
}

fn tritv_union(p1: &t, p2: &t) -> bool {
    let i: uint = 0u;
    assert (p1.nbits == p2.nbits);
    let sz: uint = p1.nbits;
    let changed = false;
    while i < sz {
        let old = tritv_get(p1, i);
        let new = trit_or(old, tritv_get(p2, i));
        changed = change(changed, old, new);
        tritv_set(i, p1, new);
        i += 1u;
    }
    ret changed;
}

fn tritv_intersect(p1: &t, p2: &t) -> bool {
    let i: uint = 0u;
    assert (p1.nbits == p2.nbits);
    let sz: uint = p1.nbits;
    let changed = false;
    while i < sz {
        let old = tritv_get(p1, i);
        let new = trit_and(old, tritv_get(p2, i));
        changed = change(changed, old, new);
        tritv_set(i, p1, new);
        i += 1u;
    }
    ret changed;
}

fn tritv_get(v: &t, i: uint) -> trit {
    let b1 = bitv::get(v.uncertain, i);
    let b2 = bitv::get(v.val, i);
    assert (!(b1 && b2));
    if b1 { dont_care } else if (b2) { ttrue } else { tfalse }
}

fn tritv_set(i: uint, v: &t, t: trit) -> bool {
    let old = tritv_get(v, i);
    alt t {
      dont_care. {
        bitv::set(v.uncertain, i, true);
        bitv::set(v.val, i, false);
      }
      ttrue. { bitv::set(v.uncertain, i, false); bitv::set(v.val, i, true); }
      tfalse. {
        bitv::set(v.uncertain, i, false);
        bitv::set(v.val, i, false);
      }
    }
    ret change(false, old, t);
}

fn tritv_copy(target: &t, source: &t) -> bool {
    assert (target.nbits == source.nbits);
    let changed =
        !bitv::equal(target.uncertain, source.uncertain) ||
            !bitv::equal(target.val, source.val);
    bitv::copy(target.uncertain, source.uncertain);
    bitv::copy(target.val, source.val);
    ret changed;
}

fn tritv_set_all(v: &t) {
    let i: uint = 0u;
    while i < v.nbits { tritv_set(i, v, ttrue); i += 1u; }
}

fn tritv_clear(v: &t) {
    let i: uint = 0u;
    while i < v.nbits { tritv_set(i, v, dont_care); i += 1u; }
}

fn tritv_clone(v: &t) -> t {
    ret {uncertain: bitv::clone(v.uncertain),
         val: bitv::clone(v.val),
         nbits: v.nbits};
}

fn tritv_doesntcare(v: &t) -> bool {
    let i: uint = 0u;
    while i < v.nbits {
        if tritv_get(v, i) != dont_care { ret false; }
        i += 1u;
    }
    ret true;
}

fn to_vec(v: &t) -> uint[] {
    let i: uint = 0u;
    let rslt: uint[] = ~[];
    while i < v.nbits {
        rslt +=
            ~[alt tritv_get(v, i) {
                dont_care. { 2u }
                ttrue. { 1u }
                tfalse. { 0u }
              }];
        i += 1u;
    }
    ret rslt;
}

fn to_str(v: &t) -> str {
    let i: uint = 0u;
    let rs: str = "";
    while i < v.nbits {
        rs +=
            alt tritv_get(v, i) {
              dont_care. { "?" }
              ttrue. { "1" }
              tfalse. { "0" }
            };
        i += 1u;
    }
    ret rs;
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
