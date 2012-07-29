import std::bitv::*;

export t;
export create_tritv;
export trit, tfalse, ttrue, dont_care;

/* for a fixed index:
   10 = "this constraint may or may not be true after execution"
   01 = "this constraint is definitely true"
   00 = "this constraint is definitely false"
   11 should never appear
 FIXME (#2178): typestate precondition (uncertain and val must
 have the same length; 11 should never appear in a given position)
 (except we're not putting typestate constraints in the compiler, as
 per discussion at).
*/

enum trit { ttrue, tfalse, dont_care, }

class t {
    // Shouldn't be mut; instead we should have a different
    // constructor that takes two bitvs
    let mut uncertain: bitv;
    let mut val: bitv;
    let nbits: uint;
    // next two should be private (#2297)
    fn set_uncertain(-b: bitv) {
        self.uncertain <- b;
    }
    fn set_val(-b: bitv) {
        self.val <- b;
    }
    fn clone() -> t {
        let rs = t(self.nbits);
        let r = self.uncertain.clone();
        rs.set_uncertain(r);
        let r1 = self.val.clone();
        rs.set_val(r1);
        rs
    }
    fn difference(p: t) -> bool {
        assert (self.nbits == p.nbits);
        let mut changed = false;
        for uint::range(0, p.nbits) |i| {
           let old = p.get(i);
           let newv = minus(old, p.get(i));
           changed = change(changed, old, newv);
           self.set(i, newv);
        };
        changed
    }
    pure fn get(i: uint) -> trit {
        let b1 = self.uncertain.get(i);
        let b2 = self.val.get(i);
        assert (!(b1 && b2));
        if b1 { dont_care } else if b2 { ttrue } else { tfalse }
    }
    pure fn set(i: uint, t: trit) -> bool {
        let old = self.get(i);
        alt t {
          dont_care {
            self.uncertain.set(i, true);
            self.val.set(i, false);
          }
          ttrue {
            self.uncertain.set(i, false);
            self.val.set(i, true);
          }
          tfalse {
            self.uncertain.set(i, false);
            self.val.set(i, false);
          }
        }
        change(false, old, t)
    }

    fn set_all() {
      for uint::range(0u, self.nbits) |i| {
         self.set(i, ttrue);
      }
    }

    fn clear() {
      for uint::range(0, self.nbits) |i| {
         self.set(i, dont_care);
      }
    }

    fn kill() {
       for uint::range(0, self.nbits) |i| {
           self.set(i, dont_care);
       }
    }

    fn doesntcare() -> bool {
        for uint::range(0, self.nbits) |i| {
           if self.get(i) != dont_care { ret false; }
        }
        true
    }

    fn to_vec() -> ~[uint] {
      let mut rslt: ~[uint] = ~[];
      for uint::range(0, self.nbits) |i| {
        vec::push(rslt,
                  alt self.get(i) {
                      dont_care { 2 }
                      ttrue     { 1 }
                      tfalse    { 0 }
                  });
      };
      rslt
    }

    fn to_str() -> str {
       let mut rs: str = "";
       for uint::range(0, self.nbits) |i| {
        rs +=
            alt self.get(i) {
              dont_care { "?" }
              ttrue { "1" }
              tfalse { "0" }
            };
       };
       rs
    }

    fn intersect(p: t) -> bool {
      assert (self.nbits == p.nbits);
      let mut changed = false;
      for uint::range(0, self.nbits) |i| {
        let old = self.get(i);
        let newv = trit_and(old, p.get(i));
        changed = change(changed, old, newv);
        self.set(i, newv);
       }
      ret changed;
    }

    fn become(source: t) -> bool {
      assert (self.nbits == source.nbits);
      let changed = !self.uncertain.equal(source.uncertain) ||
          !self.val.equal(source.val);
      self.uncertain.assign(source.uncertain);
      self.val.assign(source.val);
      changed
    }

    fn union(p: t) -> bool {
        assert (self.nbits == p.nbits);
        let mut changed = false;
        for uint::range(0, self.nbits) |i| {
           let old = self.get(i);
           let newv = trit_or(old, p.get(i));
           changed = change(changed, old, newv);
           self.set(i, newv);
        }
        ret changed;
    }

    new(len: uint) {
        self.uncertain = mk_bitv(len, true);
        self.val = mk_bitv(len, false);
        self.nbits = len;
    }
}

fn create_tritv(len: uint) -> t { t(len) }


fn minus(a: trit, b: trit) -> trit {

    /*   2 - anything = 2
         1 - 1 = 2
         1 - 0 is an error
         1 - 2 = 1
         0 - 1 is an error
         0 - anything else - 0
     */
    alt a {
      dont_care { dont_care }
      ttrue {
        alt b {
          ttrue { dont_care }
          tfalse { ttrue }
          /* internally contradictory, but
             I guess it'll get flagged? */
          dont_care {
            ttrue
          }
        }
      }
      tfalse {
        alt b {
          ttrue { tfalse }
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
      dont_care { b }
      ttrue { ttrue }
      tfalse {
        alt b {
          ttrue { dont_care }
          /* FIXME (#2538): ??????
             Again, unit tests would help here
           */
          _ {
            tfalse
          }
        }
      }
    }
}

// FIXME (#2538): This still seems kind of dodgy to me (that is,
// that 1 + ? = 1. But it might work out given that
// all variables start out in a 0 state. Probably I need
// to make it so that all constraints start out in a 0 state
// (we consider a constraint false until proven true), too.
fn trit_and(a: trit, b: trit) -> trit {
    alt a {
      dont_care { b }
      // also seems wrong for case b = ttrue
      ttrue {
        alt b {
          dont_care { ttrue }
          // ??? Seems wrong
          ttrue {
            ttrue
          }
          // false wins, since if something is uninit
          // on one path, we care
          // (Rationale: it's always safe to assume that
          // a var is uninitialized or that a constraint
          // needs to be re-established)
          tfalse {
            tfalse
          }
        }
      }
      // Rationale: if it's uninit on one path,
      // we can consider it as uninit on all paths
      tfalse {
        tfalse
      }
    }
    // if the result is dont_care, that means
    // a and b were both dont_care
}

pure fn change(changed: bool, old: trit, newv: trit) -> bool {
    changed || newv != old
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
