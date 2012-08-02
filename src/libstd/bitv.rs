import vec::{to_mut, from_elem};

export bitv;
export union;
export intersect;
export assign;
export clone;
export get;
export equal;
export clear;
export set_all;
export invert;
export difference;
export set;
export is_true;
export is_false;
export to_vec;
export to_str;
export eq_vec;
export methods;

class small_bitv {
    let mut bits: u32;
    new(bits: u32) { self.bits = bits; }
    priv {
        #[inline(always)]
        fn bits_op(right_bits: u32, f: fn(u32, u32) -> u32) -> bool {
            let old_b: u32 = self.bits;
            let new_b = f(old_b, right_bits);
            self.bits = new_b;
            old_b != new_b
        }
    }
    #[inline(always)]
    fn union(s: &small_bitv) -> bool {
        self.bits_op(s.bits, |u1, u2| { u1 | u2 })
    }
    #[inline(always)]
    fn intersect(s: &small_bitv) -> bool {
        self.bits_op(s.bits, |u1, u2| { u1 & u2 })
    }
    #[inline(always)]
    fn become(s: &small_bitv) -> bool {
        let old = self.bits;
        self.bits = s.bits;
        old != self.bits
    }
    #[inline(always)]
    fn difference(s: &small_bitv) -> bool {
        let old = self.bits;
        self.bits &= !s.bits;
        old != self.bits
    }
    #[inline(always)]
    pure fn get(i: uint) -> bool {
        (self.bits & (1 << i)) != 0
    }
    #[inline(always)]
    fn set(i: uint, x: bool) {
        if x {
            self.bits |= 1<<i;
        }
        else {
            self.bits &= !(i as u32);
        }
    }
    #[inline(always)]
    fn equals(b: &small_bitv) -> bool { self.bits == b.bits }
    #[inline(always)]
    fn clear() { self.bits = 0; }
    #[inline(always)]
    fn set_all() { self.bits = !0; }
    #[inline(always)]
    fn is_true() -> bool { self.bits == !0 }
    #[inline(always)]
    fn is_false() -> bool { self.bits == 0 }
    #[inline(always)]
    fn invert() { self.bits = !self.bits; }
}

class big_bitv {
// only mut b/c of clone and lack of other constructor
    let mut storage: ~[mut uint];
    new(-storage: ~[mut uint]) {
        self.storage <- storage;
    }
    priv {
        #[inline(always)]
        fn process(b: &big_bitv, op: fn(uint, uint) -> uint) -> bool {
            let len = b.storage.len();
            assert (self.storage.len() == len);
            let mut changed = false;
            do uint::range(0, len) |i| {
                let w0 = self.storage[i];
                let w1 = b.storage[i];
                let w = op(w0, w1);
                if w0 != w unchecked { changed = true; self.storage[i] = w; };
                true
            };
            changed
        }
    }
    #[inline(always)]
     fn each_storage(op: fn(&uint) -> bool) {
        for uint::range(0, self.storage.len()) |i| {
            let mut w = self.storage[i];
            let b = !op(w);
            self.storage[i] = w;
            if !b { break; }
        }
     }
    #[inline(always)]
    fn invert() { for self.each_storage() |w| { w = !w } }
    #[inline(always)]
    fn union(b: &big_bitv)     -> bool { self.process(b, lor) }
    #[inline(always)]
    fn intersect(b: &big_bitv) -> bool { self.process(b, land) }
    #[inline(always)]
    fn become(b: &big_bitv)    -> bool { self.process(b, right) }
    #[inline(always)]
    fn difference(b: &big_bitv) -> bool {
        self.invert();
        let b = self.intersect(b);
        self.invert();
        b
    }
    #[inline(always)]
    pure fn get(i: uint) -> bool {
        let w = i / uint_bits;
        let b = i % uint_bits;
        let x = 1 & self.storage[w] >> b;
        x == 1
    }
    #[inline(always)]
    fn set(i: uint, x: bool) {
        let w = i / uint_bits;
        let b = i % uint_bits;
        let flag = 1 << b;
        self.storage[w] = if x { self.storage[w] | flag }
                 else { self.storage[w] & !flag };
    }
    #[inline(always)]
    fn equals(b: &big_bitv) -> bool {
        let len = b.storage.len();
        for uint::iterate(0, len) |i| {
            if self.storage[i] != b.storage[i] { return false; }
        }
    }
}

enum a_bitv { big(~big_bitv), small(~small_bitv) }

enum op {union, intersect, assign, difference}

// The bitvector type
class bitv {
    let rep: a_bitv;
    let nbits: uint;

    new(nbits: uint, init: bool) {
        self.nbits = nbits;
        if nbits <= 32 {
          self.rep = small(~small_bitv(if init {!0} else {0}));
        }
        else {
          let s = to_mut(from_elem(nbits / uint_bits + 1,
                                        if init {!0} else {0}));
          self.rep = big(~big_bitv(s));
        };
    }

    priv {
        fn die() -> ! {
            fail ~"Tried to do operation on bit vectors with \
                  different sizes";
        }
        #[inline(always)]
        fn do_op(op: op, other: &bitv) -> bool {
            if self.nbits != other.nbits {
                self.die();
            }
            alt self.rep {
              small(s) {
                alt other.rep {
                  small(s1) {
                    alt op {
                      union      { s.union(s1) }
                      intersect  { s.intersect(s1) }
                      assign     { s.become(s1) }
                      difference { s.difference(s1) }
                    }
                  }
                 big(s1) {
                     self.die();
                 }
              }
            }
            big(s) {
                alt other.rep {
                  small(_) { self.die(); }
                  big(s1) {
                    alt op {
                      union      { s.union(s1) }
                      intersect  { s.intersect(s1) }
                      assign     { s.become(s1) }
                      difference { s.difference(s1) }
                    }
                  }
                }
            }
          }
        }
    }

/**
 * Calculates the union of two bitvectors
 *
 * Sets `self` to the union of `self` and `v1`. Both bitvectors must be
 * the same length. Returns 'true' if `self` changed.
*/
    #[inline(always)]
    fn union(v1: &bitv) -> bool { self.do_op(union, v1) }

/**
 * Calculates the intersection of two bitvectors
 *
 * Sets `self` to the intersection of `self` and `v1`. Both bitvectors must be
 * the same length. Returns 'true' if `self` changed.
*/
    #[inline(always)]
    fn intersect(v1: &bitv) -> bool { self.do_op(intersect, v1) }

/**
 * Assigns the value of `v1` to `self`
 *
 * Both bitvectors must be the same length. Returns `true` if `self` was
 * changed
 */
    #[inline(always)]
    fn assign(v: &bitv) -> bool { self.do_op(assign, v) }

    /// Makes a copy of a bitvector
    #[inline(always)]
    fn clone() -> ~bitv {
        ~alt self.rep {
          small(b) {
            bitv{nbits: self.nbits, rep: small(~small_bitv{bits: b.bits})}
          }
          big(b) {
            let st = to_mut(from_elem(self.nbits / uint_bits + 1, 0));
            let len = st.len();
            for uint::range(0, len) |i| { st[i] = b.storage[i]; };
            bitv{nbits: self.nbits, rep: big(~big_bitv{storage: st})}
          }
        }
    }

    /// Retrieve the value at index `i`
    #[inline(always)]
    pure fn get(i: uint) -> bool {
       assert (i < self.nbits);
       alt self.rep {
         big(b)   { b.get(i) }
         small(s) { s.get(i) }
       }
    }

/**
 * Set the value of a bit at a given index
 *
 * `i` must be less than the length of the bitvector.
 */
    #[inline(always)]
    fn set(i: uint, x: bool) {
      assert (i < self.nbits);
      alt self.rep {
        big(b) { b.set(i, x); }
        small(s) { s.set(i, x); }
      }
    }

/**
 * Compares two bitvectors
 *
 * Both bitvectors must be the same length. Returns `true` if both bitvectors
 * contain identical elements.
 */
    #[inline(always)]
    fn equal(v1: bitv) -> bool {
      if self.nbits != v1.nbits { return false; }
      alt self.rep {
        small(b) {
          alt v1.rep {
            small(b1) { b.equals(b1) }
            _ { false }
          }
        }
        big(s) {
          alt v1.rep {
            big(s1) {
              s.equals(s1)
            }
            small(_) { return false; }
          }
        }
      }
    }

    /// Set all bits to 0
    #[inline(always)]
    fn clear() {
        alt self.rep {
          small(b) { b.clear(); }
          big(s) {
            for s.each_storage() |w| { w = 0u }
          }
        }
    }

    /// Set all bits to 1
    #[inline(always)]
    fn set_all() {
      alt self.rep {
        small(b) { b.set_all(); }
        big(s) {
          for s.each_storage() |w| { w = !0u } }
      }
    }

    /// Invert all bits
    #[inline(always)]
    fn invert() {
      alt self.rep {
        small(b) { b.invert(); }
        big(s) {
          for s.each_storage() |w| { w = !w } }
      }
    }

/**
 * Calculate the difference between two bitvectors
 *
 * Sets each element of `v0` to the value of that element minus the element
 * of `v1` at the same index. Both bitvectors must be the same length.
 *
 * Returns `true` if `v0` was changed.
 */
   #[inline(always)]
    fn difference(v: ~bitv) -> bool { self.do_op(difference, v) }

        /// Returns true if all bits are 1
    #[inline(always)]
    fn is_true() -> bool {
      alt self.rep {
        small(b) { b.is_true() }
        _ {
          for self.each() |i| { if !i { return false; } }
          true
        }
      }
    }

    #[inline(always)]
    fn each(f: fn(bool) -> bool) {
        let mut i = 0;
        while i < self.nbits {
            if !f(self.get(i)) { break; }
            i += 1;
        }
    }

    /// Returns true if all bits are 0

    fn is_false() -> bool {
      alt self.rep {
        small(b) { b.is_false() }
        big(_) {
          for self.each() |i| { if i { return false; } }
          true
        }
      }
    }

    fn init_to_vec(i: uint) -> uint {
      return if self.get(i) { 1 } else { 0 };
    }

/**
 * Converts `self` to a vector of uint with the same length.
 *
 * Each uint in the resulting vector has either value 0u or 1u.
 */
    fn to_vec() -> ~[uint] {
      let sub = |x| self.init_to_vec(x);
      return vec::from_fn::<uint>(self.nbits, sub);
    }

/**
 * Converts `self` to a string.
 *
 * The resulting string has the same length as `self`, and each
 * character is either '0' or '1'.
 */
     fn to_str() -> ~str {
       let mut rs = ~"";
       for self.each() |i| { if i { rs += "1"; } else { rs += "0"; } };
       rs
     }


/**
 * Compare a bitvector to a vector of uint
 *
 * The uint vector is expected to only contain the values 0u and 1u. Both the
 * bitvector and vector must have the same length
 */
     fn eq_vec(v: ~[uint]) -> bool {
       assert self.nbits == v.len();
       let mut i = 0;
       while i < self.nbits {
           let w0 = self.get(i);
           let w1 = v[i];
           if !w0 && w1 != 0u || w0 && w1 == 0u { return false; }
           i = i + 1;
       }
       true
     }

    fn ones(f: fn(uint) -> bool) {
        for uint::range(0, self.nbits) |i| {
            if self.get(i) {
                if !f(i) { break }
            }
        }
    }

} // end of bitv class

const uint_bits: uint = 32u + (1u << 32u >> 27u);

pure fn lor(w0: uint, w1: uint) -> uint { return w0 | w1; }

pure fn land(w0: uint, w1: uint) -> uint { return w0 & w1; }

pure fn right(_w0: uint, w1: uint) -> uint { return w1; }

impl extensions of ops::index<uint,bool> for bitv {
    pure fn index(&&i: uint) -> bool {
        self.get(i)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_to_str() {
        let zerolen = bitv(0u, false);
        assert zerolen.to_str() == ~"";

        let eightbits = bitv(8u, false);
        assert eightbits.to_str() == ~"00000000";
    }

    #[test]
    fn test_0_elements() {
        let mut act;
        let mut exp;
        act = bitv(0u, false);
        exp = vec::from_elem::<uint>(0u, 0u);
        assert act.eq_vec(exp);
    }

    #[test]
    fn test_1_element() {
        let mut act;
        act = bitv(1u, false);
        assert act.eq_vec(~[0u]);
        act = bitv(1u, true);
        assert act.eq_vec(~[1u]);
    }

    #[test]
    fn test_10_elements() {
        let mut act;
        // all 0

        act = bitv(10u, false);
        assert (act.eq_vec(~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(10u, true);
        assert (act.eq_vec(~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(10u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        assert (act.eq_vec(~[1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(10u, false);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        act.set(8u, true);
        act.set(9u, true);
        assert (act.eq_vec(~[0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(10u, false);
        act.set(0u, true);
        act.set(3u, true);
        act.set(6u, true);
        act.set(9u, true);
        assert (act.eq_vec(~[1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 1u]));
    }

    #[test]
    fn test_31_elements() {
        let mut act;
        // all 0

        act = bitv(31u, false);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(31u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(31u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(31u, false);
        act.set(16u, true);
        act.set(17u, true);
        act.set(18u, true);
        act.set(19u, true);
        act.set(20u, true);
        act.set(21u, true);
        act.set(22u, true);
        act.set(23u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(31u, false);
        act.set(24u, true);
        act.set(25u, true);
        act.set(26u, true);
        act.set(27u, true);
        act.set(28u, true);
        act.set(29u, true);
        act.set(30u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(31u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u]));
    }

    #[test]
    fn test_32_elements() {
        let mut act;
        // all 0

        act = bitv(32u, false);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(32u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(32u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(32u, false);
        act.set(16u, true);
        act.set(17u, true);
        act.set(18u, true);
        act.set(19u, true);
        act.set(20u, true);
        act.set(21u, true);
        act.set(22u, true);
        act.set(23u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(32u, false);
        act.set(24u, true);
        act.set(25u, true);
        act.set(26u, true);
        act.set(27u, true);
        act.set(28u, true);
        act.set(29u, true);
        act.set(30u, true);
        act.set(31u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(32u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        act.set(31u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 1u]));
    }

    #[test]
    fn test_33_elements() {
        let mut act;
        // all 0

        act = bitv(33u, false);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(33u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(33u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(33u, false);
        act.set(16u, true);
        act.set(17u, true);
        act.set(18u, true);
        act.set(19u, true);
        act.set(20u, true);
        act.set(21u, true);
        act.set(22u, true);
        act.set(23u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(33u, false);
        act.set(24u, true);
        act.set(25u, true);
        act.set(26u, true);
        act.set(27u, true);
        act.set(28u, true);
        act.set(29u, true);
        act.set(30u, true);
        act.set(31u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 0u]));
        // mixed

        act = bitv(33u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        act.set(31u, true);
        act.set(32u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 1u, 1u]));
    }

    #[test]
    fn test_equal_differing_sizes() {
        let v0 = bitv(10u, false);
        let v1 = bitv(11u, false);
        assert !v0.equal(v1);
    }

    #[test]
    fn test_equal_greatly_differing_sizes() {
        let v0 = bitv(10u, false);
        let v1 = bitv(110u, false);
        assert !v0.equal(v1);
    }
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
