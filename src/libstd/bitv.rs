// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(deprecated_mode)];

use core::ops;
use core::prelude::*;
use core::uint;
use core::vec::{cast_to_mut, from_elem};
use core::vec;

struct SmallBitv {
    /// only the lowest nbits of this value are used. the rest is undefined.
    mut bits: u32
}

fn SmallBitv(bits: u32) -> SmallBitv {
    SmallBitv {bits: bits}
}

/// a mask that has a 1 for each defined bit in a small_bitv, assuming n bits
#[inline(always)]
fn small_mask(nbits: uint) -> u32 {
    (1 << nbits) - 1
}

impl SmallBitv {

    #[inline(always)]
    fn bits_op(right_bits: u32, nbits: uint, f: fn(u32, u32) -> u32) -> bool {
        let mask = small_mask(nbits);
        let old_b: u32 = self.bits;
        let new_b = f(old_b, right_bits);
        self.bits = new_b;
        mask & old_b != mask & new_b
    }

    #[inline(always)]
    fn union(s: &SmallBitv, nbits: uint) -> bool {
        self.bits_op(s.bits, nbits, |u1, u2| u1 | u2)
    }

    #[inline(always)]
    fn intersect(s: &SmallBitv, nbits: uint) -> bool {
        self.bits_op(s.bits, nbits, |u1, u2| u1 & u2)
    }

    #[inline(always)]
    fn become(s: &SmallBitv, nbits: uint) -> bool {
        self.bits_op(s.bits, nbits, |_u1, u2| u2)
    }

    #[inline(always)]
    fn difference(s: &SmallBitv, nbits: uint) -> bool {
        self.bits_op(s.bits, nbits, |u1, u2| u1 & !u2)
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
            self.bits &= !(1<<i as u32);
        }
    }

    #[inline(always)]
    fn equals(b: &SmallBitv, nbits: uint) -> bool {
        let mask = small_mask(nbits);
        mask & self.bits == mask & b.bits
    }

    #[inline(always)]
    fn clear() { self.bits = 0; }

    #[inline(always)]
    fn set_all() { self.bits = !0; }

    #[inline(always)]
    fn is_true(nbits: uint) -> bool {
        small_mask(nbits) & !self.bits == 0
    }

    #[inline(always)]
    fn is_false(nbits: uint) -> bool {
        small_mask(nbits) & self.bits == 0
    }

    #[inline(always)]
    fn invert() { self.bits = !self.bits; }

}

struct BigBitv {
    // only mut b/c of clone and lack of other constructor
    mut storage: ~[uint]
}

fn BigBitv(storage: ~[uint]) -> BigBitv {
    BigBitv {storage: move storage}
}

/**
 * a mask that has a 1 for each defined bit in the nth element of a big_bitv,
 * assuming n bits.
 */
#[inline(always)]
fn big_mask(nbits: uint, elem: uint) -> uint {
    let rmd = nbits % uint_bits;
    let nelems = nbits/uint_bits + if rmd == 0 {0} else {1};

    if elem < nelems - 1 || rmd == 0 {
        !0
    } else {
        (1 << rmd) - 1
    }
}

impl BigBitv {

    #[inline(always)]
    fn process(b: &BigBitv, nbits: uint, op: fn(uint, uint) -> uint) -> bool {
        let len = b.storage.len();
        assert (self.storage.len() == len);
        let mut changed = false;
        do uint::range(0, len) |i| {
            let mask = big_mask(nbits, i);
            let w0 = self.storage[i] & mask;
            let w1 = b.storage[i] & mask;
            let w = op(w0, w1) & mask;
            if w0 != w {
                unsafe {
                    changed = true;
                    self.storage[i] = w;
                }
            }
            true
        }
        changed
    }

    #[inline(always)]
     fn each_storage(op: fn(v: &mut uint) -> bool) {
        for uint::range(0, self.storage.len()) |i| {
            let mut w = self.storage[i];
            let b = !op(&mut w);
            self.storage[i] = w;
            if !b { break; }
        }
     }

    #[inline(always)]
    fn invert() { for self.each_storage() |w| { *w = !*w } }

    #[inline(always)]
    fn union(b: &BigBitv, nbits: uint) -> bool {
        self.process(b, nbits, lor)
    }

    #[inline(always)]
    fn intersect(b: &BigBitv, nbits: uint) -> bool {
        self.process(b, nbits, land)
    }

    #[inline(always)]
    fn become(b: &BigBitv, nbits: uint) -> bool {
        self.process(b, nbits, right)
    }

    #[inline(always)]
    fn difference(b: &BigBitv, nbits: uint) -> bool {
        self.process(b, nbits, difference)
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
    fn equals(b: &BigBitv, nbits: uint) -> bool {
        let len = b.storage.len();
        for uint::iterate(0, len) |i| {
            let mask = big_mask(nbits, i);
            if mask & self.storage[i] != mask & b.storage[i] {
                return false;
            }
        }
    }

}

enum BitvVariant { Big(~BigBitv), Small(~SmallBitv) }

enum Op {Union, Intersect, Assign, Difference}

// The bitvector type
pub struct Bitv {
    rep: BitvVariant,
    nbits: uint
}

pub fn Bitv (nbits: uint, init: bool) -> Bitv {
    let rep = if nbits <= 32 {
        Small(~SmallBitv(if init {!0} else {0}))
    }
    else {
        let nelems = nbits/uint_bits +
                     if nbits % uint_bits == 0 {0} else {1};
        let elem = if init {!0} else {0};
        let s = cast_to_mut(from_elem(nelems, elem));
        Big(~BigBitv(move s))
    };
    Bitv {rep: move rep, nbits: nbits}
}

priv impl Bitv {

    fn die() -> ! {
        fail ~"Tried to do operation on bit vectors with different sizes";
    }

    #[inline(always)]
    fn do_op(op: Op, other: &Bitv) -> bool {
        if self.nbits != other.nbits {
            self.die();
        }
        match self.rep {
          Small(ref s) => match other.rep {
            Small(ref s1) => match op {
              Union      => s.union(*s1,      self.nbits),
              Intersect  => s.intersect(*s1,  self.nbits),
              Assign     => s.become(*s1,     self.nbits),
              Difference => s.difference(*s1, self.nbits)
            },
            Big(_) => self.die()
          },
          Big(ref s) => match other.rep {
            Small(_) => self.die(),
            Big(ref s1) => match op {
              Union      => s.union(*s1,      self.nbits),
              Intersect  => s.intersect(*s1,  self.nbits),
              Assign     => s.become(*s1,     self.nbits),
              Difference => s.difference(*s1, self.nbits)
            }
          }
        }
    }

}

impl Bitv {

    /**
     * Calculates the union of two bitvectors
     *
     * Sets `self` to the union of `self` and `v1`. Both bitvectors must be
     * the same length. Returns 'true' if `self` changed.
    */
    #[inline(always)]
    fn union(v1: &Bitv) -> bool { self.do_op(Union, v1) }

    /**
     * Calculates the intersection of two bitvectors
     *
     * Sets `self` to the intersection of `self` and `v1`. Both bitvectors
     * must be the same length. Returns 'true' if `self` changed.
    */
    #[inline(always)]
    fn intersect(v1: &Bitv) -> bool { self.do_op(Intersect, v1) }

    /**
     * Assigns the value of `v1` to `self`
     *
     * Both bitvectors must be the same length. Returns `true` if `self` was
     * changed
     */
    #[inline(always)]
    fn assign(v: &Bitv) -> bool { self.do_op(Assign, v) }

    /// Retrieve the value at index `i`
    #[inline(always)]
    pure fn get(i: uint) -> bool {
       assert (i < self.nbits);
       match self.rep {
         Big(ref b)   => b.get(i),
         Small(ref s) => s.get(i)
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
      match self.rep {
        Big(ref b)   => b.set(i, x),
        Small(ref s) => s.set(i, x)
      }
    }

    /**
     * Compares two bitvectors
     *
     * Both bitvectors must be the same length. Returns `true` if both
     * bitvectors contain identical elements.
     */
    #[inline(always)]
    fn equal(v1: &Bitv) -> bool {
      if self.nbits != v1.nbits { return false; }
      match self.rep {
        Small(ref b) => match v1.rep {
          Small(ref b1) => b.equals(*b1, self.nbits),
          _ => false
        },
        Big(ref s) => match v1.rep {
          Big(ref s1) => s.equals(*s1, self.nbits),
          Small(_) => return false
        }
      }
    }

    /// Set all bits to 0
    #[inline(always)]
    fn clear() {
        match self.rep {
          Small(ref b) => b.clear(),
          Big(ref s) => for s.each_storage() |w| { *w = 0u }
        }
    }

    /// Set all bits to 1
    #[inline(always)]
    fn set_all() {
      match self.rep {
        Small(ref b) => b.set_all(),
        Big(ref s) => for s.each_storage() |w| { *w = !0u } }
    }

    /// Invert all bits
    #[inline(always)]
    fn invert() {
      match self.rep {
        Small(ref b) => b.invert(),
        Big(ref s) => for s.each_storage() |w| { *w = !*w } }
    }

    /**
     * Calculate the difference between two bitvectors
     *
     * Sets each element of `v0` to the value of that element minus the
     * element of `v1` at the same index. Both bitvectors must be the same
     * length.
     *
     * Returns `true` if `v0` was changed.
     */
    #[inline(always)]
    fn difference(v: &Bitv) -> bool { self.do_op(Difference, v) }

    /// Returns true if all bits are 1
    #[inline(always)]
    fn is_true() -> bool {
      match self.rep {
        Small(ref b) => b.is_true(self.nbits),
        _ => {
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
      match self.rep {
        Small(ref b) => b.is_false(self.nbits),
        Big(_) => {
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
        vec::from_fn(self.nbits, |x| self.init_to_vec(x))
    }

    /**
     * Organise the bits into bytes, such that the first bit in the
     * bitv becomes the high-order bit of the first byte. If the
     * size of the bitv is not a multiple of 8 then trailing bits
     * will be filled-in with false/0
     */
    fn to_bytes() -> ~[u8] {

        fn bit (bitv: &Bitv, byte: uint, bit: uint) -> u8 {
            let offset = byte * 8 + bit;
            if offset >= bitv.nbits {
                0
            } else {
                bitv[offset] as u8 << (7 - bit)
            }
        }

        let len = self.nbits/8 +
                  if self.nbits % 8 == 0 { 0 } else { 1 };
        vec::from_fn(len, |i|
            bit(&self, i, 0) |
            bit(&self, i, 1) |
            bit(&self, i, 2) |
            bit(&self, i, 3) |
            bit(&self, i, 4) |
            bit(&self, i, 5) |
            bit(&self, i, 6) |
            bit(&self, i, 7)
        )
    }

    /**
     * Transform self into a [bool] by turning each bit into a bool
     */
    fn to_bools() -> ~[bool] {
        vec::from_fn(self.nbits, |i| self[i])
    }

    /**
     * Converts `self` to a string.
     *
     * The resulting string has the same length as `self`, and each
     * character is either '0' or '1'.
     */
     fn to_str() -> ~str {
       let mut rs = ~"";
       for self.each() |i| { if i { rs += ~"1"; } else { rs += ~"0"; } };
       rs
     }


    /**
     * Compare a bitvector to a vector of uint
     *
     * The uint vector is expected to only contain the values 0u and 1u. Both
     * the bitvector and vector must have the same length
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

}

impl Bitv: Clone {
    /// Makes a copy of a bitvector
    #[inline(always)]
    fn clone(&self) -> Bitv {
        match self.rep {
          Small(ref b) => {
            Bitv{nbits: self.nbits, rep: Small(~SmallBitv{bits: b.bits})}
          }
          Big(ref b) => {
            let st = cast_to_mut(from_elem(self.nbits / uint_bits + 1, 0));
            let len = st.len();
            for uint::range(0, len) |i| { st[i] = b.storage[i]; };
            Bitv{nbits: self.nbits, rep: Big(~BigBitv{storage: move st})}
          }
        }
    }

}

/**
 * Transform a byte-vector into a bitv. Each byte becomes 8 bits,
 * with the most significant bits of each byte coming first. Each
 * bit becomes true if equal to 1 or false if equal to 0.
 */
pub fn from_bytes(bytes: &[u8]) -> Bitv {
    from_fn(bytes.len() * 8, |i| {
        let b = bytes[i / 8] as uint;
        let offset = i % 8;
        b >> (7 - offset) & 1 == 1
    })
}

/**
 * Transform a [bool] into a bitv by converting each bool into a bit.
 */
pub fn from_bools(bools: &[bool]) -> Bitv {
    from_fn(bools.len(), |i| bools[i])
}

/**
 * Create a bitv of the specified length where the value at each
 * index is f(index).
 */
pub fn from_fn(len: uint, f: fn(index: uint) -> bool) -> Bitv {
    let bitv = Bitv(len, false);
    for uint::range(0, len) |i| {
        bitv.set(i, f(i));
    }
    move bitv
}

const uint_bits: uint = 32u + (1u << 32u >> 27u);

pure fn lor(w0: uint, w1: uint) -> uint { return w0 | w1; }

pure fn land(w0: uint, w1: uint) -> uint { return w0 & w1; }

pure fn difference(w0: uint, w1: uint) -> uint { return w0 & !w1; }

pure fn right(_w0: uint, w1: uint) -> uint { return w1; }

impl Bitv: ops::Index<uint,bool> {
    pure fn index(&self, i: uint) -> bool {
        self.get(i)
    }
}

#[cfg(test)]
mod tests {
    use core::prelude::*;

    use bitv::*;
    use bitv;

    use core::uint;
    use core::vec;

    #[test]
    pub fn test_to_str() {
        let zerolen = Bitv(0u, false);
        assert zerolen.to_str() == ~"";

        let eightbits = Bitv(8u, false);
        assert eightbits.to_str() == ~"00000000";
    }

    #[test]
    pub fn test_0_elements() {
        let mut act;
        let mut exp;
        act = Bitv(0u, false);
        exp = vec::from_elem::<uint>(0u, 0u);
        assert act.eq_vec(exp);
    }

    #[test]
    pub fn test_1_element() {
        let mut act;
        act = Bitv(1u, false);
        assert act.eq_vec(~[0u]);
        act = Bitv(1u, true);
        assert act.eq_vec(~[1u]);
    }

    #[test]
    pub fn test_2_elements() {
        let b = bitv::Bitv(2, false);
        b.set(0, true);
        b.set(1, false);
        assert b.to_str() == ~"10";
    }

    #[test]
    pub fn test_10_elements() {
        let mut act;
        // all 0

        act = Bitv(10u, false);
        assert (act.eq_vec(~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = Bitv(10u, true);
        assert (act.eq_vec(~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = Bitv(10u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        assert (act.eq_vec(~[1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = Bitv(10u, false);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        act.set(8u, true);
        act.set(9u, true);
        assert (act.eq_vec(~[0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = Bitv(10u, false);
        act.set(0u, true);
        act.set(3u, true);
        act.set(6u, true);
        act.set(9u, true);
        assert (act.eq_vec(~[1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 1u]));
    }

    #[test]
    pub fn test_31_elements() {
        let mut act;
        // all 0

        act = Bitv(31u, false);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = Bitv(31u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = Bitv(31u, false);
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

        act = Bitv(31u, false);
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

        act = Bitv(31u, false);
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

        act = Bitv(31u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u]));
    }

    #[test]
    pub fn test_32_elements() {
        let mut act;
        // all 0

        act = Bitv(32u, false);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = Bitv(32u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = Bitv(32u, false);
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

        act = Bitv(32u, false);
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

        act = Bitv(32u, false);
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

        act = Bitv(32u, false);
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
    pub fn test_33_elements() {
        let mut act;
        // all 0

        act = Bitv(33u, false);
        assert (act.eq_vec(
                       ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = Bitv(33u, true);
        assert (act.eq_vec(
                       ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = Bitv(33u, false);
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

        act = Bitv(33u, false);
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

        act = Bitv(33u, false);
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

        act = Bitv(33u, false);
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
    pub fn test_equal_differing_sizes() {
        let v0 = Bitv(10u, false);
        let v1 = Bitv(11u, false);
        assert !v0.equal(&v1);
    }

    #[test]
    pub fn test_equal_greatly_differing_sizes() {
        let v0 = Bitv(10u, false);
        let v1 = Bitv(110u, false);
        assert !v0.equal(&v1);
    }

    #[test]
    pub fn test_equal_sneaky_small() {
        let a = bitv::Bitv(1, false);
        a.set(0, true);

        let b = bitv::Bitv(1, true);
        b.set(0, true);

        assert a.equal(&b);
    }

    #[test]
    pub fn test_equal_sneaky_big() {
        let a = bitv::Bitv(100, false);
        for uint::range(0, 100) |i| {
            a.set(i, true);
        }

        let b = bitv::Bitv(100, true);
        for uint::range(0, 100) |i| {
            b.set(i, true);
        }

        assert a.equal(&b);
    }

    #[test]
    pub fn test_from_bytes() {
        let bitv = from_bytes([0b10110110, 0b00000000, 0b11111111]);
        let str = ~"10110110" + ~"00000000" + ~"11111111";
        assert bitv.to_str() == str;
    }

    #[test]
    pub fn test_to_bytes() {
        let bv = Bitv(3, true);
        bv.set(1, false);
        assert bv.to_bytes() == ~[0b10100000];

        let bv = Bitv(9, false);
        bv.set(2, true);
        bv.set(8, true);
        assert bv.to_bytes() == ~[0b00100000, 0b10000000];
    }

    #[test]
    pub fn test_from_bools() {
        assert from_bools([true, false, true, true]).to_str() == ~"1011";
    }

    #[test]
    pub fn test_to_bools() {
        let bools = ~[false, false, true, false, false, true, true, false];
        assert from_bytes([0b00100110]).to_bools() == bools;
    }

    #[test]
    pub fn test_small_difference() {
      let b1 = Bitv(3, false);
      let b2 = Bitv(3, false);
      b1.set(0, true);
      b1.set(1, true);
      b2.set(1, true);
      b2.set(2, true);
      assert b1.difference(&b2);
      assert b1[0];
      assert !b1[1];
      assert !b1[2];
    }

    #[test]
    pub fn test_big_difference() {
      let b1 = Bitv(100, false);
      let b2 = Bitv(100, false);
      b1.set(0, true);
      b1.set(40, true);
      b2.set(40, true);
      b2.set(80, true);
      assert b1.difference(&b2);
      assert b1[0];
      assert !b1[40];
      assert !b1[80];
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
