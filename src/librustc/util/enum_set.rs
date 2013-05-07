// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core;

#[deriving(Eq, IterBytes)]
pub struct EnumSet<E> {
    bits: uint
}

pub trait CLike {
    pub fn to_uint(&self) -> uint;
    pub fn from_uint(uint) -> Self;
}

fn bit<E:CLike>(e: E) -> uint {
    1 << e.to_uint()
}

pub impl<E:CLike> EnumSet<E> {
    fn empty() -> EnumSet<E> {
        EnumSet {bits: 0}
    }

    fn is_empty(&self) -> bool {
        self.bits == 0
    }

    fn intersects(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) != 0
    }

    fn contains(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) == e.bits
    }

    fn add(&mut self, e: E) {
        self.bits |= bit(e);
    }

    fn plus(&self, e: E) -> EnumSet<E> {
        EnumSet {bits: self.bits | bit(e)}
    }

    fn contains_elem(&self, e: E) -> bool {
        (self.bits & bit(e)) != 0
    }

    fn each(&self, f: &fn(E) -> bool) {
        let mut bits = self.bits;
        let mut index = 0;
        while bits != 0 {
            if (bits & 1) != 0 {
                let e = CLike::from_uint(index);
                if !f(e) {
                    return;
                }
            }
            index += 1;
            bits >>= 1;
        }
    }
}

impl<E:CLike> core::Sub<EnumSet<E>, EnumSet<E>> for EnumSet<E> {
    fn sub(&self, e: &EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & !e.bits}
    }
}

impl<E:CLike> core::BitOr<EnumSet<E>, EnumSet<E>> for EnumSet<E> {
    fn bitor(&self, e: &EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits}
    }
}

impl<E:CLike> core::BitAnd<EnumSet<E>, EnumSet<E>> for EnumSet<E> {
    fn bitand(&self, e: &EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits}
    }
}
