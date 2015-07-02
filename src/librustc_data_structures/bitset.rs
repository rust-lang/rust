// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Add, Sub, Shl, Shr, BitAnd, BitOr};
use std::num::{Zero, One};
use std::cmp::PartialEq;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BitSet<N> {
    bits: N
}

pub trait UnsignedInt : Zero
                  + One
                  + Add<Self, Output=Self>
                  + Sub<Self, Output=Self>
                  + Shl<Self, Output=Self>
                  + Shr<Self, Output=Self>
                  + BitAnd<Self, Output=Self>
                  + BitOr<Self, Output=Self>
                  + PartialEq<Self>
                  + PartialOrd<Self>
                  + Copy {}

impl UnsignedInt for u8 {}

pub type U8BitSet = BitSet<u8>;

impl<N: UnsignedInt> BitSet<N> {
    pub fn empty() -> BitSet<N> {
        BitSet { bits: Zero::zero() }
    }

    pub fn insert(&mut self, value: N) {
        self.bits = self.bits | (N::one() << value);
    }

    pub fn contains(&self, value: N) -> bool {
        ((self.bits >> value) & N::one()) == N::one()
    }

    pub fn iter(&self) -> BitSetIter<N> {
        self.into_iter()
    }

    pub fn is_empty(&self) -> bool {
        self.bits == N::zero()
    }

    pub fn is_superset(&self, other: BitSet<N>) -> bool {
        (self.bits & other.bits) == other.bits
    }
}

pub struct BitSetIter<N> {
    bits: N,
    index: N
}

impl<N: UnsignedInt> Iterator for BitSetIter<N> {
    type Item = N;

    fn next(&mut self) -> Option<N> {
        if self.bits == N::zero() {
            return None;
        }

        while (self.bits & N::one()) == N::zero() {
            self.index = self.index + N::one();
            self.bits  = self.bits >> N::one();
        }

        let result = self.index;

        self.index = self.index + N::one();
        self.bits = self.bits >> N::one();
        Some(result)
    }
}


impl<'a, N: UnsignedInt> IntoIterator for &'a BitSet<N> {
    type Item = N;
    type IntoIter = BitSetIter<N>;

    fn into_iter(self) -> BitSetIter<N> {
        BitSetIter { bits: self.bits, index: N::zero() }
    }
}
