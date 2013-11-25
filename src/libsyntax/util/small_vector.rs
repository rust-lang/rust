// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::vec::MoveIterator;
use std::util;

/// A vector type optimized for cases where the size is almost always 0 or 1
pub enum SmallVector<T> {
    priv Zero,
    priv One(T),
    priv Many(~[T]),
}

impl<T> Container for SmallVector<T> {
    fn len(&self) -> uint {
        match *self {
            Zero => 0,
            One(*) => 1,
            Many(ref vals) => vals.len()
        }
    }
}

impl<T> FromIterator<T> for SmallVector<T> {
    fn from_iterator<I: Iterator<T>>(iter: &mut I) -> SmallVector<T> {
        let mut v = Zero;
        for val in *iter {
            v.push(val);
        }
        v
    }
}

impl<T> SmallVector<T> {
    pub fn zero() -> SmallVector<T> {
        Zero
    }

    pub fn one(v: T) -> SmallVector<T> {
        One(v)
    }

    pub fn many(vs: ~[T]) -> SmallVector<T> {
        Many(vs)
    }

    pub fn push(&mut self, v: T) {
        match *self {
            Zero => *self = One(v),
            One(*) => {
                let mut tmp = Many(~[]);
                util::swap(self, &mut tmp);
                match *self {
                    Many(ref mut vs) => {
                        match tmp {
                            One(v1) => {
                                vs.push(v1);
                                vs.push(v);
                            }
                            _ => unreachable!()
                        }
                    }
                    _ => unreachable!()
                }
            }
            Many(ref mut vs) => vs.push(v)
        }
    }

    pub fn get<'a>(&'a self, idx: uint) -> &'a T {
        match *self {
            One(ref v) if idx == 0 => v,
            Many(ref vs) => &vs[idx],
            _ => fail!("Out of bounds access")
        }
    }

    pub fn iter<'a>(&'a self) -> SmallVectorIterator<'a, T> {
        SmallVectorIterator {
            vec: self,
            idx: 0
        }
    }

    pub fn move_iter(self) -> SmallVectorMoveIterator<T> {
        match self {
            Zero => ZeroIterator,
            One(v) => OneIterator(v),
            Many(vs) => ManyIterator(vs.move_iter())
        }
    }
}

pub struct SmallVectorIterator<'vec, T> {
    priv vec: &'vec SmallVector<T>,
    priv idx: uint
}

impl<'vec, T> Iterator<&'vec T> for SmallVectorIterator<'vec, T> {
    fn next(&mut self) -> Option<&'vec T> {
        if self.idx == self.vec.len() {
            return None;
        }

        self.idx += 1;
        Some(self.vec.get(self.idx - 1))
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let rem = self.vec.len() - self.idx;
        (rem, Some(rem))
    }
}

pub enum SmallVectorMoveIterator<T> {
    priv ZeroIterator,
    priv OneIterator(T),
    priv ManyIterator(MoveIterator<T>),
}

impl<T> Iterator<T> for SmallVectorMoveIterator<T> {
    fn next(&mut self) -> Option<T> {
        match *self {
            ZeroIterator => None,
            OneIterator(*) => {
                let mut replacement = ZeroIterator;
                util::swap(self, &mut replacement);
                match replacement {
                    OneIterator(v) => Some(v),
                    _ => unreachable!()
                }
            }
            ManyIterator(ref mut inner) => inner.next()
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        match *self {
            ZeroIterator => (0, Some(0)),
            OneIterator(*) => (1, Some(1)),
            ManyIterator(ref inner) => inner.size_hint()
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_len() {
        let v: SmallVector<int> = SmallVector::zero();
        assert_eq!(0, v.len());

        assert_eq!(1, SmallVector::one(1).len());
        assert_eq!(5, SmallVector::many(~[1, 2, 3, 4, 5]).len());
    }

    #[test]
    fn test_push_get() {
        let mut v = SmallVector::zero();
        v.push(1);
        assert_eq!(1, v.len());
        assert_eq!(&1, v.get(0));
        v.push(2);
        assert_eq!(2, v.len());
        assert_eq!(&2, v.get(1));
        v.push(3);
        assert_eq!(3, v.len());
        assert_eq!(&3, v.get(2));
    }

    #[test]
    fn test_from_iterator() {
        let v: SmallVector<int> = (~[1, 2, 3]).move_iter().collect();
        assert_eq!(3, v.len());
        assert_eq!(&1, v.get(0));
        assert_eq!(&2, v.get(1));
        assert_eq!(&3, v.get(2));
    }

    #[test]
    fn test_iter() {
        let v = SmallVector::zero();
        let v: ~[&int] = v.iter().collect();
        assert_eq!(~[], v);

        let v = SmallVector::one(1);
        assert_eq!(~[&1], v.iter().collect());

        let v = SmallVector::many(~[1, 2, 3]);
        assert_eq!(~[&1, &2, &3], v.iter().collect());
    }

    #[test]
    fn test_move_iter() {
        let v = SmallVector::zero();
        let v: ~[int] = v.move_iter().collect();
        assert_eq!(~[], v);

        let v = SmallVector::one(1);
        assert_eq!(~[1], v.move_iter().collect());

        let v = SmallVector::many(~[1, 2, 3]);
        assert_eq!(~[1, 2, 3], v.move_iter().collect());
    }
}
