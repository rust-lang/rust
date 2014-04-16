// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::vec;

/// A vector type optimized for cases where the size is almost always 0 or 1
pub struct SmallVector<T> {
    repr: SmallVectorRepr<T>,
}

enum SmallVectorRepr<T> {
    Zero,
    One(T),
    Many(Vec<T> ),
}

impl<T> Container for SmallVector<T> {
    fn len(&self) -> uint {
        match self.repr {
            Zero => 0,
            One(..) => 1,
            Many(ref vals) => vals.len()
        }
    }
}

impl<T> FromIterator<T> for SmallVector<T> {
    fn from_iter<I: Iterator<T>>(iter: I) -> SmallVector<T> {
        let mut v = SmallVector::zero();
        v.extend(iter);
        v
    }
}

impl<T> Extendable<T> for SmallVector<T> {
    fn extend<I: Iterator<T>>(&mut self, mut iter: I) {
        for val in iter {
            self.push(val);
        }
    }
}

impl<T> SmallVector<T> {
    pub fn zero() -> SmallVector<T> {
        SmallVector { repr: Zero }
    }

    pub fn one(v: T) -> SmallVector<T> {
        SmallVector { repr: One(v) }
    }

    pub fn many(vs: Vec<T>) -> SmallVector<T> {
        SmallVector { repr: Many(vs) }
    }

    pub fn push(&mut self, v: T) {
        match self.repr {
            Zero => self.repr = One(v),
            One(..) => {
                let one = mem::replace(&mut self.repr, Zero);
                match one {
                    One(v1) => mem::replace(&mut self.repr, Many(vec!(v1, v))),
                    _ => unreachable!()
                };
            }
            Many(ref mut vs) => vs.push(v)
        }
    }

    pub fn push_all(&mut self, other: SmallVector<T>) {
        for v in other.move_iter() {
            self.push(v);
        }
    }

    pub fn get<'a>(&'a self, idx: uint) -> &'a T {
        match self.repr {
            One(ref v) if idx == 0 => v,
            Many(ref vs) => vs.get(idx),
            _ => fail!("out of bounds access")
        }
    }

    pub fn expect_one(self, err: &'static str) -> T {
        match self.repr {
            One(v) => v,
            Many(v) => {
                if v.len() == 1 {
                    v.move_iter().next().unwrap()
                } else {
                    fail!(err)
                }
            }
            _ => fail!(err)
        }
    }

    pub fn move_iter(self) -> MoveItems<T> {
        let repr = match self.repr {
            Zero => ZeroIterator,
            One(v) => OneIterator(v),
            Many(vs) => ManyIterator(vs.move_iter())
        };
        MoveItems { repr: repr }
    }
}

pub struct MoveItems<T> {
    repr: MoveItemsRepr<T>,
}

enum MoveItemsRepr<T> {
    ZeroIterator,
    OneIterator(T),
    ManyIterator(vec::MoveItems<T>),
}

impl<T> Iterator<T> for MoveItems<T> {
    fn next(&mut self) -> Option<T> {
        match self.repr {
            ZeroIterator => None,
            OneIterator(..) => {
                let mut replacement = ZeroIterator;
                mem::swap(&mut self.repr, &mut replacement);
                match replacement {
                    OneIterator(v) => Some(v),
                    _ => unreachable!()
                }
            }
            ManyIterator(ref mut inner) => inner.next()
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        match self.repr {
            ZeroIterator => (0, Some(0)),
            OneIterator(..) => (1, Some(1)),
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
        assert_eq!(5, SmallVector::many(vec!(1, 2, 3, 4, 5)).len());
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
    fn test_from_iter() {
        let v: SmallVector<int> = (vec!(1, 2, 3)).move_iter().collect();
        assert_eq!(3, v.len());
        assert_eq!(&1, v.get(0));
        assert_eq!(&2, v.get(1));
        assert_eq!(&3, v.get(2));
    }

    #[test]
    fn test_move_iter() {
        let v = SmallVector::zero();
        let v: Vec<int> = v.move_iter().collect();
        assert_eq!(Vec::new(), v);

        let v = SmallVector::one(1);
        assert_eq!(vec!(1), v.move_iter().collect());

        let v = SmallVector::many(vec!(1, 2, 3));
        assert_eq!(vec!(1, 2, 3), v.move_iter().collect());
    }

    #[test]
    #[should_fail]
    fn test_expect_one_zero() {
        let _: int = SmallVector::zero().expect_one("");
    }

    #[test]
    #[should_fail]
    fn test_expect_one_many() {
        SmallVector::many(vec!(1, 2)).expect_one("");
    }

    #[test]
    fn test_expect_one_one() {
        assert_eq!(1, SmallVector::one(1).expect_one(""));
        assert_eq!(1, SmallVector::many(vec!(1)).expect_one(""));
    }
}
