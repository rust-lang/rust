// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::SmallVectorRepr::*;
use self::IntoIterRepr::*;

use std::iter::{IntoIterator, FromIterator};
use std::mem;
use std::slice;
use std::vec;

use util::move_map::MoveMap;

/// A vector type optimized for cases where the size is almost always 0 or 1
pub struct SmallVector<T> {
    repr: SmallVectorRepr<T>,
}

enum SmallVectorRepr<T> {
    Zero,
    One(T),
    Many(Vec<T>),
}

impl<T> FromIterator<T> for SmallVector<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> SmallVector<T> {
        let mut v = SmallVector::zero();
        v.extend(iter);
        v
    }
}

impl<T> Extend<T> for SmallVector<T> {
    fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
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

    pub fn as_slice(&self) -> &[T] {
        match self.repr {
            Zero => {
                let result: &[T] = &[];
                result
            }
            One(ref v) => {
                unsafe { slice::from_raw_parts(v, 1) }
            }
            Many(ref vs) => vs
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        match self.repr {
            Zero => None,
            One(..) => {
                let one = mem::replace(&mut self.repr, Zero);
                match one {
                    One(v1) => Some(v1),
                    _ => unreachable!()
                }
            }
            Many(ref mut vs) => vs.pop(),
        }
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
        for v in other.into_iter() {
            self.push(v);
        }
    }

    pub fn get(&self, idx: usize) -> &T {
        match self.repr {
            One(ref v) if idx == 0 => v,
            Many(ref vs) => &vs[idx],
            _ => panic!("out of bounds access")
        }
    }

    pub fn expect_one(self, err: &'static str) -> T {
        match self.repr {
            One(v) => v,
            Many(v) => {
                if v.len() == 1 {
                    v.into_iter().next().unwrap()
                } else {
                    panic!(err)
                }
            }
            _ => panic!(err)
        }
    }

    /// Deprecated: use `into_iter`.
    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.0.0", reason = "use into_iter")]
    pub fn move_iter(self) -> IntoIter<T> {
        self.into_iter()
    }

    pub fn len(&self) -> usize {
        match self.repr {
            Zero => 0,
            One(..) => 1,
            Many(ref vals) => vals.len()
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<T> IntoIterator for SmallVector<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        let repr = match self.repr {
            Zero => ZeroIterator,
            One(v) => OneIterator(v),
            Many(vs) => ManyIterator(vs.into_iter())
        };
        IntoIter { repr: repr }
    }
}

pub struct IntoIter<T> {
    repr: IntoIterRepr<T>,
}

enum IntoIterRepr<T> {
    ZeroIterator,
    OneIterator(T),
    ManyIterator(vec::IntoIter<T>),
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

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

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.repr {
            ZeroIterator => (0, Some(0)),
            OneIterator(..) => (1, Some(1)),
            ManyIterator(ref inner) => inner.size_hint()
        }
    }
}

impl<T> MoveMap<T> for SmallVector<T> {
    fn move_flat_map<F, I>(self, mut f: F) -> Self
        where F: FnMut(T) -> I,
              I: IntoIterator<Item=T>
    {
        match self.repr {
            Zero => Self::zero(),
            One(v) => f(v).into_iter().collect(),
            Many(vs) => SmallVector { repr: Many(vs.move_flat_map(f)) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len() {
        let v: SmallVector<isize> = SmallVector::zero();
        assert_eq!(0, v.len());

        assert_eq!(1, SmallVector::one(1).len());
        assert_eq!(5, SmallVector::many(vec![1, 2, 3, 4, 5]).len());
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
        let v: SmallVector<isize> = (vec![1, 2, 3]).into_iter().collect();
        assert_eq!(3, v.len());
        assert_eq!(&1, v.get(0));
        assert_eq!(&2, v.get(1));
        assert_eq!(&3, v.get(2));
    }

    #[test]
    fn test_move_iter() {
        let v = SmallVector::zero();
        let v: Vec<isize> = v.into_iter().collect();
        assert_eq!(v, Vec::new());

        let v = SmallVector::one(1);
        assert_eq!(v.into_iter().collect::<Vec<_>>(), [1]);

        let v = SmallVector::many(vec![1, 2, 3]);
        assert_eq!(v.into_iter().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_expect_one_zero() {
        let _: isize = SmallVector::zero().expect_one("");
    }

    #[test]
    #[should_panic]
    fn test_expect_one_many() {
        SmallVector::many(vec!(1, 2)).expect_one("");
    }

    #[test]
    fn test_expect_one_one() {
        assert_eq!(1, SmallVector::one(1).expect_one(""));
        assert_eq!(1, SmallVector::many(vec!(1)).expect_one(""));
    }
}
