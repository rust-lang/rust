// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::small_vec::SmallVec;

pub type SmallVector<T> = SmallVec<[T; 1]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len() {
        let v: SmallVector<isize> = SmallVector::new();
        assert_eq!(0, v.len());

        assert_eq!(1, SmallVector::one(1).len());
        assert_eq!(5, SmallVector::many(vec![1, 2, 3, 4, 5]).len());
    }

    #[test]
    fn test_push_get() {
        let mut v = SmallVector::new();
        v.push(1);
        assert_eq!(1, v.len());
        assert_eq!(1, v[0]);
        v.push(2);
        assert_eq!(2, v.len());
        assert_eq!(2, v[1]);
        v.push(3);
        assert_eq!(3, v.len());
        assert_eq!(3, v[2]);
    }

    #[test]
    fn test_from_iter() {
        let v: SmallVector<isize> = (vec![1, 2, 3]).into_iter().collect();
        assert_eq!(3, v.len());
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
        assert_eq!(3, v[2]);
    }

    #[test]
    fn test_move_iter() {
        let v = SmallVector::new();
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
        let _: isize = SmallVector::new().expect_one("");
    }

    #[test]
    #[should_panic]
    fn test_expect_one_many() {
        SmallVector::many(vec![1, 2]).expect_one("");
    }

    #[test]
    fn test_expect_one_one() {
        assert_eq!(1, SmallVector::one(1).expect_one(""));
        assert_eq!(1, SmallVector::many(vec![1]).expect_one(""));
    }
}
