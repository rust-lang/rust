// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A vector type intended to be used for collecting from iterators onto the stack.
//!
//! Space for up to N elements is provided on the stack.  If more elements are collected, Vec is
//! used to store the values on the heap. SmallVec is similar to AccumulateVec, but adds
//! the ability to push elements.
//!
//! The N above is determined by Array's implementor, by way of an associated constant.

use smallvec::{Array, SmallVec};

pub type OneVector<T> = SmallVec<[T; 1]>;

pub trait ExpectOne<A: Array> {
    fn expect_one(self, err: &'static str) -> A::Item;
}

impl<A: Array> ExpectOne<A> for SmallVec<A> {
    fn expect_one(self, err: &'static str) -> A::Item {
        assert!(self.len() == 1, err);
        self.into_iter().next().unwrap()
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;

    #[test]
    #[should_panic]
    fn test_expect_one_zero() {
        let _: isize = OneVector::new().expect_one("");
    }

    #[test]
    #[should_panic]
    fn test_expect_one_many() {
        OneVector::from_vec(vec![1, 2]).expect_one("");
    }

    #[test]
    fn test_expect_one_one() {
        assert_eq!(1, (smallvec![1] as OneVector<_>).expect_one(""));
        assert_eq!(1, OneVector::from_vec(vec![1]).expect_one(""));
    }
}
