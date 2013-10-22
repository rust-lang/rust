// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A type that represents one of two alternatives

#[allow(missing_doc)];

use option::{Some, None};
use option;
use clone::Clone;
use container::Container;
use cmp::Eq;
use iter::{Iterator, FilterMap};
use result::Result;
use result;
use str::StrSlice;
use vec;
use vec::{OwnedVector, ImmutableVector};

/// `Either` is a type that represents one of two alternatives
#[deriving(Clone, Eq, IterBytes)]
pub enum Either<L, R> {
    Left(L),
    Right(R)
}

impl<L, R> Either<L, R> {
    /// Applies a function based on the given either value
    ///
    /// If `value` is `Left(L)` then `f_left` is applied to its contents, if
    /// `value` is `Right(R)` then `f_right` is applied to its contents, and the
    /// result is returned.
    #[inline]
    pub fn either<T>(&self, f_left: &fn(&L) -> T, f_right: &fn(&R) -> T) -> T {
        match *self {
            Left(ref l) => f_left(l),
            Right(ref r) => f_right(r)
        }
    }

    /// Flips between left and right of a given `Either`
    #[inline]
    pub fn flip(self) -> Either<R, L> {
        match self {
            Right(r) => Left(r),
            Left(l) => Right(l)
        }
    }

    /// Checks whether the given value is a `Left`
    #[inline]
    pub fn is_left(&self) -> bool {
        match *self {
            Left(_) => true,
            _ => false
        }
    }

    /// Checks whether the given value is a `Right`
    #[inline]
    pub fn is_right(&self) -> bool {
        match *self {
            Right(_) => true,
            _ => false
        }
    }

    /// Retrieves the value from a `Left`.
    /// Fails with a specified reason if the `Either` is `Right`.
    #[inline]
    pub fn expect_left(self, reason: &str) -> L {
        match self {
            Left(x) => x,
            Right(_) => fail!("{}", reason.to_owned())
        }
    }

    /// Retrieves the value from a `Left`. Fails if the `Either` is `Right`.
    #[inline]
    pub fn unwrap_left(self) -> L {
        self.expect_left("called Either::unwrap_left()` on `Right` value")
    }

    /// Retrieves the value from a `Right`.
    /// Fails with a specified reason if the `Either` is `Left`.
    #[inline]
    pub fn expect_right(self, reason: &str) -> R {
        match self {
            Right(x) => x,
            Left(_) => fail!("{}", reason.to_owned())
        }
    }

    /// Retrieves the value from a `Right`. Fails if the `Either` is `Left`.
    #[inline]
    pub fn unwrap_right(self) -> R {
        self.expect_right("called Either::unwrap_right()` on `Left` value")
    }
}

/// A generic trait for converting a value to a `Either`
pub trait ToEither<L, R> {
    /// Convert to the `either` type
    fn to_either(&self) -> Either<L, R>;
}

/// A generic trait for converting a value to a `Either`
pub trait IntoEither<L, R> {
    /// Convert to the `either` type
    fn into_either(self) -> Either<L, R>;
}

/// A generic trait for converting a value to a `Either`
pub trait AsEither<L, R> {
    /// Convert to the `either` type
    fn as_either<'a>(&'a self) -> Either<&'a L, &'a R>;
}

impl<L, R: Clone> option::ToOption<R> for Either<L, R> {
    #[inline]
    fn to_option(&self)-> option::Option<R> {
        match *self {
            Left(_) => None,
            Right(ref r) => Some(r.clone()),
        }
    }
}

impl<L, R> option::IntoOption<R> for Either<L, R> {
    #[inline]
    fn into_option(self)-> option::Option<R> {
        match self {
            Left(_) => None,
            Right(r) => Some(r),
        }
    }
}

impl<L, R> option::AsOption<R> for Either<L, R> {
    #[inline]
    fn as_option<'a>(&'a self) -> option::Option<&'a R> {
        match *self {
            Left(_) => None,
            Right(ref r) => Some(r),
        }
    }
}

impl<L: Clone, R: Clone> result::ToResult<R, L> for Either<L, R> {
    #[inline]
    fn to_result(&self)-> result::Result<R, L> {
        match *self {
            Left(ref l) => result::Err(l.clone()),
            Right(ref r) => result::Ok(r.clone()),
        }
    }
}

impl<L, R> result::IntoResult<R, L> for Either<L, R> {
    #[inline]
    fn into_result(self)-> result::Result<R, L> {
        match self {
            Left(l) => result::Err(l),
            Right(r) => result::Ok(r),
        }
    }
}

impl<L, R> result::AsResult<R, L> for Either<L, R> {
    #[inline]
    fn as_result<'a>(&'a self) -> result::Result<&'a R, &'a L> {
        match *self {
            Left(ref l) => result::Err(l),
            Right(ref r) => result::Ok(r),
        }
    }
}

impl<L: Clone, R: Clone> ToEither<L, R> for Either<L, R> {
    fn to_either(&self) -> Either<L, R> { self.clone() }
}

impl<L, R> IntoEither<L, R> for Either<L, R> {
    fn into_either(self) -> Either<L, R> { self }
}

impl<L, R> AsEither<L, R> for Either<L, R> {
    fn as_either<'a>(&'a self) -> Either<&'a L, &'a R> {
        match *self {
            Left(ref l) => Left(l),
            Right(ref r) => Right(r),
        }
    }
}

/// An iterator yielding the `Left` values of its source
pub type Lefts<L, R, Iter> = FilterMap<'static, Either<L, R>, L, Iter>;

/// An iterator yielding the `Right` values of its source
pub type Rights<L, R, Iter> = FilterMap<'static, Either<L, R>, R, Iter>;

/// Extracts all the left values
pub fn lefts<L, R, Iter: Iterator<Either<L, R>>>(eithers: Iter)
    -> Lefts<L, R, Iter> {
    do eithers.filter_map |elt| {
        match elt {
            Left(x) => Some(x),
            _ => None,
        }
    }
}

/// Extracts all the right values
pub fn rights<L, R, Iter: Iterator<Either<L, R>>>(eithers: Iter)
    -> Rights<L, R, Iter> {
    do eithers.filter_map |elt| {
        match elt {
            Right(x) => Some(x),
            _ => None,
        }
    }
}


// FIXME: #8228 Replaceable by an external iterator?
/// Extracts from a vector of either all the left values and right values
///
/// Returns a structure containing a vector of left values and a vector of
/// right values.
pub fn partition<L, R>(eithers: ~[Either<L, R>]) -> (~[L], ~[R]) {
    let n_lefts = eithers.iter().count(|elt| elt.is_left());
    let mut lefts = vec::with_capacity(n_lefts);
    let mut rights = vec::with_capacity(eithers.len() - n_lefts);
    for elt in eithers.move_iter() {
        match elt {
            Left(l) => lefts.push(l),
            Right(r) => rights.push(r)
        }
    }
    return (lefts, rights);
}

#[cfg(test)]
mod tests {
    use super::*;

    use option::{IntoOption, ToOption, AsOption};
    use option;
    use result::{IntoResult, ToResult, AsResult};
    use result;

    #[test]
    fn test_either_left() {
        let val = Left(10);
        fn f_left(x: &int) -> bool { *x == 10 }
        fn f_right(_x: &uint) -> bool { false }
        assert!(val.either(f_left, f_right));
    }

    #[test]
    fn test_either_right() {
        let val = Right(10u);
        fn f_left(_x: &int) -> bool { false }
        fn f_right(x: &uint) -> bool { *x == 10u }
        assert!(val.either(f_left, f_right));
    }

    #[test]
    fn test_lefts() {
        let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
        let result = lefts(input.move_iter()).to_owned_vec();
        assert_eq!(result, ~[10, 12, 14]);
    }

    #[test]
    fn test_lefts_none() {
        let input: ~[Either<int, int>] = ~[Right(10), Right(10)];
        let result = lefts(input.move_iter()).to_owned_vec();
        assert_eq!(result.len(), 0u);
    }

    #[test]
    fn test_lefts_empty() {
        let input: ~[Either<int, int>] = ~[];
        let result = lefts(input.move_iter()).to_owned_vec();
        assert_eq!(result.len(), 0u);
    }

    #[test]
    fn test_rights() {
        let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
        let result = rights(input.move_iter()).to_owned_vec();
        assert_eq!(result, ~[11, 13]);
    }

    #[test]
    fn test_rights_none() {
        let input: ~[Either<int, int>] = ~[Left(10), Left(10)];
        let result = rights(input.move_iter()).to_owned_vec();
        assert_eq!(result.len(), 0u);
    }

    #[test]
    fn test_rights_empty() {
        let input: ~[Either<int, int>] = ~[];
        let result = rights(input.move_iter()).to_owned_vec();
        assert_eq!(result.len(), 0u);
    }

    #[test]
    fn test_partition() {
        let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
        let (lefts, rights) = partition(input);
        assert_eq!(lefts[0], 10);
        assert_eq!(lefts[1], 12);
        assert_eq!(lefts[2], 14);
        assert_eq!(rights[0], 11);
        assert_eq!(rights[1], 13);
    }

    #[test]
    fn test_partition_no_lefts() {
        let input: ~[Either<int, int>] = ~[Right(10), Right(11)];
        let (lefts, rights) = partition(input);
        assert_eq!(lefts.len(), 0u);
        assert_eq!(rights.len(), 2u);
    }

    #[test]
    fn test_partition_no_rights() {
        let input: ~[Either<int, int>] = ~[Left(10), Left(11)];
        let (lefts, rights) = partition(input);
        assert_eq!(lefts.len(), 2u);
        assert_eq!(rights.len(), 0u);
    }

    #[test]
    fn test_partition_empty() {
        let input: ~[Either<int, int>] = ~[];
        let (lefts, rights) = partition(input);
        assert_eq!(lefts.len(), 0u);
        assert_eq!(rights.len(), 0u);
    }

    #[test]
    pub fn test_to_option() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.to_option(), option::Some(100));
        assert_eq!(left.to_option(), option::None);
    }

    #[test]
    pub fn test_into_option() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.into_option(), option::Some(100));
        assert_eq!(left.into_option(), option::None);
    }

    #[test]
    pub fn test_as_option() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.as_option().unwrap(), &100);
        assert_eq!(left.as_option(), option::None);
    }

    #[test]
    pub fn test_to_result() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.to_result(), result::Ok(100));
        assert_eq!(left.to_result(), result::Err(404));
    }

    #[test]
    pub fn test_into_result() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.into_result(), result::Ok(100));
        assert_eq!(left.into_result(), result::Err(404));
    }

    #[test]
    pub fn test_as_result() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        let x = 100;
        assert_eq!(right.as_result(), result::Ok(&x));

        let x = 404;
        assert_eq!(left.as_result(), result::Err(&x));
    }

    #[test]
    pub fn test_to_either() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.to_either(), Right(100));
        assert_eq!(left.to_either(), Left(404));
    }

    #[test]
    pub fn test_into_either() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.into_either(), Right(100));
        assert_eq!(left.into_either(), Left(404));
    }

    #[test]
    pub fn test_as_either() {
        let right: Either<int, int> = Right(100);
        let left: Either<int, int> = Left(404);

        assert_eq!(right.as_either().unwrap_right(), &100);
        assert_eq!(left.as_either().unwrap_left(), &404);
    }
}
