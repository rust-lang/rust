// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! A type that represents one of two alternatives

use cmp::Eq;
use result::Result;

/// The either type
enum Either<T, U> {
    Left(T),
    Right(U)
}

fn either<T, U, V>(f_left: fn((&T)) -> V,
                   f_right: fn((&U)) -> V, value: &Either<T, U>) -> V {
    /*!
     * Applies a function based on the given either value
     *
     * If `value` is left(T) then `f_left` is applied to its contents, if
     * `value` is right(U) then `f_right` is applied to its contents, and the
     * result is returned.
     */

    match *value {
      Left(ref l) => f_left(l),
      Right(ref r) => f_right(r)
    }
}

fn lefts<T: copy, U>(eithers: &[Either<T, U>]) -> ~[T] {
    //! Extracts from a vector of either all the left values

    let mut result: ~[T] = ~[];
    for vec::each(eithers) |elt| {
        match elt {
          Left(l) => vec::push(result, l),
          _ => { /* fallthrough */ }
        }
    }
    return result;
}

fn rights<T, U: copy>(eithers: &[Either<T, U>]) -> ~[U] {
    //! Extracts from a vector of either all the right values

    let mut result: ~[U] = ~[];
    for vec::each(eithers) |elt| {
        match elt {
          Right(r) => vec::push(result, r),
          _ => { /* fallthrough */ }
        }
    }
    return result;
}

fn partition<T: copy, U: copy>(eithers: &[Either<T, U>])
    -> {lefts: ~[T], rights: ~[U]} {
    /*!
     * Extracts from a vector of either all the left values and right values
     *
     * Returns a structure containing a vector of left values and a vector of
     * right values.
     */

    let mut lefts: ~[T] = ~[];
    let mut rights: ~[U] = ~[];
    for vec::each(eithers) |elt| {
        match elt {
          Left(l) => vec::push(lefts, l),
          Right(r) => vec::push(rights, r)
        }
    }
    return {lefts: lefts, rights: rights};
}

pure fn flip<T: copy, U: copy>(eith: &Either<T, U>) -> Either<U, T> {
    //! Flips between left and right of a given either

    match *eith {
      Right(r) => Left(r),
      Left(l) => Right(l)
    }
}

pure fn to_result<T: copy, U: copy>(eith: &Either<T, U>) -> Result<U, T> {
    /*!
     * Converts either::t to a result::t
     *
     * Converts an `either` type to a `result` type, making the "right" choice
     * an ok result, and the "left" choice a fail
     */

    match *eith {
      Right(r) => result::Ok(r),
      Left(l) => result::Err(l)
    }
}

pure fn is_left<T, U>(eith: &Either<T, U>) -> bool {
    //! Checks whether the given value is a left

    match *eith { Left(_) => true, _ => false }
}

pure fn is_right<T, U>(eith: &Either<T, U>) -> bool {
    //! Checks whether the given value is a right

    match *eith { Right(_) => true, _ => false }
}

pure fn unwrap_left<T,U>(+eith: Either<T,U>) -> T {
    //! Retrieves the value in the left branch. Fails if the either is Right.

    match move eith {
        Left(move x) => x, Right(_) => fail ~"either::unwrap_left Right"
    }
}

pure fn unwrap_right<T,U>(+eith: Either<T,U>) -> U {
    //! Retrieves the value in the right branch. Fails if the either is Left.

    match move eith {
        Right(move x) => x, Left(_) => fail ~"either::unwrap_right Left"
    }
}

impl<T:Eq,U:Eq> Either<T,U> : Eq {
    pure fn eq(&&other: Either<T,U>) -> bool {
        match self {
            Left(a) => {
                match other {
                    Left(b) => a.eq(b),
                    Right(_) => false
                }
            }
            Right(a) => {
                match other {
                    Left(_) => false,
                    Right(b) => a.eq(b)
                }
            }
        }
    }
    pure fn ne(&&other: Either<T,U>) -> bool { !self.eq(other) }
}

#[test]
fn test_either_left() {
    let val = Left(10);
    fn f_left(x: &int) -> bool { *x == 10 }
    fn f_right(_x: &uint) -> bool { false }
    assert (either(f_left, f_right, &val));
}

#[test]
fn test_either_right() {
    let val = Right(10u);
    fn f_left(_x: &int) -> bool { false }
    fn f_right(x: &uint) -> bool { *x == 10u }
    assert (either(f_left, f_right, &val));
}

#[test]
fn test_lefts() {
    let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
    let result = lefts(input);
    assert (result == ~[10, 12, 14]);
}

#[test]
fn test_lefts_none() {
    let input: ~[Either<int, int>] = ~[Right(10), Right(10)];
    let result = lefts(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_lefts_empty() {
    let input: ~[Either<int, int>] = ~[];
    let result = lefts(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_rights() {
    let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
    let result = rights(input);
    assert (result == ~[11, 13]);
}

#[test]
fn test_rights_none() {
    let input: ~[Either<int, int>] = ~[Left(10), Left(10)];
    let result = rights(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_rights_empty() {
    let input: ~[Either<int, int>] = ~[];
    let result = rights(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_partition() {
    let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
    let result = partition(input);
    assert (result.lefts[0] == 10);
    assert (result.lefts[1] == 12);
    assert (result.lefts[2] == 14);
    assert (result.rights[0] == 11);
    assert (result.rights[1] == 13);
}

#[test]
fn test_partition_no_lefts() {
    let input: ~[Either<int, int>] = ~[Right(10), Right(11)];
    let result = partition(input);
    assert (vec::len(result.lefts) == 0u);
    assert (vec::len(result.rights) == 2u);
}

#[test]
fn test_partition_no_rights() {
    let input: ~[Either<int, int>] = ~[Left(10), Left(11)];
    let result = partition(input);
    assert (vec::len(result.lefts) == 2u);
    assert (vec::len(result.rights) == 0u);
}

#[test]
fn test_partition_empty() {
    let input: ~[Either<int, int>] = ~[];
    let result = partition(input);
    assert (vec::len(result.lefts) == 0u);
    assert (vec::len(result.rights) == 0u);
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
