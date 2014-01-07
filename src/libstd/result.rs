// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Signaling success or failure states (`Result` type)

use clone::Clone;
use cmp::Eq;
use fmt;
use iter::{Iterator, FromIterator};
use option::{None, Option, Some};
use str::OwnedStr;
use to_str::ToStr;

/// `Result` is a type that represents either success (`Ok`) or failure (`Err`).
#[deriving(Clone, DeepClone, Eq, Ord, TotalEq, TotalOrd, ToStr)]
pub enum Result<T, E> {
    /// Contains the success value
    Ok(T),

    /// Contains the error value
    Err(E)
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

impl<T, E> Result<T, E> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns true if the result is `Ok`
    #[inline]
    pub fn is_ok(&self) -> bool {
        match *self {
            Ok(_) => true,
            Err(_) => false
        }
    }

    /// Returns true if the result is `Err`
    #[inline]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }


    /////////////////////////////////////////////////////////////////////////
    // Adapter for each variant
    /////////////////////////////////////////////////////////////////////////

    /// Convert from `Result<T, E>` to `Option<T>`
    #[inline]
    pub fn ok(self) -> Option<T> {
        match self {
            Ok(x)  => Some(x),
            Err(_) => None,
        }
    }

    /// Convert from `Result<T, E>` to `Option<E>`
    #[inline]
    pub fn err(self) -> Option<E> {
        match self {
            Ok(_)  => None,
            Err(x) => Some(x),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Convert from `Result<T, E>` to `Result<&T, &E>`
    #[inline]
    pub fn as_ref<'r>(&'r self) -> Result<&'r T, &'r E> {
        match *self {
            Ok(ref x) => Ok(x),
            Err(ref x) => Err(x),
        }
    }

    /// Convert from `Result<T, E>` to `Result<&mut T, &mut E>`
    #[inline]
    pub fn as_mut<'r>(&'r mut self) -> Result<&'r mut T, &'r mut E> {
        match *self {
            Ok(ref mut x) => Ok(x),
            Err(ref mut x) => Err(x),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Transforming contained values
    /////////////////////////////////////////////////////////////////////////

    /// Maps an `Result<T, E>` to `Result<U, E>` by applying a function to an
    /// contained `Ok` value, leaving an `Err` value untouched.
    ///
    /// This function can be used to compose the results of two functions.
    ///
    /// Example:
    ///
    ///     let res = read_file(file).map(|buf| {
    ///         parse_bytes(buf)
    ///     })
    #[inline]
    pub fn map<U>(self, op: |T| -> U) -> Result<U,E> {
        match self {
          Ok(t) => Ok(op(t)),
          Err(e) => Err(e)
        }
    }

    /// Maps an `Result<T, E>` to `Result<T, F>` by applying a function to an
    /// contained `Err` value, leaving an `Ok` value untouched.
    ///
    /// This function can be used to pass through a successful result while handling
    /// an error.
    #[inline]
    pub fn map_err<F>(self, op: |E| -> F) -> Result<T,F> {
        match self {
          Ok(t) => Ok(t),
          Err(e) => Err(op(e))
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `res` if the result is `Ok`, otherwise returns the `Err` value of `self`.
    #[inline]
    pub fn and<U>(self, res: Result<U, E>) -> Result<U, E> {
        match self {
            Ok(_) => res,
            Err(e) => Err(e),
        }
    }

    /// Calls `op` if the result is `Ok`, otherwise returns the `Err` value of `self`.
    ///
    /// This function can be used for control flow based on result values
    #[inline]
    pub fn and_then<U>(self, op: |T| -> Result<U, E>) -> Result<U, E> {
        match self {
            Ok(t) => op(t),
            Err(e) => Err(e),
        }
    }

    /// Returns `res` if the result is `Err`, otherwise returns the `Ok` value of `self`.
    #[inline]
    pub fn or(self, res: Result<T, E>) -> Result<T, E> {
        match self {
            Ok(_) => self,
            Err(_) => res,
        }
    }

    /// Calls `op` if the result is `Err`, otherwise returns the `Ok` value of `self`.
    ///
    /// This function can be used for control flow based on result values
    #[inline]
    pub fn or_else<F>(self, op: |E| -> Result<T, F>) -> Result<T, F> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => op(e),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Common special cases
    /////////////////////////////////////////////////////////////////////////

    /// Unwraps a result, yielding the content of an `Ok`.
    /// Fails if the value is an `Err`.
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            Ok(t) => t,
            Err(_) => fail!("called `Result::unwrap()` on an `Err` value")
        }
    }

    /// Unwraps a result, yielding the content of an `Err`.
    /// Fails if the value is an `Ok`.
    #[inline]
    pub fn unwrap_err(self) -> E {
        match self {
            Ok(_) => fail!("called `Result::unwrap_err()` on an `Ok` value"),
            Err(e) => e
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

impl<T: fmt::Default, E: fmt::Default> fmt::Default for Result<T, E> {
    #[inline]
    fn fmt(s: &Result<T, E>, f: &mut fmt::Formatter) {
        match *s {
            Ok(ref t) => write!(f.buf, "Ok({})", *t),
            Err(ref e) => write!(f.buf, "Err({})", *e)
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Free functions
/////////////////////////////////////////////////////////////////////////////

/// Takes each element in the `Iterator`: if it is an `Err`, no further
/// elements are taken, and the `Err` is returned. Should no `Err` occur, a
/// vector containing the values of each `Result` is returned.
///
/// Here is an example which increments every integer in a vector,
/// checking for overflow:
///
///     fn inc_conditionally(x: uint) -> Result<uint, &'static str> {
///         if x == uint::max_value { return Err("overflow"); }
///         else { return Ok(x+1u); }
///     }
///     let v = [1u, 2, 3];
///     let res = collect(v.iter().map(|&x| inc_conditionally(x)));
///     assert!(res == Ok(~[2u, 3, 4]));
#[inline]
pub fn collect<T, E, Iter: Iterator<Result<T, E>>, V: FromIterator<T>>(iter: Iter) -> Result<V, E> {
    // FIXME(#11084): This should be twice as fast once this bug is closed.
    let mut iter = iter.scan(None, |state, x| {
        match x {
            Ok(x) => Some(x),
            Err(err) => {
                *state = Some(err);
                None
            }
        }
    });

    let v: V = FromIterator::from_iterator(&mut iter);

    match iter.state {
        Some(err) => Err(err),
        None => Ok(v),
    }
}

/// Perform a fold operation over the result values from an iterator.
///
/// If an `Err` is encountered, it is immediately returned.
/// Otherwise, the folded value is returned.
#[inline]
pub fn fold<T,
            V,
            E,
            Iter: Iterator<Result<T, E>>>(
            mut iterator: Iter,
            mut init: V,
            f: |V, T| -> V)
            -> Result<V, E> {
    for t in iterator {
        match t {
            Ok(v) => init = f(init, v),
            Err(u) => return Err(u)
        }
    }
    Ok(init)
}

/// Perform a trivial fold operation over the result values
/// from an iterator.
///
/// If an `Err` is encountered, it is immediately returned.
/// Otherwise, a simple `Ok(())` is returned.
#[inline]
pub fn fold_<T,E,Iter:Iterator<Result<T,E>>>(iterator: Iter) -> Result<(),E> {
    fold(iterator, (), |_, _| ())
}

/////////////////////////////////////////////////////////////////////////////
// Tests
/////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    use iter::range;

    pub fn op1() -> Result<int, ~str> { Ok(666) }
    pub fn op2() -> Result<int, ~str> { Err(~"sadface") }

    #[test]
    pub fn test_and() {
        assert_eq!(op1().and(Ok(667)).unwrap(), 667);
        assert_eq!(op1().and(Err::<(), ~str>(~"bad")).unwrap_err(), ~"bad");

        assert_eq!(op2().and(Ok(667)).unwrap_err(), ~"sadface");
        assert_eq!(op2().and(Err::<(), ~str>(~"bad")).unwrap_err(), ~"sadface");
    }

    #[test]
    pub fn test_and_then() {
        assert_eq!(op1().and_then(|i| Ok::<int, ~str>(i + 1)).unwrap(), 667);
        assert_eq!(op1().and_then(|_| Err::<int, ~str>(~"bad")).unwrap_err(), ~"bad");

        assert_eq!(op2().and_then(|i| Ok::<int, ~str>(i + 1)).unwrap_err(), ~"sadface");
        assert_eq!(op2().and_then(|_| Err::<int, ~str>(~"bad")).unwrap_err(), ~"sadface");
    }

    #[test]
    pub fn test_or() {
        assert_eq!(op1().or(Ok(667)).unwrap(), 666);
        assert_eq!(op1().or(Err(~"bad")).unwrap(), 666);

        assert_eq!(op2().or(Ok(667)).unwrap(), 667);
        assert_eq!(op2().or(Err(~"bad")).unwrap_err(), ~"bad");
    }

    #[test]
    pub fn test_or_else() {
        assert_eq!(op1().or_else(|_| Ok::<int, ~str>(667)).unwrap(), 666);
        assert_eq!(op1().or_else(|e| Err::<int, ~str>(e + "!")).unwrap(), 666);

        assert_eq!(op2().or_else(|_| Ok::<int, ~str>(667)).unwrap(), 667);
        assert_eq!(op2().or_else(|e| Err::<int, ~str>(e + "!")).unwrap_err(), ~"sadface!");
    }

    #[test]
    pub fn test_impl_map() {
        assert_eq!(Ok::<~str, ~str>(~"a").map(|x| x + "b"), Ok(~"ab"));
        assert_eq!(Err::<~str, ~str>(~"a").map(|x| x + "b"), Err(~"a"));
    }

    #[test]
    pub fn test_impl_map_err() {
        assert_eq!(Ok::<~str, ~str>(~"a").map_err(|x| x + "b"), Ok(~"a"));
        assert_eq!(Err::<~str, ~str>(~"a").map_err(|x| x + "b"), Err(~"ab"));
    }

    #[test]
    fn test_collect() {
        let v: Result<~[int], ()> = collect(range(0, 0).map(|_| Ok::<int, ()>(0)));
        assert_eq!(v, Ok(~[]));

        let v: Result<~[int], ()> = collect(range(0, 3).map(|x| Ok::<int, ()>(x)));
        assert_eq!(v, Ok(~[0, 1, 2]));

        let v: Result<~[int], int> = collect(range(0, 3)
                                             .map(|x| if x > 1 { Err(x) } else { Ok(x) }));
        assert_eq!(v, Err(2));

        // test that it does not take more elements than it needs
        let functions = [|| Ok(()), || Err(1), || fail!()];

        let v: Result<~[()], int> = collect(functions.iter().map(|f| (*f)()));
        assert_eq!(v, Err(1));
    }

    #[test]
    fn test_fold() {
        assert_eq!(fold_(range(0, 0)
                        .map(|_| Ok::<(), ()>(()))),
                   Ok(()));
        assert_eq!(fold(range(0, 3)
                        .map(|x| Ok::<int, ()>(x)),
                        0, |a, b| a + b),
                   Ok(3));
        assert_eq!(fold_(range(0, 3)
                        .map(|x| if x > 1 { Err(x) } else { Ok(()) })),
                   Err(2));

        // test that it does not take more elements than it needs
        let functions = [|| Ok(()), || Err(1), || fail!()];

        assert_eq!(fold_(functions.iter()
                        .map(|f| (*f)())),
                   Err(1));
    }

    #[test]
    pub fn test_to_str() {
        let ok: Result<int, ~str> = Ok(100);
        let err: Result<int, ~str> = Err(~"Err");

        assert_eq!(ok.to_str(), ~"Ok(100)");
        assert_eq!(err.to_str(), ~"Err(Err)");
    }

    #[test]
    pub fn test_fmt_default() {
        let ok: Result<int, ~str> = Ok(100);
        let err: Result<int, ~str> = Err(~"Err");

        assert_eq!(format!("{}", ok), ~"Ok(100)");
        assert_eq!(format!("{}", err), ~"Err(Err)");
    }
}
