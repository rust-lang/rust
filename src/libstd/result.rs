// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A type representing either success or failure

#[allow(missing_doc)];

use clone::Clone;
use cmp::Eq;
use either;
use iterator::Iterator;
use option::{None, Option, Some};
use vec;
use vec::{OwnedVector, ImmutableVector};
use container::Container;

/// The result type
#[deriving(Clone, Eq)]
pub enum Result<T, U> {
    /// Contains the successful result value
    Ok(T),
    /// Contains the error value
    Err(U)
}

impl<T, E> Result<T, E> {
    /**
     * Convert to the `either` type
     *
     * `Ok` result variants are converted to `either::Right` variants, `Err`
     * result variants are converted to `either::Left`.
     */
    #[inline]
    pub fn to_either(self)-> either::Either<E, T>{
        match self {
            Ok(t) => either::Right(t),
            Err(e) => either::Left(e),
        }
    }

    /**
     * Get a reference to the value out of a successful result
     *
     * # Failure
     *
     * If the result is an error
     */
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a T {
        match *self {
            Ok(ref t) => t,
            Err(ref e) => fail!("get_ref called on `Err` result: %?", *e),
        }
    }

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

    /**
     * Call a method based on a previous result
     *
     * If `self` is `Ok` then the value is extracted and passed to `op`
     * whereupon `op`s result is returned. if `self` is `Err` then it is
     * immediately returned. This function can be used to compose the results
     * of two functions.
     *
     * Example:
     *
     *     do read_file(file).iter |buf| {
     *         print_buf(buf)
     *     }
     */
    #[inline]
    pub fn iter(&self, f: &fn(&T)) {
        match *self {
            Ok(ref t) => f(t),
            Err(_) => (),
        }
    }

    /**
     * Call a method based on a previous result
     *
     * If `self` is `Err` then the value is extracted and passed to `op`
     * whereupon `op`s result is returned. if `self` is `Ok` then it is
     * immediately returned.  This function can be used to pass through a
     * successful result while handling an error.
     */
    #[inline]
    pub fn iter_err(&self, f: &fn(&E)) {
        match *self {
            Ok(_) => (),
            Err(ref e) => f(e),
        }
    }

    /// Unwraps a result, assuming it is an `Ok(T)`
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            Ok(t) => t,
            Err(_) => fail!("unwrap called on an `Err` result"),
        }
    }

    /// Unwraps a result, assuming it is an `Err(U)`
    #[inline]
    pub fn unwrap_err(self) -> E {
        match self {
            Err(e) => e,
            Ok(_) => fail!("unwrap called on an `Ok` result"),
        }
    }

    /**
     * Call a method based on a previous result
     *
     * If `self` is `Ok` then the value is extracted and passed to `op`
     * whereupon `op`s result is returned. if `self` is `Err` then it is
     * immediately returned. This function can be used to compose the results
     * of two functions.
     *
     * Example:
     *
     *     let res = do read_file(file) |buf| {
     *         Ok(parse_bytes(buf))
     *     };
     */
    #[inline]
    pub fn chain<U>(self, op: &fn(T) -> Result<U, E>) -> Result<U, E> {
        match self {
            Ok(t) => op(t),
            Err(e) => Err(e),
        }
    }

    /**
     * Call a function based on a previous result
     *
     * If `self` is `Err` then the value is extracted and passed to `op`
     * whereupon `op`s result is returned. if `self` is `Ok` then it is
     * immediately returned.  This function can be used to pass through a
     * successful result while handling an error.
     */
    #[inline]
    pub fn chain_err<F>(self, op: &fn(E) -> Result<T, F>) -> Result<T, F> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => op(e),
        }
    }
}

impl<T: Clone, E> Result<T, E> {
    /**
     * Get the value out of a successful result
     *
     * # Failure
     *
     * If the result is an error
     */
    #[inline]
    pub fn get(&self) -> T {
        match *self {
            Ok(ref t) => t.clone(),
            Err(ref e) => fail!("get called on `Err` result: %?", *e),
        }
    }

    /**
     * Call a method based on a previous result
     *
     * If `self` is `Err` then the value is extracted and passed to `op`
     * whereupon `op`s result is wrapped in an `Err` and returned. if `self` is
     * `Ok` then it is immediately returned.  This function can be used to pass
     * through a successful result while handling an error.
     */
    #[inline]
    pub fn map_err<F:Clone>(&self, op: &fn(&E) -> F) -> Result<T,F> {
        match *self {
            Ok(ref t) => Ok(t.clone()),
            Err(ref e) => Err(op(e))
        }
    }
}

impl<T, E: Clone> Result<T, E> {
    /**
     * Get the value out of an error result
     *
     * # Failure
     *
     * If the result is not an error
     */
    #[inline]
    pub fn get_err(&self) -> E {
        match *self {
            Err(ref e) => e.clone(),
            Ok(_) => fail!("get_err called on `Ok` result")
        }
    }

    /**
     * Call a method based on a previous result
     *
     * If `self` is `Ok` then the value is extracted and passed to `op`
     * whereupon `op`s result is wrapped in `Ok` and returned. if `self` is
     * `Err` then it is immediately returned.  This function can be used to
     * compose the results of two functions.
     *
     * Example:
     *
     *     let res = do read_file(file).map |buf| {
     *         parse_bytes(buf)
     *     };
     */
    #[inline]
    pub fn map<U:Clone>(&self, op: &fn(&T) -> U) -> Result<U,E> {
        match *self {
            Ok(ref t) => Ok(op(t)),
            Err(ref e) => Err(e.clone())
        }
    }
}

/**
 * Maps each element in the vector `ts` using the operation `op`.  Should an
 * error occur, no further mappings are performed and the error is returned.
 * Should no error occur, a vector containing the result of each map is
 * returned.
 *
 * Here is an example which increments every integer in a vector,
 * checking for overflow:
 *
 *     fn inc_conditionally(x: uint) -> result<uint,str> {
 *         if x == uint::max_value { return Err("overflow"); }
 *         else { return Ok(x+1u); }
 *     }
 *     map(~[1u, 2u, 3u], inc_conditionally).chain {|incd|
 *         assert!(incd == ~[2u, 3u, 4u]);
 *     }
 */
#[inline]
pub fn map_vec<T,U,V>(ts: &[T], op: &fn(&T) -> Result<V,U>)
                      -> Result<~[V],U> {
    let mut vs: ~[V] = vec::with_capacity(ts.len());
    foreach t in ts.iter() {
        match op(t) {
          Ok(v) => vs.push(v),
          Err(u) => return Err(u)
        }
    }
    return Ok(vs);
}

#[inline]
#[allow(missing_doc)]
pub fn map_opt<T,
               U,
               V>(
               o_t: &Option<T>,
               op: &fn(&T) -> Result<V,U>)
               -> Result<Option<V>,U> {
    match *o_t {
        None => Ok(None),
        Some(ref t) => match op(t) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e)
        }
    }
}

/**
 * Same as map, but it operates over two parallel vectors.
 *
 * A precondition is used here to ensure that the vectors are the same
 * length.  While we do not often use preconditions in the standard
 * library, a precondition is used here because result::t is generally
 * used in 'careful' code contexts where it is both appropriate and easy
 * to accommodate an error like the vectors being of different lengths.
 */
#[inline]
pub fn map_vec2<S,T,U,V>(ss: &[S], ts: &[T],
                op: &fn(&S,&T) -> Result<V,U>) -> Result<~[V],U> {

    assert!(vec::same_length(ss, ts));
    let n = ts.len();
    let mut vs = vec::with_capacity(n);
    let mut i = 0u;
    while i < n {
        match op(&ss[i],&ts[i]) {
          Ok(v) => vs.push(v),
          Err(u) => return Err(u)
        }
        i += 1u;
    }
    return Ok(vs);
}

/**
 * Applies op to the pairwise elements from `ss` and `ts`, aborting on
 * error.  This could be implemented using `map_zip()` but it is more efficient
 * on its own as no result vector is built.
 */
#[inline]
pub fn iter_vec2<S,T,U>(ss: &[S], ts: &[T],
                         op: &fn(&S,&T) -> Result<(),U>) -> Result<(),U> {

    assert!(vec::same_length(ss, ts));
    let n = ts.len();
    let mut i = 0u;
    while i < n {
        match op(&ss[i],&ts[i]) {
          Ok(()) => (),
          Err(u) => return Err(u)
        }
        i += 1u;
    }
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use either;

    pub fn op1() -> Result<int, ~str> { Ok(666) }

    pub fn op2(i: int) -> Result<uint, ~str> {
        Ok(i as uint + 1u)
    }

    pub fn op3() -> Result<int, ~str> { Err(~"sadface") }

    #[test]
    pub fn chain_success() {
        assert_eq!(op1().chain(op2).get(), 667u);
    }

    #[test]
    pub fn chain_failure() {
        assert_eq!(op3().chain( op2).get_err(), ~"sadface");
    }

    #[test]
    pub fn test_impl_iter() {
        let mut valid = false;
        Ok::<~str, ~str>(~"a").iter(|_x| valid = true);
        assert!(valid);

        Err::<~str, ~str>(~"b").iter(|_x| valid = false);
        assert!(valid);
    }

    #[test]
    pub fn test_impl_iter_err() {
        let mut valid = true;
        Ok::<~str, ~str>(~"a").iter_err(|_x| valid = false);
        assert!(valid);

        valid = false;
        Err::<~str, ~str>(~"b").iter_err(|_x| valid = true);
        assert!(valid);
    }

    #[test]
    pub fn test_impl_map() {
        assert_eq!(Ok::<~str, ~str>(~"a").map(|_x| ~"b"), Ok(~"b"));
        assert_eq!(Err::<~str, ~str>(~"a").map(|_x| ~"b"), Err(~"a"));
    }

    #[test]
    pub fn test_impl_map_err() {
        assert_eq!(Ok::<~str, ~str>(~"a").map_err(|_x| ~"b"), Ok(~"a"));
        assert_eq!(Err::<~str, ~str>(~"a").map_err(|_x| ~"b"), Err(~"b"));
    }

    #[test]
    pub fn test_get_ref_method() {
        let foo: Result<int, ()> = Ok(100);
        assert_eq!(*foo.get_ref(), 100);
    }

    #[test]
    pub fn test_to_either() {
        let r: Result<int, ()> = Ok(100);
        let err: Result<(), int> = Err(404);

        assert_eq!(r.to_either(), either::Right(100));
        assert_eq!(err.to_either(), either::Left(404));
    }
}
