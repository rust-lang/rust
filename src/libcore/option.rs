// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Operations on the ubiquitous `Option` type.

Type `Option` represents an optional value.

Every `Option<T>` value can either be `Some(T)` or `None`. Where in other
languages you might use a nullable type, in Rust you would use an option
type.

Options are most commonly used with pattern matching to query the presence
of a value and take action, always accounting for the `None` case.

# Example

~~~
let msg = Some(~"howdy");

// Take a reference to the contained string
match msg {
    Some(ref m) => io::println(m),
    None => ()
}

// Remove the contained string, destroying the Option
let unwrapped_msg = match msg {
    Some(m) => m,
    None => ~"default message"
};
~~~

*/

use cmp::{Eq,Ord};
use ops::Add;
use kinds::Copy;
use util;
use num::Zero;
use old_iter::{BaseIter, MutableIter, ExtendedIter};
use old_iter;
use str::StrSlice;

#[cfg(test)] use str;

/// The option type
#[deriving(Clone, Eq)]
pub enum Option<T> {
    None,
    Some(T),
}

impl<T:Ord> Ord for Option<T> {
    fn lt(&self, other: &Option<T>) -> bool {
        match (self, other) {
            (&None, &None) => false,
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&Some(ref a), &Some(ref b)) => *a < *b
        }
    }

    fn le(&self, other: &Option<T>) -> bool {
        match (self, other) {
            (&None, &None) => true,
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&Some(ref a), &Some(ref b)) => *a <= *b
        }
    }

    fn ge(&self, other: &Option<T>) -> bool {
        ! (self < other)
    }

    fn gt(&self, other: &Option<T>) -> bool {
        ! (self <= other)
    }
}

impl<T: Copy + Add<T,T>> Add<Option<T>, Option<T>> for Option<T> {
    #[inline(always)]
    fn add(&self, other: &Option<T>) -> Option<T> {
        match (*self, *other) {
            (None, None) => None,
            (_, None) => *self,
            (None, _) => *other,
            (Some(ref lhs), Some(ref rhs)) => Some(*lhs + *rhs)
        }
    }
}

impl<T> BaseIter<T> for Option<T> {
    /// Performs an operation on the contained value by reference
    #[inline(always)]
    #[cfg(stage0)]
    fn each<'a>(&'a self, f: &fn(x: &'a T) -> bool) {
        match *self { None => (), Some(ref t) => { f(t); } }
    }
    /// Performs an operation on the contained value by reference
    #[inline(always)]
    #[cfg(not(stage0))]
    fn each<'a>(&'a self, f: &fn(x: &'a T) -> bool) -> bool {
        match *self { None => true, Some(ref t) => { f(t) } }
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<uint> {
        if self.is_some() { Some(1) } else { Some(0) }
    }
}

impl<T> MutableIter<T> for Option<T> {
    #[cfg(stage0)]
    #[inline(always)]
    fn each_mut<'a>(&'a mut self, f: &fn(&'a mut T) -> bool) {
        match *self { None => (), Some(ref mut t) => { f(t); } }
    }
    #[cfg(not(stage0))]
    #[inline(always)]
    fn each_mut<'a>(&'a mut self, f: &fn(&'a mut T) -> bool) -> bool {
        match *self { None => true, Some(ref mut t) => { f(t) } }
    }
}

impl<A> ExtendedIter<A> for Option<A> {
    #[cfg(stage0)]
    pub fn eachi(&self, blk: &fn(uint, v: &A) -> bool) {
        old_iter::eachi(self, blk)
    }
    #[cfg(not(stage0))]
    pub fn eachi(&self, blk: &fn(uint, v: &A) -> bool) -> bool {
        old_iter::eachi(self, blk)
    }
    pub fn all(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::all(self, blk)
    }
    pub fn any(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::any(self, blk)
    }
    pub fn foldl<B>(&self, b0: B, blk: &fn(&B, &A) -> B) -> B {
        old_iter::foldl(self, b0, blk)
    }
    pub fn position(&self, f: &fn(&A) -> bool) -> Option<uint> {
        old_iter::position(self, f)
    }
    fn map_to_vec<B>(&self, op: &fn(&A) -> B) -> ~[B] {
        old_iter::map_to_vec(self, op)
    }
    fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: &fn(&A) -> IB)
        -> ~[B] {
        old_iter::flat_map_to_vec(self, op)
    }
}

pub impl<T> Option<T> {
    /// Returns true if the option equals `none`
    fn is_none(&const self) -> bool {
        match *self { None => true, Some(_) => false }
    }

    /// Returns true if the option contains some value
    #[inline(always)]
    fn is_some(&const self) -> bool { !self.is_none() }

    #[inline(always)]
    fn chain<U>(self, f: &fn(t: T) -> Option<U>) -> Option<U> {
        /*!
         * Update an optional value by optionally running its content through a
         * function that returns an option.
         */

        match self {
            Some(t) => f(t),
            None => None
        }
    }

    #[inline(always)]
    fn or(self, optb: Option<T>) -> Option<T> {
        /*!
         * Returns the leftmost Some() value, or None if both are None.
         */
        match self {
            Some(opta) => Some(opta),
            _ => optb
        }
    }

    /**
     * Update an optional value by optionally running its content by reference
     * through a function that returns an option.
     */
    #[inline(always)]
    fn chain_ref<'a, U>(&'a self, f: &fn(x: &'a T) -> Option<U>) -> Option<U> {
        match *self { Some(ref x) => f(x), None => None }
    }

    /// Maps a `some` value from one type to another by reference
    #[inline(always)]
    fn map<'a, U>(&self, f: &fn(&'a T) -> U) -> Option<U> {
        match *self { Some(ref x) => Some(f(x)), None => None }
    }

    /// As `map`, but consumes the option and gives `f` ownership to avoid
    /// copying.
    #[inline(always)]
    fn map_consume<U>(self, f: &fn(v: T) -> U) -> Option<U> {
        match self { None => None, Some(v) => Some(f(v)) }
    }

    /// Applies a function to the contained value or returns a default
    #[inline(always)]
    fn map_default<'a, U>(&'a self, def: U, f: &fn(&'a T) -> U) -> U {
        match *self { None => def, Some(ref t) => f(t) }
    }

    /// As `map_default`, but consumes the option and gives `f`
    /// ownership to avoid copying.
    #[inline(always)]
    fn map_consume_default<U>(self, def: U, f: &fn(v: T) -> U) -> U {
        match self { None => def, Some(v) => f(v) }
    }

    /// Apply a function to the contained value or do nothing
    fn mutate(&mut self, f: &fn(T) -> T) {
        if self.is_some() {
            *self = Some(f(self.swap_unwrap()));
        }
    }

    /// Apply a function to the contained value or set it to a default
    fn mutate_default(&mut self, def: T, f: &fn(T) -> T) {
        if self.is_some() {
            *self = Some(f(self.swap_unwrap()));
        } else {
            *self = Some(def);
        }
    }

    /**
    Gets an immutable reference to the value inside an option.

    # Failure

    Fails if the value equals `None`

    # Safety note

    In general, because this function may fail, its use is discouraged
    (calling `get` on `None` is akin to dereferencing a null pointer).
    Instead, prefer to use pattern matching and handle the `None`
    case explicitly.
     */
    #[inline(always)]
    fn get_ref<'a>(&'a self) -> &'a T {
        match *self {
          Some(ref x) => x,
          None => fail!(~"option::get_ref none")
        }
    }

    /**
    Gets a mutable reference to the value inside an option.

    # Failure

    Fails if the value equals `None`

    # Safety note

    In general, because this function may fail, its use is discouraged
    (calling `get` on `None` is akin to dereferencing a null pointer).
    Instead, prefer to use pattern matching and handle the `None`
    case explicitly.
     */
    #[inline(always)]
    fn get_mut_ref<'a>(&'a mut self) -> &'a mut T {
        match *self {
          Some(ref mut x) => x,
          None => fail!(~"option::get_mut_ref none")
        }
    }

    #[inline(always)]
    fn unwrap(self) -> T {
        /*!
        Moves a value out of an option type and returns it.

        Useful primarily for getting strings, vectors and unique pointers out
        of option types without copying them.

        # Failure

        Fails if the value equals `None`.

        # Safety note

        In general, because this function may fail, its use is discouraged.
        Instead, prefer to use pattern matching and handle the `None`
        case explicitly.
         */
        match self {
          Some(x) => x,
          None => fail!(~"option::unwrap none")
        }
    }

    /**
     * The option dance. Moves a value out of an option type and returns it,
     * replacing the original with `None`.
     *
     * # Failure
     *
     * Fails if the value equals `None`.
     */
    #[inline(always)]
    fn swap_unwrap(&mut self) -> T {
        if self.is_none() { fail!(~"option::swap_unwrap none") }
        util::replace(self, None).unwrap()
    }

    /**
     * Gets the value out of an option, printing a specified message on
     * failure
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    #[inline(always)]
    fn expect(self, reason: &str) -> T {
        match self {
          Some(val) => val,
          None => fail!(reason.to_owned()),
        }
    }
}

pub impl<T:Copy> Option<T> {
    /**
    Gets the value out of an option

    # Failure

    Fails if the value equals `None`

    # Safety note

    In general, because this function may fail, its use is discouraged
    (calling `get` on `None` is akin to dereferencing a null pointer).
    Instead, prefer to use pattern matching and handle the `None`
    case explicitly.
    */
    #[inline(always)]
    fn get(self) -> T {
        match self {
          Some(copy x) => return x,
          None => fail!(~"option::get none")
        }
    }

    /// Returns the contained value or a default
    #[inline(always)]
    fn get_or_default(self, def: T) -> T {
        match self { Some(copy x) => x, None => def }
    }

    /// Applies a function zero or more times until the result is none.
    #[inline(always)]
    fn while_some(self, blk: &fn(v: T) -> Option<T>) {
        let mut opt = self;
        while opt.is_some() {
            opt = blk(opt.unwrap());
        }
    }
}

pub impl<T:Copy + Zero> Option<T> {
    /// Returns the contained value or zero (for this type)
    #[inline(always)]
    fn get_or_zero(self) -> T {
        match self { Some(copy x) => x, None => Zero::zero() }
    }
}

#[test]
fn test_unwrap_ptr() {
    unsafe {
        let x = ~0;
        let addr_x: *int = ::cast::transmute(&*x);
        let opt = Some(x);
        let y = opt.unwrap();
        let addr_y: *int = ::cast::transmute(&*y);
        assert!(addr_x == addr_y);
    }
}

#[test]
fn test_unwrap_str() {
    let x = ~"test";
    let addr_x = str::as_buf(x, |buf, _len| buf);
    let opt = Some(x);
    let y = opt.unwrap();
    let addr_y = str::as_buf(y, |buf, _len| buf);
    assert!(addr_x == addr_y);
}

#[test]
fn test_unwrap_resource() {
    struct R {
       i: @mut int,
    }

    #[unsafe_destructor]
    impl ::ops::Drop for R {
       fn finalize(&self) { *(self.i) += 1; }
    }

    fn R(i: @mut int) -> R {
        R {
            i: i
        }
    }

    let i = @mut 0;
    {
        let x = R(i);
        let opt = Some(x);
        let _y = opt.unwrap();
    }
    assert!(*i == 1);
}

#[test]
fn test_option_dance() {
    let x = Some(());
    let mut y = Some(5);
    let mut y2 = 0;
    for x.each |_x| {
        y2 = y.swap_unwrap();
    }
    assert!(y2 == 5);
    assert!(y.is_none());
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_option_too_much_dance() {
    let mut y = Some(util::NonCopyable());
    let _y2 = y.swap_unwrap();
    let _y3 = y.swap_unwrap();
}

#[test]
fn test_option_while_some() {
    let mut i = 0;
    do Some(10).while_some |j| {
        i += 1;
        if (j > 0) {
            Some(j-1)
        } else {
            None
        }
    }
    assert!(i == 11);
}

#[test]
fn test_get_or_zero() {
    let some_stuff = Some(42);
    assert!(some_stuff.get_or_zero() == 42);
    let no_stuff: Option<int> = None;
    assert!(no_stuff.get_or_zero() == 0);
}
