// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use kinds::Copy;
use util;
use num::Zero;
use iter::BaseIter;

#[cfg(test)] use ptr;
#[cfg(test)] use str;

/// The option type
#[deriving_eq]
pub enum Option<T> {
    None,
    Some(T),
}

impl<T:Ord> Ord for Option<T> {
    pure fn lt(&self, other: &Option<T>) -> bool {
        match (self, other) {
            (&None, &None) => false,
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&Some(ref a), &Some(ref b)) => *a < *b
        }
    }

    pure fn le(&self, other: &Option<T>) -> bool {
        match (self, other) {
            (&None, &None) => true,
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&Some(ref a), &Some(ref b)) => *a <= *b
        }
    }

    pure fn ge(&self, other: &Option<T>) -> bool {
        ! (self < other)
    }

    pure fn gt(&self, other: &Option<T>) -> bool {
        ! (self <= other)
    }
}

#[inline(always)]
pub pure fn get<T:Copy>(opt: Option<T>) -> T {
    /*!
    Gets the value out of an option

    # Failure

    Fails if the value equals `None`

    # Safety note

    In general, because this function may fail, its use is discouraged
    (calling `get` on `None` is akin to dereferencing a null pointer).
    Instead, prefer to use pattern matching and handle the `None`
    case explicitly.
    */

    match opt {
      Some(copy x) => return x,
      None => fail!(~"option::get none")
    }
}

#[inline(always)]
pub pure fn get_ref<T>(opt: &r/Option<T>) -> &r/T {
    /*!
    Gets an immutable reference to the value inside an option.

    # Failure

    Fails if the value equals `None`

    # Safety note

    In general, because this function may fail, its use is discouraged
    (calling `get` on `None` is akin to dereferencing a null pointer).
    Instead, prefer to use pattern matching and handle the `None`
    case explicitly.
     */
    match *opt {
        Some(ref x) => x,
        None => fail!(~"option::get_ref none")
    }
}

#[inline(always)]
pub pure fn map<T, U>(opt: &r/Option<T>, f: fn(x: &r/T) -> U) -> Option<U> {
    //! Maps a `some` value by reference from one type to another

    match *opt { Some(ref x) => Some(f(x)), None => None }
}

#[inline(always)]
pub pure fn map_consume<T, U>(opt: Option<T>,
                              f: fn(v: T) -> U) -> Option<U> {
    /*!
     * As `map`, but consumes the option and gives `f` ownership to avoid
     * copying.
     */
    match opt { None => None, Some(v) => Some(f(v)) }
}

#[inline(always)]
pub pure fn chain<T, U>(opt: Option<T>,
                        f: fn(t: T) -> Option<U>) -> Option<U> {
    /*!
     * Update an optional value by optionally running its content through a
     * function that returns an option.
     */

    match opt {
        Some(t) => f(t),
        None => None
    }
}

#[inline(always)]
pub pure fn chain_ref<T, U>(opt: &Option<T>,
                            f: fn(x: &T) -> Option<U>) -> Option<U> {
    /*!
     * Update an optional value by optionally running its content by reference
     * through a function that returns an option.
     */

    match *opt { Some(ref x) => f(x), None => None }
}

#[inline(always)]
pub pure fn or<T>(opta: Option<T>, optb: Option<T>) -> Option<T> {
    /*!
     * Returns the leftmost Some() value, or None if both are None.
     */
    match opta {
        Some(opta) => Some(opta),
        _ => optb
    }
}

#[inline(always)]
pub pure fn while_some<T>(x: Option<T>, blk: fn(v: T) -> Option<T>) {
    //! Applies a function zero or more times until the result is none.

    let mut opt = x;
    while opt.is_some() {
        opt = blk(unwrap(opt));
    }
}

#[inline(always)]
pub pure fn is_none<T>(opt: &Option<T>) -> bool {
    //! Returns true if the option equals `none`

    match *opt { None => true, Some(_) => false }
}

#[inline(always)]
pub pure fn is_some<T>(opt: &Option<T>) -> bool {
    //! Returns true if the option contains some value

    !is_none(opt)
}

#[inline(always)]
pub pure fn get_or_zero<T:Copy + Zero>(opt: Option<T>) -> T {
    //! Returns the contained value or zero (for this type)

    match opt { Some(copy x) => x, None => Zero::zero() }
}

#[inline(always)]
pub pure fn get_or_default<T:Copy>(opt: Option<T>, def: T) -> T {
    //! Returns the contained value or a default

    match opt { Some(copy x) => x, None => def }
}

#[inline(always)]
pub pure fn map_default<T, U>(opt: &r/Option<T>, def: U,
                              f: fn(&r/T) -> U) -> U {
    //! Applies a function to the contained value or returns a default

    match *opt { None => def, Some(ref t) => f(t) }
}

#[inline(always)]
pub pure fn unwrap<T>(opt: Option<T>) -> T {
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
    match opt {
        Some(x) => x,
        None => fail!(~"option::unwrap none")
    }
}

#[inline(always)]
pub fn swap_unwrap<T>(opt: &mut Option<T>) -> T {
    /*!
    The option dance. Moves a value out of an option type and returns it,
    replacing the original with `None`.

    # Failure

    Fails if the value equals `None`.
     */
    if opt.is_none() { fail!(~"option::swap_unwrap none") }
    unwrap(util::replace(opt, None))
}

#[inline(always)]
pub pure fn expect<T>(opt: Option<T>, reason: &str) -> T {
    //! As unwrap, but with a specified failure message.
    match opt {
        Some(val) => val,
        None => fail!(reason.to_owned()),
    }
}

impl<T> BaseIter<T> for Option<T> {
    /// Performs an operation on the contained value by reference
    #[inline(always)]
    pure fn each(&self, f: fn(x: &self/T) -> bool) {
        match *self { None => (), Some(ref t) => { f(t); } }
    }

    #[inline(always)]
    pure fn size_hint(&self) -> Option<uint> {
        if self.is_some() { Some(1) } else { Some(0) }
    }
}

pub impl<T> Option<T> {
    /// Returns true if the option equals `none`
    #[inline(always)]
    pure fn is_none(&self) -> bool { is_none(self) }

    /// Returns true if the option contains some value
    #[inline(always)]
    pure fn is_some(&self) -> bool { is_some(self) }

    /**
     * Update an optional value by optionally running its content by reference
     * through a function that returns an option.
     */
    #[inline(always)]
    pure fn chain_ref<U>(&self, f: fn(x: &T) -> Option<U>) -> Option<U> {
        chain_ref(self, f)
    }

    /// Maps a `some` value from one type to another by reference
    #[inline(always)]
    pure fn map<U>(&self, f: fn(&self/T) -> U) -> Option<U> { map(self, f) }

    /// As `map`, but consumes the option and gives `f` ownership to avoid
    /// copying.
    #[inline(always)]
    pure fn map_consume<U>(self, f: fn(v: T) -> U) -> Option<U> {
        map_consume(self, f)
    }

    /// Applies a function to the contained value or returns a default
    #[inline(always)]
    pure fn map_default<U>(&self, def: U, f: fn(&self/T) -> U) -> U {
        map_default(self, def, f)
    }

    /// As `map_default`, but consumes the option and gives `f`
    /// ownership to avoid copying.
    #[inline(always)]
    pure fn map_consume_default<U>(self, def: U, f: fn(v: T) -> U) -> U {
        match self { None => def, Some(v) => f(v) }
    }

    /// Apply a function to the contained value or do nothing
    fn mutate(&mut self, f: fn(T) -> T) {
        if self.is_some() {
            *self = Some(f(self.swap_unwrap()));
        }
    }

    /// Apply a function to the contained value or set it to a default
    fn mutate_default(&mut self, def: T, f: fn(T) -> T) {
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
    pure fn get_ref(&self) -> &self/T { get_ref(self) }

    /**
     * Gets the value out of an option without copying.
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    #[inline(always)]
    pure fn unwrap(self) -> T { unwrap(self) }

    /**
     * The option dance. Moves a value out of an option type and returns it,
     * replacing the original with `None`.
     *
     * # Failure
     *
     * Fails if the value equals `None`.
     */
    #[inline(always)]
    fn swap_unwrap(&mut self) -> T { swap_unwrap(self) }

    /**
     * Gets the value out of an option, printing a specified message on
     * failure
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    #[inline(always)]
    pure fn expect(self, reason: &str) -> T { expect(self, reason) }
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
    pure fn get(self) -> T { get(self) }

    #[inline(always)]
    pure fn get_or_default(self, def: T) -> T { get_or_default(self, def) }

    /// Applies a function zero or more times until the result is none.
    #[inline(always)]
    pure fn while_some(self, blk: fn(v: T) -> Option<T>) {
        while_some(self, blk)
    }
}

pub impl<T:Copy + Zero> Option<T> {
    #[inline(always)]
    pure fn get_or_zero(self) -> T { get_or_zero(self) }
}

#[test]
fn test_unwrap_ptr() {
    let x = ~0;
    let addr_x = ptr::addr_of(&(*x));
    let opt = Some(x);
    let y = unwrap(opt);
    let addr_y = ptr::addr_of(&(*y));
    fail_unless!(addr_x == addr_y);
}

#[test]
fn test_unwrap_str() {
    let x = ~"test";
    let addr_x = str::as_buf(x, |buf, _len| buf);
    let opt = Some(x);
    let y = unwrap(opt);
    let addr_y = str::as_buf(y, |buf, _len| buf);
    fail_unless!(addr_x == addr_y);
}

#[test]
fn test_unwrap_resource() {
    struct R {
       i: @mut int,
    }

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
        let _y = unwrap(opt);
    }
    fail_unless!(*i == 1);
}

#[test]
fn test_option_dance() {
    let x = Some(());
    let mut y = Some(5);
    let mut y2 = 0;
    for x.each |_x| {
        y2 = swap_unwrap(&mut y);
    }
    fail_unless!(y2 == 5);
    fail_unless!(y.is_none());
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_option_too_much_dance() {
    let mut y = Some(util::NonCopyable());
    let _y2 = swap_unwrap(&mut y);
    let _y3 = swap_unwrap(&mut y);
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
    fail_unless!(i == 11);
}

#[test]
fn test_get_or_zero() {
    let some_stuff = Some(42);
    fail_unless!(some_stuff.get_or_zero() == 42);
    let no_stuff: Option<int> = None;
    fail_unless!(no_stuff.get_or_zero() == 0);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
