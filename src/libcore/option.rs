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
let unwrapped_msg = match move msg {
    Some(move m) => m,
    None => ~"default message"
};
~~~

*/

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::Eq;
use kinds::Copy;
use option;
use ptr;
use str;
use util;
use num::Zero;

/// The option type
#[deriving_eq]
pub enum Option<T> {
    None,
    Some(T),
}

pub pure fn get<T: Copy>(opt: Option<T>) -> T {
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
      None => fail ~"option::get none"
    }
}

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
        None => fail ~"option::get_ref none"
    }
}

pub pure fn map<T, U>(opt: &Option<T>, f: fn(x: &T) -> U) -> Option<U> {
    //! Maps a `some` value by reference from one type to another

    match *opt { Some(ref x) => Some(f(x)), None => None }
}

pub pure fn map_consume<T, U>(opt: Option<T>,
                              f: fn(v: T) -> U) -> Option<U> {
    /*!
     * As `map`, but consumes the option and gives `f` ownership to avoid
     * copying.
     */
    if opt.is_some() { Some(f(option::unwrap(move opt))) } else { None }
}

pub pure fn chain<T, U>(opt: Option<T>,
                        f: fn(t: T) -> Option<U>) -> Option<U> {
    /*!
     * Update an optional value by optionally running its content through a
     * function that returns an option.
     */

    match move opt {
        Some(move t) => f(move t),
        None => None
    }
}

pub pure fn chain_ref<T, U>(opt: &Option<T>,
                            f: fn(x: &T) -> Option<U>) -> Option<U> {
    /*!
     * Update an optional value by optionally running its content by reference
     * through a function that returns an option.
     */

    match *opt { Some(ref x) => f(x), None => None }
}

pub pure fn or<T>(opta: Option<T>, optb: Option<T>) -> Option<T> {
    /*!
     * Returns the leftmost some() value, or none if both are none.
     */
    match move opta {
        Some(move opta) => Some(move opta),
        _ => move optb
    }
}

#[inline(always)]
pub pure fn while_some<T>(x: Option<T>, blk: fn(v: T) -> Option<T>) {
    //! Applies a function zero or more times until the result is none.

    let mut opt = move x;
    while opt.is_some() {
        opt = blk(unwrap(move opt));
    }
}

pub pure fn is_none<T>(opt: &Option<T>) -> bool {
    //! Returns true if the option equals `none`

    match *opt { None => true, Some(_) => false }
}

pub pure fn is_some<T>(opt: &Option<T>) -> bool {
    //! Returns true if the option contains some value

    !is_none(opt)
}

pub pure fn get_or_zero<T: Copy Zero>(opt: Option<T>) -> T {
    //! Returns the contained value or zero (for this type)

    match opt { Some(copy x) => x, None => Zero::zero() }
}

pub pure fn get_or_default<T: Copy>(opt: Option<T>, def: T) -> T {
    //! Returns the contained value or a default

    match opt { Some(copy x) => x, None => def }
}

pub pure fn map_default<T, U>(opt: &Option<T>, def: U,
                              f: fn(x: &T) -> U) -> U {
    //! Applies a function to the contained value or returns a default

    match *opt { None => move def, Some(ref t) => f(t) }
}

pub pure fn iter<T>(opt: &Option<T>, f: fn(x: &T)) {
    //! Performs an operation on the contained value by reference
    match *opt { None => (), Some(ref t) => f(t) }
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
    match move opt {
        Some(move x) => move x,
        None => fail ~"option::unwrap none"
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
    if opt.is_none() { fail ~"option::swap_unwrap none" }
    unwrap(util::replace(opt, None))
}

pub pure fn expect<T>(opt: Option<T>, reason: &str) -> T {
    //! As unwrap, but with a specified failure message.
    match move opt {
        Some(move val) => val,
        None => fail reason.to_owned(),
    }
}

impl<T> Option<T> {
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
    pure fn map<U>(&self, f: fn(x: &T) -> U) -> Option<U> { map(self, f) }

    /// Applies a function to the contained value or returns a default
    #[inline(always)]
    pure fn map_default<U>(&self, def: U, f: fn(x: &T) -> U) -> U {
        map_default(self, move def, f)
    }

    /// Performs an operation on the contained value by reference
    #[inline(always)]
    pure fn iter(&self, f: fn(x: &T)) { iter(self, f) }

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

impl<T: Copy> Option<T> {
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

impl<T: Copy Zero> Option<T> {
    #[inline(always)]
    pure fn get_or_zero(self) -> T { get_or_zero(self) }
}

#[test]
fn test_unwrap_ptr() {
    let x = ~0;
    let addr_x = ptr::addr_of(&(*x));
    let opt = Some(move x);
    let y = unwrap(move opt);
    let addr_y = ptr::addr_of(&(*y));
    assert addr_x == addr_y;
}

#[test]
fn test_unwrap_str() {
    let x = ~"test";
    let addr_x = str::as_buf(x, |buf, _len| buf);
    let opt = Some(move x);
    let y = unwrap(move opt);
    let addr_y = str::as_buf(y, |buf, _len| buf);
    assert addr_x == addr_y;
}

#[test]
fn test_unwrap_resource() {
    struct R {
       i: @mut int,
       drop { *(self.i) += 1; }
    }

    fn R(i: @mut int) -> R {
        R {
            i: i
        }
    }

    let i = @mut 0;
    {
        let x = R(i);
        let opt = Some(move x);
        let _y = unwrap(move opt);
    }
    assert *i == 1;
}

#[test]
fn test_option_dance() {
    let x = Some(());
    let mut y = Some(5);
    let mut y2 = 0;
    do x.iter |_x| {
        y2 = swap_unwrap(&mut y);
    }
    assert y2 == 5;
    assert y.is_none();
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
    assert i == 11;
}

#[test]
fn test_get_or_zero() {
    let some_stuff = Some(42);
    assert some_stuff.get_or_zero() == 42;
    let no_stuff: Option<int> = None;
    assert no_stuff.get_or_zero() == 0;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
