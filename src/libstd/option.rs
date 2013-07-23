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

use clone::Clone;
use cmp::{Eq,Ord};
use ops::Add;
use util;
use num::Zero;
use iterator::Iterator;
use str::StrSlice;
use clone::DeepClone;

#[cfg(test)] use iterator::IteratorUtil;

/// The option type
#[deriving(Clone, DeepClone, Eq)]
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
        !(self < other)
    }

    fn gt(&self, other: &Option<T>) -> bool {
        !(self <= other)
    }
}

impl<T:Clone+Add<T,T>> Add<Option<T>, Option<T>> for Option<T> {
    #[inline]
    fn add(&self, other: &Option<T>) -> Option<T> {
        match (&*self, &*other) {
            (&None, &None) => None,
            (_, &None) => (*self).clone(),
            (&None, _) => (*other).clone(),
            (&Some(ref lhs), &Some(ref rhs)) => Some(*lhs + *rhs)
        }
    }
}

impl<T> Option<T> {
    /// Return an iterator over the possibly contained value
    #[inline]
    pub fn iter<'r>(&'r self) -> OptionIterator<'r, T> {
        match *self {
            Some(ref x) => OptionIterator{opt: Some(x)},
            None => OptionIterator{opt: None}
        }
    }

    /// Return a mutable iterator over the possibly contained value
    #[inline]
    pub fn mut_iter<'r>(&'r mut self) -> OptionMutIterator<'r, T> {
        match *self {
            Some(ref mut x) => OptionMutIterator{opt: Some(x)},
            None => OptionMutIterator{opt: None}
        }
    }

    /// Returns true if the option equals `none`
    #[inline]
    pub fn is_none(&self) -> bool {
        match *self { None => true, Some(_) => false }
    }

    /// Returns true if the option contains some value
    #[inline]
    pub fn is_some(&self) -> bool { !self.is_none() }

    /// Update an optional value by optionally running its content through a
    /// function that returns an option.
    #[inline]
    pub fn chain<U>(self, f: &fn(t: T) -> Option<U>) -> Option<U> {
        match self {
            Some(t) => f(t),
            None => None
        }
    }

    /// Returns the leftmost Some() value, or None if both are None.
    #[inline]
    pub fn or(self, optb: Option<T>) -> Option<T> {
        match self {
            Some(opta) => Some(opta),
            _ => optb
        }
    }

    /// Update an optional value by optionally running its content by reference
    /// through a function that returns an option.
    #[inline]
    pub fn chain_ref<'a, U>(&'a self, f: &fn(x: &'a T) -> Option<U>)
                            -> Option<U> {
        match *self {
            Some(ref x) => f(x),
            None => None
        }
    }

    /// Filters an optional value using given function.
    #[inline(always)]
    pub fn filtered(self, f: &fn(t: &T) -> bool) -> Option<T> {
        match self {
            Some(x) => if(f(&x)) {Some(x)} else {None},
            None => None
        }
    }

    /// Maps a `some` value from one type to another by reference
    #[inline]
    pub fn map<'a, U>(&'a self, f: &fn(&'a T) -> U) -> Option<U> {
        match *self { Some(ref x) => Some(f(x)), None => None }
    }

    /// Maps a `some` value from one type to another by a mutable reference
    #[inline]
    pub fn map_mut<'a, U>(&'a mut self, f: &fn(&'a mut T) -> U) -> Option<U> {
        match *self { Some(ref mut x) => Some(f(x)), None => None }
    }

    /// Maps a `some` value from one type to another by a mutable reference,
    /// or returns a default value.
    #[inline]
    pub fn map_mut_default<'a, U>(&'a mut self, def: U, f: &fn(&'a mut T) -> U) -> U {
        match *self { Some(ref mut x) => f(x), None => def }
    }

    /// As `map`, but consumes the option and gives `f` ownership to avoid
    /// copying.
    #[inline]
    pub fn map_consume<U>(self, f: &fn(v: T) -> U) -> Option<U> {
        match self { None => None, Some(v) => Some(f(v)) }
    }

    /// Applies a function to the contained value or returns a default
    #[inline]
    pub fn map_default<'a, U>(&'a self, def: U, f: &fn(&'a T) -> U) -> U {
        match *self { None => def, Some(ref t) => f(t) }
    }

    /// As `map_default`, but consumes the option and gives `f`
    /// ownership to avoid copying.
    #[inline]
    pub fn map_consume_default<U>(self, def: U, f: &fn(v: T) -> U) -> U {
        match self { None => def, Some(v) => f(v) }
    }

    /// Take the value out of the option, leaving a `None` in its place.
    #[inline]
    pub fn take(&mut self) -> Option<T> {
        util::replace(self, None)
    }

    /// As `map_consume`, but swaps a None into the original option rather
    /// than consuming it by-value.
    #[inline]
    pub fn take_map<U>(&mut self, blk: &fn(T) -> U) -> Option<U> {
        self.take().map_consume(blk)
    }

    /// As `map_consume_default`, but swaps a None into the original option
    /// rather than consuming it by-value.
    #[inline]
    pub fn take_map_default<U> (&mut self, def: U, blk: &fn(T) -> U) -> U {
        self.take().map_consume_default(def, blk)
    }

    /// Apply a function to the contained value or do nothing
    pub fn mutate(&mut self, f: &fn(T) -> T) {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
        }
    }

    /// Apply a function to the contained value or set it to a default
    pub fn mutate_default(&mut self, def: T, f: &fn(T) -> T) {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
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
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a T {
        match *self {
          Some(ref x) => x,
          None => fail!("option::get_ref none")
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
    #[inline]
    pub fn get_mut_ref<'a>(&'a mut self) -> &'a mut T {
        match *self {
          Some(ref mut x) => x,
          None => fail!("option::get_mut_ref none")
        }
    }

    #[inline]
    pub fn unwrap(self) -> T {
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
          None => fail!("option::unwrap none")
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
    #[inline]
    pub fn take_unwrap(&mut self) -> T {
        if self.is_none() { fail!("option::take_unwrap none") }
        self.take().unwrap()
    }

    /**
     * Gets the value out of an option, printing a specified message on
     * failure
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    #[inline]
    pub fn expect(self, reason: &str) -> T {
        match self {
          Some(val) => val,
          None => fail!(reason.to_owned()),
        }
    }

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
    #[inline]
    pub fn get(self) -> T {
        match self {
          Some(x) => return x,
          None => fail!("option::get none")
        }
    }

    /// Returns the contained value or a default
    #[inline]
    pub fn get_or_default(self, def: T) -> T {
        match self { Some(x) => x, None => def }
    }

    /// Applies a function zero or more times until the result is none.
    #[inline]
    pub fn while_some(self, blk: &fn(v: T) -> Option<T>) {
        let mut opt = self;
        while opt.is_some() {
            opt = blk(opt.unwrap());
        }
    }
}

impl<T:Zero> Option<T> {
    /// Returns the contained value or zero (for this type)
    #[inline]
    pub fn get_or_zero(self) -> T {
        match self {
            Some(x) => x,
            None => Zero::zero()
        }
    }
}

impl<T> Zero for Option<T> {
    fn zero() -> Option<T> { None }
    fn is_zero(&self) -> bool { self.is_none() }
}

/// Immutable iterator over an `Option<A>`
pub struct OptionIterator<'self, A> {
    priv opt: Option<&'self A>
}

impl<'self, A> Iterator<&'self A> for OptionIterator<'self, A> {
    fn next(&mut self) -> Option<&'self A> {
        util::replace(&mut self.opt, None)
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        match self.opt {
            Some(_) => (1, Some(1)),
            None => (0, Some(0)),
        }
    }
}

/// Mutable iterator over an `Option<A>`
pub struct OptionMutIterator<'self, A> {
    priv opt: Option<&'self mut A>
}

impl<'self, A> Iterator<&'self mut A> for OptionMutIterator<'self, A> {
    fn next(&mut self) -> Option<&'self mut A> {
        util::replace(&mut self.opt, None)
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        match self.opt {
            Some(_) => (1, Some(1)),
            None => (0, Some(0)),
        }
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
        assert_eq!(addr_x, addr_y);
    }
}

#[test]
fn test_unwrap_str() {
    let x = ~"test";
    let addr_x = x.as_imm_buf(|buf, _len| buf);
    let opt = Some(x);
    let y = opt.unwrap();
    let addr_y = y.as_imm_buf(|buf, _len| buf);
    assert_eq!(addr_x, addr_y);
}

#[test]
fn test_unwrap_resource() {
    struct R {
       i: @mut int,
    }

    #[unsafe_destructor]
    impl ::ops::Drop for R {
       fn drop(&self) { *(self.i) += 1; }
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
    assert_eq!(*i, 1);
}

#[test]
fn test_option_dance() {
    let x = Some(());
    let mut y = Some(5);
    let mut y2 = 0;
    for x.iter().advance |_x| {
        y2 = y.take_unwrap();
    }
    assert_eq!(y2, 5);
    assert!(y.is_none());
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_option_too_much_dance() {
    let mut y = Some(util::NonCopyable);
    let _y2 = y.take_unwrap();
    let _y3 = y.take_unwrap();
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
    assert_eq!(i, 11);
}

#[test]
fn test_get_or_zero() {
    let some_stuff = Some(42);
    assert_eq!(some_stuff.get_or_zero(), 42);
    let no_stuff: Option<int> = None;
    assert_eq!(no_stuff.get_or_zero(), 0);
}

#[test]
fn test_filtered() {
    let some_stuff = Some(42);
    let modified_stuff = some_stuff.filtered(|&x| {x < 10});
    assert_eq!(some_stuff.get(), 42);
    assert!(modified_stuff.is_none());
}

#[test]
fn test_iter() {
    let val = 5;

    let x = Some(val);
    let mut it = x.iter();

    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next(), Some(&val));
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
}

#[test]
fn test_mut_iter() {
    let val = 5;
    let new_val = 11;

    let mut x = Some(val);
    let mut it = x.mut_iter();

    assert_eq!(it.size_hint(), (1, Some(1)));

    match it.next() {
        Some(interior) => {
            assert_eq!(*interior, val);
            *interior = new_val;
            assert_eq!(x, Some(new_val));
        }
        None => assert!(false),
    }

    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
}
