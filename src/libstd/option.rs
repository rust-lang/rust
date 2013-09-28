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

 ```
let msg = Some(~"howdy");

// Take a reference to the contained string
match msg {
    Some(ref m) => io::println(*m),
    None => ()
}

// Remove the contained string, destroying the Option
let unwrapped_msg = match msg {
    Some(m) => m,
    None => ~"default message"
};
 ```

*/

use clone::Clone;
use cmp::{Eq,Ord};
use default::Default;
use either;
use util;
use num::Zero;
use iter;
use iter::{Iterator, DoubleEndedIterator, ExactSize};
use result;
use str::{StrSlice, OwnedStr};
use to_str::ToStr;
use clone::DeepClone;

/// The option type
#[deriving(Clone, DeepClone, Eq)]
pub enum Option<T> {
    None,
    Some(T),
}

impl<T: Eq + Ord> Ord for Option<T> {
    fn lt(&self, other: &Option<T>) -> bool {
        iter::order::lt(self.iter(), other.iter())
    }

    fn le(&self, other: &Option<T>) -> bool {
        iter::order::le(self.iter(), other.iter())
    }

    fn ge(&self, other: &Option<T>) -> bool {
        iter::order::ge(self.iter(), other.iter())
    }

    fn gt(&self, other: &Option<T>) -> bool {
        iter::order::gt(self.iter(), other.iter())
    }
}

// FIXME: #8242 implementing manually because deriving doesn't work for some reason
impl<T: ToStr> ToStr for Option<T> {
    fn to_str(&self) -> ~str {
        match *self {
            Some(ref x) => {
                let mut s = ~"Some(";
                s.push_str(x.to_str());
                s.push_str(")");
                s
            }
            None => ~"None"
        }
    }
}

impl<T> Option<T> {
    /// Return an iterator over the possibly contained value
    #[inline]
    pub fn iter<'r>(&'r self) -> OptionIterator<&'r T> {
        match *self {
            Some(ref x) => OptionIterator{opt: Some(x)},
            None => OptionIterator{opt: None}
        }
    }

    /// Return a mutable iterator over the possibly contained value
    #[inline]
    pub fn mut_iter<'r>(&'r mut self) -> OptionIterator<&'r mut T> {
        match *self {
            Some(ref mut x) => OptionIterator{opt: Some(x)},
            None => OptionIterator{opt: None}
        }
    }

    /// Return a consuming iterator over the possibly contained value
    #[inline]
    pub fn move_iter(self) -> OptionIterator<T> {
        OptionIterator{opt: self}
    }

    /// Returns true if the option equals `None`
    #[inline]
    pub fn is_none(&self) -> bool {
        match *self { None => true, Some(_) => false }
    }

    /// Returns true if the option contains a `Some` value
    #[inline]
    pub fn is_some(&self) -> bool { !self.is_none() }

    /// Returns `None` if the option is `None`, otherwise returns `optb`.
    #[inline]
    pub fn and(self, optb: Option<T>) -> Option<T> {
        match self {
            Some(_) => optb,
            None => None,
        }
    }

    /// Returns `None` if the option is `None`, otherwise calls `f` with the
    /// wrapped value and returns the result.
    #[inline]
    pub fn and_then<U>(self, f: &fn(T) -> Option<U>) -> Option<U> {
        match self {
            Some(x) => f(x),
            None => None,
        }
    }

    /// Returns `None` if the option is `None`, otherwise calls `f` with a
    /// reference to the wrapped value and returns the result.
    #[inline]
    pub fn and_then_ref<'a, U>(&'a self, f: &fn(&'a T) -> Option<U>) -> Option<U> {
        match *self {
            Some(ref x) => f(x),
            None => None
        }
    }

    /// Returns `None` if the option is `None`, otherwise calls `f` with a
    /// mutable reference to the wrapped value and returns the result.
    #[inline]
    pub fn and_then_mut_ref<'a, U>(&'a mut self, f: &fn(&'a mut T) -> Option<U>) -> Option<U> {
        match *self {
            Some(ref mut x) => f(x),
            None => None
        }
    }

    /// Returns the option if it contains a value, otherwise returns `optb`.
    #[inline]
    pub fn or(self, optb: Option<T>) -> Option<T> {
        match self {
            Some(_) => self,
            None => optb
        }
    }

    /// Returns the option if it contains a value, otherwise calls `f` and
    /// returns the result.
    #[inline]
    pub fn or_else(self, f: &fn() -> Option<T>) -> Option<T> {
        match self {
            Some(_) => self,
            None => f(),
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

    /// Maps a `Some` value from one type to another by reference
    #[inline]
    pub fn map<'a, U>(&'a self, f: &fn(&'a T) -> U) -> Option<U> {
        match *self { Some(ref x) => Some(f(x)), None => None }
    }

    /// Maps a `Some` value from one type to another by a mutable reference
    #[inline]
    pub fn map_mut<'a, U>(&'a mut self, f: &fn(&'a mut T) -> U) -> Option<U> {
        match *self { Some(ref mut x) => Some(f(x)), None => None }
    }

    /// Applies a function to the contained value or returns a default
    #[inline]
    pub fn map_default<'a, U>(&'a self, def: U, f: &fn(&'a T) -> U) -> U {
        match *self { None => def, Some(ref t) => f(t) }
    }

    /// Maps a `Some` value from one type to another by a mutable reference,
    /// or returns a default value.
    #[inline]
    pub fn map_mut_default<'a, U>(&'a mut self, def: U, f: &fn(&'a mut T) -> U) -> U {
        match *self { Some(ref mut x) => f(x), None => def }
    }

    /// As `map`, but consumes the option and gives `f` ownership to avoid
    /// copying.
    #[inline]
    pub fn map_move<U>(self, f: &fn(T) -> U) -> Option<U> {
        match self { Some(x) => Some(f(x)), None => None }
    }

    /// As `map_default`, but consumes the option and gives `f`
    /// ownership to avoid copying.
    #[inline]
    pub fn map_move_default<U>(self, def: U, f: &fn(T) -> U) -> U {
        match self { None => def, Some(t) => f(t) }
    }

    /// Take the value out of the option, leaving a `None` in its place.
    #[inline]
    pub fn take(&mut self) -> Option<T> {
        util::replace(self, None)
    }

    /// Apply a function to the contained value or do nothing.
    /// Returns true if the contained value was mutated.
    pub fn mutate(&mut self, f: &fn(T) -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
            true
        } else { false }
    }

    /// Apply a function to the contained value or set it to a default.
    /// Returns true if the contained value was mutated, or false if set to the default.
    pub fn mutate_default(&mut self, def: T, f: &fn(T) -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
            true
        } else {
            *self = Some(def);
            false
        }
    }

    /// Gets an immutable reference to the value inside an option.
    ///
    /// # Failure
    ///
    /// Fails if the value equals `None`
    ///
    /// # Safety note
    ///
    /// In general, because this function may fail, its use is discouraged
    /// (calling `get` on `None` is akin to dereferencing a null pointer).
    /// Instead, prefer to use pattern matching and handle the `None`
    /// case explicitly.
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a T {
        match *self {
            Some(ref x) => x,
            None => fail2!("called `Option::get_ref()` on a `None` value"),
        }
    }

    /// Gets a mutable reference to the value inside an option.
    ///
    /// # Failure
    ///
    /// Fails if the value equals `None`
    ///
    /// # Safety note
    ///
    /// In general, because this function may fail, its use is discouraged
    /// (calling `get` on `None` is akin to dereferencing a null pointer).
    /// Instead, prefer to use pattern matching and handle the `None`
    /// case explicitly.
    #[inline]
    pub fn get_mut_ref<'a>(&'a mut self) -> &'a mut T {
        match *self {
            Some(ref mut x) => x,
            None => fail2!("called `Option::get_mut_ref()` on a `None` value"),
        }
    }

    /// Moves a value out of an option type and returns it.
    ///
    /// Useful primarily for getting strings, vectors and unique pointers out
    /// of option types without copying them.
    ///
    /// # Failure
    ///
    /// Fails if the value equals `None`.
    ///
    /// # Safety note
    ///
    /// In general, because this function may fail, its use is discouraged.
    /// Instead, prefer to use pattern matching and handle the `None`
    /// case explicitly.
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            Some(x) => x,
            None => fail2!("called `Option::unwrap()` on a `None` value"),
        }
    }

    /// The option dance. Moves a value out of an option type and returns it,
    /// replacing the original with `None`.
    ///
    /// # Failure
    ///
    /// Fails if the value equals `None`.
    #[inline]
    pub fn take_unwrap(&mut self) -> T {
        if self.is_none() {
            fail2!("called `Option::take_unwrap()` on a `None` value")
        }
        self.take().unwrap()
    }

    ///  Gets the value out of an option, printing a specified message on
    ///  failure
    ///
    ///  # Failure
    ///
    ///  Fails if the value equals `None`
    #[inline]
    pub fn expect(self, reason: &str) -> T {
        match self {
            Some(val) => val,
            None => fail2!("{}", reason.to_owned()),
        }
    }

    /// Returns the contained value or a default
    #[inline]
    pub fn unwrap_or(self, def: T) -> T {
        match self {
            Some(x) => x,
            None => def
        }
    }

    /// Returns the contained value or computes it from a closure
    #[inline]
    pub fn unwrap_or_else(self, f: &fn() -> T) -> T {
        match self {
            Some(x) => x,
            None => f()
        }
    }

    /// Applies a function zero or more times until the result is `None`.
    #[inline]
    pub fn while_some(self, blk: &fn(v: T) -> Option<T>) {
        let mut opt = self;
        while opt.is_some() {
            opt = blk(opt.unwrap());
        }
    }
}

/// A generic trait for converting a value to a `Option`
pub trait ToOption<T> {
    /// Convert to the `option` type
    fn to_option(&self) -> Option<T>;
}

/// A generic trait for converting a value to a `Option`
pub trait IntoOption<T> {
    /// Convert to the `option` type
    fn into_option(self) -> Option<T>;
}

/// A generic trait for converting a value to a `Option`
pub trait AsOption<T> {
    /// Convert to the `option` type
    fn as_option<'a>(&'a self) -> Option<&'a T>;
}

impl<T: Clone> ToOption<T> for Option<T> {
    #[inline]
    fn to_option(&self) -> Option<T> { self.clone() }
}

impl<T> IntoOption<T> for Option<T> {
    #[inline]
    fn into_option(self) -> Option<T> { self }
}

impl<T> AsOption<T> for Option<T> {
    #[inline]
    fn as_option<'a>(&'a self) -> Option<&'a T> {
        match *self {
            Some(ref x) => Some(x),
            None => None,
        }
    }
}

impl<T: Clone> result::ToResult<T, ()> for Option<T> {
    #[inline]
    fn to_result(&self) -> result::Result<T, ()> {
        match *self {
            Some(ref x) => result::Ok(x.clone()),
            None => result::Err(()),
        }
    }
}

impl<T> result::IntoResult<T, ()> for Option<T> {
    #[inline]
    fn into_result(self) -> result::Result<T, ()> {
        match self {
            Some(x) => result::Ok(x),
            None => result::Err(()),
        }
    }
}

impl<T: Clone> either::ToEither<(), T> for Option<T> {
    #[inline]
    fn to_either(&self) -> either::Either<(), T> {
        match *self {
            Some(ref x) => either::Right(x.clone()),
            None => either::Left(()),
        }
    }
}

impl<T> either::IntoEither<(), T> for Option<T> {
    #[inline]
    fn into_either(self) -> either::Either<(), T> {
        match self {
            Some(x) => either::Right(x),
            None => either::Left(()),
        }
    }
}

impl<T: Default> Option<T> {
    /// Returns the contained value or default (for this type)
    #[inline]
    pub fn unwrap_or_default(self) -> T {
        match self {
            Some(x) => x,
            None => Default::default()
        }
    }
}

impl<T> Default for Option<T> {
    #[inline]
    fn default() -> Option<T> { None }
}

impl<T: Zero> Option<T> {
    /// Returns the contained value or zero (for this type)
    #[inline]
    pub fn unwrap_or_zero(self) -> T {
        match self {
            Some(x) => x,
            None => Zero::zero()
        }
    }
}

/// An iterator that yields either one or zero elements
#[deriving(Clone, DeepClone)]
pub struct OptionIterator<A> {
    priv opt: Option<A>
}

impl<A> Iterator<A> for OptionIterator<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        self.opt.take()
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        match self.opt {
            Some(_) => (1, Some(1)),
            None => (0, Some(0)),
        }
    }
}

impl<A> DoubleEndedIterator<A> for OptionIterator<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.opt.take()
    }
}

impl<A> ExactSize<A> for OptionIterator<A> {}

#[cfg(test)]
mod tests {
    use super::*;

    use either::{IntoEither, ToEither};
    use either;
    use result::{IntoResult, ToResult};
    use result;
    use util;

    #[test]
    fn test_get_ptr() {
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
    fn test_get_str() {
        let x = ~"test";
        let addr_x = x.as_imm_buf(|buf, _len| buf);
        let opt = Some(x);
        let y = opt.unwrap();
        let addr_y = y.as_imm_buf(|buf, _len| buf);
        assert_eq!(addr_x, addr_y);
    }

    #[test]
    fn test_get_resource() {
        struct R {
           i: @mut int,
        }

        #[unsafe_destructor]
        impl ::ops::Drop for R {
           fn drop(&mut self) { *(self.i) += 1; }
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
        for _x in x.iter() {
            y2 = y.take_unwrap();
        }
        assert_eq!(y2, 5);
        assert!(y.is_none());
    }
    #[test] #[should_fail]
    fn test_option_too_much_dance() {
        let mut y = Some(util::NonCopyable);
        let _y2 = y.take_unwrap();
        let _y3 = y.take_unwrap();
    }

    #[test]
    fn test_and() {
        let x: Option<int> = Some(1);
        assert_eq!(x.and(Some(2)), Some(2));
        assert_eq!(x.and(None), None);

        let x: Option<int> = None;
        assert_eq!(x.and(Some(2)), None);
        assert_eq!(x.and(None), None);
    }

    #[test]
    fn test_and_then() {
        let x: Option<int> = Some(1);
        assert_eq!(x.and_then(|x| Some(x + 1)), Some(2));
        assert_eq!(x.and_then(|_| None::<int>), None);

        let x: Option<int> = None;
        assert_eq!(x.and_then(|x| Some(x + 1)), None);
        assert_eq!(x.and_then(|_| None::<int>), None);
    }

    #[test]
    fn test_or() {
        let x: Option<int> = Some(1);
        assert_eq!(x.or(Some(2)), Some(1));
        assert_eq!(x.or(None), Some(1));

        let x: Option<int> = None;
        assert_eq!(x.or(Some(2)), Some(2));
        assert_eq!(x.or(None), None);
    }

    #[test]
    fn test_or_else() {
        let x: Option<int> = Some(1);
        assert_eq!(x.or_else(|| Some(2)), Some(1));
        assert_eq!(x.or_else(|| None), Some(1));

        let x: Option<int> = None;
        assert_eq!(x.or_else(|| Some(2)), Some(2));
        assert_eq!(x.or_else(|| None), None);
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
    fn test_unwrap() {
        assert_eq!(Some(1).unwrap(), 1);
        assert_eq!(Some(~"hello").unwrap(), ~"hello");
    }

    #[test]
    #[should_fail]
    fn test_unwrap_fail1() {
        let x: Option<int> = None;
        x.unwrap();
    }

    #[test]
    #[should_fail]
    fn test_unwrap_fail2() {
        let x: Option<~str> = None;
        x.unwrap();
    }

    #[test]
    fn test_unwrap_or() {
        let x: Option<int> = Some(1);
        assert_eq!(x.unwrap_or(2), 1);

        let x: Option<int> = None;
        assert_eq!(x.unwrap_or(2), 2);
    }

    #[test]
    fn test_unwrap_or_else() {
        let x: Option<int> = Some(1);
        assert_eq!(x.unwrap_or_else(|| 2), 1);

        let x: Option<int> = None;
        assert_eq!(x.unwrap_or_else(|| 2), 2);
    }

    #[test]
    fn test_unwrap_or_zero() {
        let some_stuff = Some(42);
        assert_eq!(some_stuff.unwrap_or_zero(), 42);
        let no_stuff: Option<int> = None;
        assert_eq!(no_stuff.unwrap_or_zero(), 0);
    }

    #[test]
    fn test_filtered() {
        let some_stuff = Some(42);
        let modified_stuff = some_stuff.filtered(|&x| {x < 10});
        assert_eq!(some_stuff.unwrap(), 42);
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
        {
            let mut it = x.mut_iter();

            assert_eq!(it.size_hint(), (1, Some(1)));

            match it.next() {
                Some(interior) => {
                    assert_eq!(*interior, val);
                    *interior = new_val;
                }
                None => assert!(false),
            }

            assert_eq!(it.size_hint(), (0, Some(0)));
            assert!(it.next().is_none());
        }
        assert_eq!(x, Some(new_val));
    }

    #[test]
    fn test_ord() {
        let small = Some(1.0);
        let big = Some(5.0);
        let nan = Some(0.0/0.0);
        assert!(!(nan < big));
        assert!(!(nan > big));
        assert!(small < big);
        assert!(None < big);
        assert!(big > None);
    }

    #[test]
    fn test_mutate() {
        let mut x = Some(3i);
        assert!(x.mutate(|i| i+1));
        assert_eq!(x, Some(4i));
        assert!(x.mutate_default(0, |i| i+1));
        assert_eq!(x, Some(5i));
        x = None;
        assert!(!x.mutate(|i| i+1));
        assert_eq!(x, None);
        assert!(!x.mutate_default(0i, |i| i+1));
        assert_eq!(x, Some(0i));
    }

    #[test]
    pub fn test_to_option() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.to_option(), Some(100));
        assert_eq!(none.to_option(), None);
    }

    #[test]
    pub fn test_into_option() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.into_option(), Some(100));
        assert_eq!(none.into_option(), None);
    }

    #[test]
    pub fn test_as_option() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.as_option().unwrap(), &100);
        assert_eq!(none.as_option(), None);
    }

    #[test]
    pub fn test_to_result() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.to_result(), result::Ok(100));
        assert_eq!(none.to_result(), result::Err(()));
    }

    #[test]
    pub fn test_into_result() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.into_result(), result::Ok(100));
        assert_eq!(none.into_result(), result::Err(()));
    }

    #[test]
    pub fn test_to_either() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.to_either(), either::Right(100));
        assert_eq!(none.to_either(), either::Left(()));
    }

    #[test]
    pub fn test_into_either() {
        let some: Option<int> = Some(100);
        let none: Option<int> = None;

        assert_eq!(some.into_either(), either::Right(100));
        assert_eq!(none.into_either(), either::Left(()));
    }
}
