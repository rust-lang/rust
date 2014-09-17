// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Optional values
//!
//! Type `Option` represents an optional value: every `Option`
//! is either `Some` and contains a value, or `None`, and
//! does not. `Option` types are very common in Rust code, as
//! they have a number of uses:
//!
//! * Initial values
//! * Return values for functions that are not defined
//!   over their entire input range (partial functions)
//! * Return value for otherwise reporting simple errors, where `None` is
//!   returned on error
//! * Optional struct fields
//! * Struct fields that can be loaned or "taken"
//! * Optional function arguments
//! * Nullable pointers
//! * Swapping things out of difficult situations
//!
//! Options are commonly paired with pattern matching to query the presence
//! of a value and take action, always accounting for the `None` case.
//!
//! ```
//! fn divide(numerator: f64, denominator: f64) -> Option<f64> {
//!     if denominator == 0.0 {
//!         None
//!     } else {
//!         Some(numerator / denominator)
//!     }
//! }
//!
//! // The return value of the function is an option
//! let result = divide(2.0, 3.0);
//!
//! // Pattern match to retrieve the value
//! match result {
//!     // The division was valid
//!     Some(x) => println!("Result: {}", x),
//!     // The division was invalid
//!     None    => println!("Cannot divide by 0")
//! }
//! ```
//!
//
// FIXME: Show how `Option` is used in practice, with lots of methods
//
//! # Options and pointers ("nullable" pointers)
//!
//! Rust's pointer types must always point to a valid location; there are
//! no "null" pointers. Instead, Rust has *optional* pointers, like
//! the optional owned box, `Option<Box<T>>`.
//!
//! The following example uses `Option` to create an optional box of
//! `int`. Notice that in order to use the inner `int` value first the
//! `check_optional` function needs to use pattern matching to
//! determine whether the box has a value (i.e. it is `Some(...)`) or
//! not (`None`).
//!
//! ```
//! let optional: Option<Box<int>> = None;
//! check_optional(&optional);
//!
//! let optional: Option<Box<int>> = Some(box 9000);
//! check_optional(&optional);
//!
//! fn check_optional(optional: &Option<Box<int>>) {
//!     match *optional {
//!         Some(ref p) => println!("have value {}", p),
//!         None => println!("have no value")
//!     }
//! }
//! ```
//!
//! This usage of `Option` to create safe nullable pointers is so
//! common that Rust does special optimizations to make the
//! representation of `Option<Box<T>>` a single pointer. Optional pointers
//! in Rust are stored as efficiently as any other pointer type.
//!
//! # Examples
//!
//! Basic pattern matching on `Option`:
//!
//! ```
//! let msg = Some("howdy");
//!
//! // Take a reference to the contained string
//! match msg {
//!     Some(ref m) => println!("{}", *m),
//!     None => ()
//! }
//!
//! // Remove the contained string, destroying the Option
//! let unwrapped_msg = match msg {
//!     Some(m) => m,
//!     None => "default message"
//! };
//! ```
//!
//! Initialize a result to `None` before a loop:
//!
//! ```
//! enum Kingdom { Plant(uint, &'static str), Animal(uint, &'static str) }
//!
//! // A list of data to search through.
//! let all_the_big_things = [
//!     Plant(250, "redwood"),
//!     Plant(230, "noble fir"),
//!     Plant(229, "sugar pine"),
//!     Animal(25, "blue whale"),
//!     Animal(19, "fin whale"),
//!     Animal(15, "north pacific right whale"),
//! ];
//!
//! // We're going to search for the name of the biggest animal,
//! // but to start with we've just got `None`.
//! let mut name_of_biggest_animal = None;
//! let mut size_of_biggest_animal = 0;
//! for big_thing in all_the_big_things.iter() {
//!     match *big_thing {
//!         Animal(size, name) if size > size_of_biggest_animal => {
//!             // Now we've found the name of some big animal
//!             size_of_biggest_animal = size;
//!             name_of_biggest_animal = Some(name);
//!         }
//!         Animal(..) | Plant(..) => ()
//!     }
//! }
//!
//! match name_of_biggest_animal {
//!     Some(name) => println!("the biggest animal is {}", name),
//!     None => println!("there are no animals :(")
//! }
//! ```

#![stable]

use cmp::{PartialEq, Eq, Ord};
use default::Default;
use slice::Slice;
use iter::{Iterator, DoubleEndedIterator, FromIterator, ExactSize};
use mem;
use slice;

// Note that this is not a lang item per se, but it has a hidden dependency on
// `Iterator`, which is one. The compiler assumes that the `next` method of
// `Iterator` is an enumeration with one type parameter and two variants,
// which basically means it must be `Option`.

/// The `Option` type.
#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord, Show)]
#[stable]
pub enum Option<T> {
    /// No value
    None,
    /// Some value `T`
    Some(T)
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

impl<T> Option<T> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns `true` if the option is a `Some` value
    ///
    /// # Example
    ///
    /// ```
    /// let x: Option<uint> = Some(2);
    /// assert_eq!(x.is_some(), true);
    ///
    /// let x: Option<uint> = None;
    /// assert_eq!(x.is_some(), false);
    /// ```
    #[inline]
    #[stable]
    pub fn is_some(&self) -> bool {
        match *self {
            Some(_) => true,
            None => false
        }
    }

    /// Returns `true` if the option is a `None` value
    ///
    /// # Example
    ///
    /// ```
    /// let x: Option<uint> = Some(2);
    /// assert_eq!(x.is_none(), false);
    ///
    /// let x: Option<uint> = None;
    /// assert_eq!(x.is_none(), true);
    /// ```
    #[inline]
    #[stable]
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Convert from `Option<T>` to `Option<&T>`
    ///
    /// # Example
    ///
    /// Convert an `Option<String>` into an `Option<int>`, preserving the original.
    /// The `map` method takes the `self` argument by value, consuming the original,
    /// so this technique uses `as_ref` to first take an `Option` to a reference
    /// to the value inside the original.
    ///
    /// ```
    /// let num_as_str: Option<String> = Some("10".to_string());
    /// // First, cast `Option<String>` to `Option<&String>` with `as_ref`,
    /// // then consume *that* with `map`, leaving `num_as_str` on the stack.
    /// let num_as_int: Option<uint> = num_as_str.as_ref().map(|n| n.len());
    /// println!("still can print num_as_str: {}", num_as_str);
    /// ```
    #[inline]
    #[stable]
    pub fn as_ref<'r>(&'r self) -> Option<&'r T> {
        match *self { Some(ref x) => Some(x), None => None }
    }

    /// Convert from `Option<T>` to `Option<&mut T>`
    ///
    /// # Example
    ///
    /// ```
    /// let mut x = Some(2u);
    /// match x.as_mut() {
    ///     Some(&ref mut v) => *v = 42,
    ///     None => {},
    /// }
    /// assert_eq!(x, Some(42u));
    /// ```
    #[inline]
    #[unstable = "waiting for mut conventions"]
    pub fn as_mut<'r>(&'r mut self) -> Option<&'r mut T> {
        match *self { Some(ref mut x) => Some(x), None => None }
    }

    /// Convert from `Option<T>` to `&mut [T]` (without copying)
    ///
    /// # Example
    ///
    /// ```
    /// let mut x = Some("Diamonds");
    /// {
    ///     let v = x.as_mut_slice();
    ///     assert!(v == ["Diamonds"]);
    ///     v[0] = "Dirt";
    ///     assert!(v == ["Dirt"]);
    /// }
    /// assert_eq!(x, Some("Dirt"));
    /// ```
    #[inline]
    #[unstable = "waiting for mut conventions"]
    pub fn as_mut_slice<'r>(&'r mut self) -> &'r mut [T] {
        match *self {
            Some(ref mut x) => {
                let result: &mut [T] = slice::mut_ref_slice(x);
                result
            }
            None => {
                let result: &mut [T] = &mut [];
                result
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Getting to contained values
    /////////////////////////////////////////////////////////////////////////

    /// Unwraps an option, yielding the content of a `Some`
    ///
    /// # Failure
    ///
    /// Fails if the value is a `None` with a custom failure message provided by
    /// `msg`.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some("value");
    /// assert_eq!(x.expect("the world is ending"), "value");
    /// ```
    ///
    /// ```{.should_fail}
    /// let x: Option<&str> = None;
    /// x.expect("the world is ending"); // fails with `world is ending`
    /// ```
    #[inline]
    #[unstable = "waiting for conventions"]
    pub fn expect(self, msg: &str) -> T {
        match self {
            Some(val) => val,
            None => fail!(msg),
        }
    }

    /// Returns the inner `T` of a `Some(T)`.
    ///
    /// # Failure
    ///
    /// Fails if the self value equals `None`.
    ///
    /// # Safety note
    ///
    /// In general, because this function may fail, its use is discouraged.
    /// Instead, prefer to use pattern matching and handle the `None`
    /// case explicitly.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some("air");
    /// assert_eq!(x.unwrap(), "air");
    /// ```
    ///
    /// ```{.should_fail}
    /// let x: Option<&str> = None;
    /// assert_eq!(x.unwrap(), "air"); // fails
    /// ```
    #[inline]
    #[unstable = "waiting for conventions"]
    pub fn unwrap(self) -> T {
        match self {
            Some(val) => val,
            None => fail!("called `Option::unwrap()` on a `None` value"),
        }
    }

    /// Returns the contained value or a default.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(Some("car").unwrap_or("bike"), "car");
    /// assert_eq!(None.unwrap_or("bike"), "bike");
    /// ```
    #[inline]
    #[unstable = "waiting for conventions"]
    pub fn unwrap_or(self, def: T) -> T {
        match self {
            Some(x) => x,
            None => def
        }
    }

    /// Returns the contained value or computes it from a closure.
    ///
    /// # Example
    ///
    /// ```
    /// let k = 10u;
    /// assert_eq!(Some(4u).unwrap_or_else(|| 2 * k), 4u);
    /// assert_eq!(None.unwrap_or_else(|| 2 * k), 20u);
    /// ```
    #[inline]
    #[unstable = "waiting for conventions"]
    pub fn unwrap_or_else(self, f: || -> T) -> T {
        match self {
            Some(x) => x,
            None => f()
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Transforming contained values
    /////////////////////////////////////////////////////////////////////////

    /// Maps an `Option<T>` to `Option<U>` by applying a function to a contained value
    ///
    /// # Example
    ///
    /// Convert an `Option<String>` into an `Option<uint>`, consuming the original:
    ///
    /// ```
    /// let num_as_str: Option<String> = Some("10".to_string());
    /// // `Option::map` takes self *by value*, consuming `num_as_str`
    /// let num_as_int: Option<uint> = num_as_str.map(|n| n.len());
    /// ```
    #[inline]
    #[unstable = "waiting for unboxed closures"]
    pub fn map<U>(self, f: |T| -> U) -> Option<U> {
        match self { Some(x) => Some(f(x)), None => None }
    }

    /// Applies a function to the contained value or returns a default.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some("foo");
    /// assert_eq!(x.map_or(42u, |v| v.len()), 3u);
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.map_or(42u, |v| v.len()), 42u);
    /// ```
    #[inline]
    #[unstable = "waiting for unboxed closures"]
    pub fn map_or<U>(self, def: U, f: |T| -> U) -> U {
        match self { None => def, Some(t) => f(t) }
    }

    /// Applies a function to the contained value or computes a default.
    ///
    /// # Example
    ///
    /// ```
    /// let k = 21u;
    ///
    /// let x = Some("foo");
    /// assert_eq!(x.map_or_else(|| 2 * k, |v| v.len()), 3u);
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.map_or_else(|| 2 * k, |v| v.len()), 42u);
    /// ```
    #[inline]
    #[unstable = "waiting for unboxed closures"]
    pub fn map_or_else<U>(self, def: || -> U, f: |T| -> U) -> U {
        match self { None => def(), Some(t) => f(t) }
    }

    /// Deprecated.
    ///
    /// Applies a function to the contained value or does nothing.
    /// Returns true if the contained value was mutated.
    #[deprecated = "removed due to lack of use"]
    pub fn mutate(&mut self, f: |T| -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take().unwrap()));
            true
        } else { false }
    }

    /// Deprecated.
    ///
    /// Applies a function to the contained value or sets it to a default.
    /// Returns true if the contained value was mutated, or false if set to the default.
    #[deprecated = "removed due to lack of use"]
    pub fn mutate_or_set(&mut self, def: T, f: |T| -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take().unwrap()));
            true
        } else {
            *self = Some(def);
            false
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Iterator constructors
    /////////////////////////////////////////////////////////////////////////

    /// Returns an iterator over the possibly contained value.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some(4u);
    /// assert_eq!(x.iter().next(), Some(&4));
    ///
    /// let x: Option<uint> = None;
    /// assert_eq!(x.iter().next(), None);
    /// ```
    #[inline]
    #[unstable = "waiting for iterator conventions"]
    pub fn iter<'r>(&'r self) -> Item<&'r T> {
        Item{opt: self.as_ref()}
    }

    /// Deprecated: use `iter_mut`
    #[deprecated = "use iter_mut"]
    pub fn mut_iter<'r>(&'r mut self) -> Item<&'r mut T> {
        self.iter_mut()
    }

    /// Returns a mutable iterator over the possibly contained value.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x = Some(4u);
    /// match x.iter_mut().next() {
    ///     Some(&ref mut v) => *v = 42u,
    ///     None => {},
    /// }
    /// assert_eq!(x, Some(42));
    ///
    /// let mut x: Option<uint> = None;
    /// assert_eq!(x.iter_mut().next(), None);
    /// ```
    #[inline]
    #[unstable = "waiting for iterator conventions"]
    pub fn iter_mut<'r>(&'r mut self) -> Item<&'r mut T> {
        Item{opt: self.as_mut()}
    }

    /// Deprecated: use `into_iter`.
    #[deprecated = "use into_iter"]
    pub fn move_iter(self) -> Item<T> {
        self.into_iter()
    }

    /// Returns a consuming iterator over the possibly contained value.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some("string");
    /// let v: Vec<&str> = x.into_iter().collect();
    /// assert_eq!(v, vec!["string"]);
    ///
    /// let x = None;
    /// let v: Vec<&str> = x.into_iter().collect();
    /// assert_eq!(v, vec![]);
    /// ```
    #[inline]
    #[unstable = "waiting for iterator conventions"]
    pub fn into_iter(self) -> Item<T> {
        Item{opt: self}
    }

    /////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `None` if the option is `None`, otherwise returns `optb`.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some(2u);
    /// let y: Option<&str> = None;
    /// assert_eq!(x.and(y), None);
    ///
    /// let x: Option<uint> = None;
    /// let y = Some("foo");
    /// assert_eq!(x.and(y), None);
    ///
    /// let x = Some(2u);
    /// let y = Some("foo");
    /// assert_eq!(x.and(y), Some("foo"));
    ///
    /// let x: Option<uint> = None;
    /// let y: Option<&str> = None;
    /// assert_eq!(x.and(y), None);
    /// ```
    #[inline]
    #[stable]
    pub fn and<U>(self, optb: Option<U>) -> Option<U> {
        match self {
            Some(_) => optb,
            None => None,
        }
    }

    /// Returns `None` if the option is `None`, otherwise calls `f` with the
    /// wrapped value and returns the result.
    ///
    /// # Example
    ///
    /// ```
    /// fn sq(x: uint) -> Option<uint> { Some(x * x) }
    /// fn nope(_: uint) -> Option<uint> { None }
    ///
    /// assert_eq!(Some(2).and_then(sq).and_then(sq), Some(16));
    /// assert_eq!(Some(2).and_then(sq).and_then(nope), None);
    /// assert_eq!(Some(2).and_then(nope).and_then(sq), None);
    /// assert_eq!(None.and_then(sq).and_then(sq), None);
    /// ```
    #[inline]
    #[unstable = "waiting for unboxed closures"]
    pub fn and_then<U>(self, f: |T| -> Option<U>) -> Option<U> {
        match self {
            Some(x) => f(x),
            None => None,
        }
    }

    /// Returns the option if it contains a value, otherwise returns `optb`.
    ///
    /// # Example
    ///
    /// ```
    /// let x = Some(2u);
    /// let y = None;
    /// assert_eq!(x.or(y), Some(2u));
    ///
    /// let x = None;
    /// let y = Some(100u);
    /// assert_eq!(x.or(y), Some(100u));
    ///
    /// let x = Some(2u);
    /// let y = Some(100u);
    /// assert_eq!(x.or(y), Some(2u));
    ///
    /// let x: Option<uint> = None;
    /// let y = None;
    /// assert_eq!(x.or(y), None);
    /// ```
    #[inline]
    #[stable]
    pub fn or(self, optb: Option<T>) -> Option<T> {
        match self {
            Some(_) => self,
            None => optb
        }
    }

    /// Returns the option if it contains a value, otherwise calls `f` and
    /// returns the result.
    ///
    /// # Example
    ///
    /// ```
    /// fn nobody() -> Option<&'static str> { None }
    /// fn vikings() -> Option<&'static str> { Some("vikings") }
    ///
    /// assert_eq!(Some("barbarians").or_else(vikings), Some("barbarians"));
    /// assert_eq!(None.or_else(vikings), Some("vikings"));
    /// assert_eq!(None.or_else(nobody), None);
    /// ```
    #[inline]
    #[unstable = "waiting for unboxed closures"]
    pub fn or_else(self, f: || -> Option<T>) -> Option<T> {
        match self {
            Some(_) => self,
            None => f()
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Misc
    /////////////////////////////////////////////////////////////////////////

    /// Takes the value out of the option, leaving a `None` in its place.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x = Some(2u);
    /// x.take();
    /// assert_eq!(x, None);
    ///
    /// let mut x: Option<uint> = None;
    /// x.take();
    /// assert_eq!(x, None);
    /// ```
    #[inline]
    #[stable]
    pub fn take(&mut self) -> Option<T> {
        mem::replace(self, None)
    }

    /// Deprecated.
    ///
    /// Filters an optional value using a given function.
    #[inline(always)]
    #[deprecated = "removed due to lack of use"]
    pub fn filtered(self, f: |t: &T| -> bool) -> Option<T> {
        match self {
            Some(x) => if f(&x) { Some(x) } else { None },
            None => None
        }
    }

    /// Deprecated.
    ///
    /// Applies a function zero or more times until the result is `None`.
    #[inline]
    #[deprecated = "removed due to lack of use"]
    pub fn while_some(self, f: |v: T| -> Option<T>) {
        let mut opt = self;
        loop {
            match opt {
                Some(x) => opt = f(x),
                None => break
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Common special cases
    /////////////////////////////////////////////////////////////////////////

    /// Deprecated: use `take().unwrap()` instead.
    ///
    /// The option dance. Moves a value out of an option type and returns it,
    /// replacing the original with `None`.
    ///
    /// # Failure
    ///
    /// Fails if the value equals `None`.
    #[inline]
    #[deprecated = "use take().unwrap() instead"]
    pub fn take_unwrap(&mut self) -> T {
        match self.take() {
            Some(x) => x,
            None => fail!("called `Option::take_unwrap()` on a `None` value")
        }
    }

    /// Deprecated: use `as_ref().unwrap()` instead.
    ///
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
    #[deprecated = "use .as_ref().unwrap() instead"]
    pub fn get_ref<'a>(&'a self) -> &'a T {
        match *self {
            Some(ref x) => x,
            None => fail!("called `Option::get_ref()` on a `None` value"),
        }
    }

    /// Deprecated: use `as_mut().unwrap()` instead.
    ///
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
    #[deprecated = "use .as_mut().unwrap() instead"]
    pub fn get_mut_ref<'a>(&'a mut self) -> &'a mut T {
        match *self {
            Some(ref mut x) => x,
            None => fail!("called `Option::get_mut_ref()` on a `None` value"),
        }
    }
}

impl<T: Default> Option<T> {
    /// Returns the contained value or a default
    ///
    /// Consumes the `self` argument then, if `Some`, returns the contained
    /// value, otherwise if `None`, returns the default value for that
    /// type.
    ///
    /// # Example
    ///
    /// Convert a string to an integer, turning poorly-formed strings
    /// into 0 (the default value for integers). `from_str` converts
    /// a string to any other type that implements `FromStr`, returning
    /// `None` on error.
    ///
    /// ```
    /// let good_year_from_input = "1909";
    /// let bad_year_from_input = "190blarg";
    /// let good_year = from_str(good_year_from_input).unwrap_or_default();
    /// let bad_year = from_str(bad_year_from_input).unwrap_or_default();
    ///
    /// assert_eq!(1909i, good_year);
    /// assert_eq!(0i, bad_year);
    /// ```
    #[inline]
    #[unstable = "waiting for conventions"]
    pub fn unwrap_or_default(self) -> T {
        match self {
            Some(x) => x,
            None => Default::default()
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

impl<T> Slice<T> for Option<T> {
    /// Convert from `Option<T>` to `&[T]` (without copying)
    #[inline]
    #[stable]
    fn as_slice<'a>(&'a self) -> &'a [T] {
        match *self {
            Some(ref x) => slice::ref_slice(x),
            None => {
                let result: &[_] = &[];
                result
            }
        }
    }
}

impl<T> Default for Option<T> {
    #[inline]
    fn default() -> Option<T> { None }
}

/////////////////////////////////////////////////////////////////////////////
// The Option Iterator
/////////////////////////////////////////////////////////////////////////////

/// An `Option` iterator that yields either one or zero elements
///
/// The `Item` iterator is returned by the `iter`, `iter_mut` and `into_iter`
/// methods on `Option`.
#[deriving(Clone)]
#[unstable = "waiting for iterator conventions"]
pub struct Item<A> {
    opt: Option<A>
}

impl<A> Iterator<A> for Item<A> {
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

impl<A> DoubleEndedIterator<A> for Item<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.opt.take()
    }
}

impl<A> ExactSize<A> for Item<A> {}

/////////////////////////////////////////////////////////////////////////////
// Free functions
/////////////////////////////////////////////////////////////////////////////

/// Deprecated: use `Iterator::collect` instead.
#[inline]
#[deprecated = "use Iterator::collect instead"]
pub fn collect<T, Iter: Iterator<Option<T>>, V: FromIterator<T>>(mut iter: Iter) -> Option<V> {
    iter.collect()
}

impl<A, V: FromIterator<A>> FromIterator<Option<A>> for Option<V> {
    /// Takes each element in the `Iterator`: if it is `None`, no further
    /// elements are taken, and the `None` is returned. Should no `None` occur, a
    /// container with the values of each `Option` is returned.
    ///
    /// Here is an example which increments every integer in a vector,
    /// checking for overflow:
    ///
    /// ```rust
    /// use std::uint;
    ///
    /// let v = vec!(1u, 2u);
    /// let res: Option<Vec<uint>> = v.iter().map(|x: &uint|
    ///     if *x == uint::MAX { None }
    ///     else { Some(x + 1) }
    /// ).collect();
    /// assert!(res == Some(vec!(2u, 3u)));
    /// ```
    #[inline]
    fn from_iter<I: Iterator<Option<A>>>(iter: I) -> Option<V> {
        // FIXME(#11084): This could be replaced with Iterator::scan when this
        // performance bug is closed.

        struct Adapter<Iter> {
            iter: Iter,
            found_none: bool,
        }

        impl<T, Iter: Iterator<Option<T>>> Iterator<T> for Adapter<Iter> {
            #[inline]
            fn next(&mut self) -> Option<T> {
                match self.iter.next() {
                    Some(Some(value)) => Some(value),
                    Some(None) => {
                        self.found_none = true;
                        None
                    }
                    None => None,
                }
            }
        }

        let mut adapter = Adapter { iter: iter, found_none: false };
        let v: V = FromIterator::from_iter(adapter.by_ref());

        if adapter.found_none {
            None
        } else {
            Some(v)
        }
    }
}
