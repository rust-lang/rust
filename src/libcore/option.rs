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

use cmp::{PartialEq, Eq, Ord};
use default::Default;
use iter::{Iterator, DoubleEndedIterator, FromIterator, ExactSize};
use mem;
use slice;

/// The `Option`
#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord, Show)]
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
    #[inline]
    pub fn is_some(&self) -> bool {
        match *self {
            Some(_) => true,
            None => false
        }
    }

    /// Returns `true` if the option is a `None` value
    #[inline]
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
    pub fn as_ref<'r>(&'r self) -> Option<&'r T> {
        match *self { Some(ref x) => Some(x), None => None }
    }

    /// Convert from `Option<T>` to `Option<&mut T>`
    #[inline]
    pub fn as_mut<'r>(&'r mut self) -> Option<&'r mut T> {
        match *self { Some(ref mut x) => Some(x), None => None }
    }

    /// Convert from `Option<T>` to `&[T]` (without copying)
    #[inline]
    pub fn as_slice<'r>(&'r self) -> &'r [T] {
        match *self {
            Some(ref x) => slice::ref_slice(x),
            None => &[]
        }
    }

    /// Convert from `Option<T>` to `&mut [T]` (without copying)
    #[inline]
    pub fn as_mut_slice<'r>(&'r mut self) -> &'r mut [T] {
        match *self {
            Some(ref mut x) => slice::mut_ref_slice(x),
            None => &mut []
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
    #[inline]
    pub fn expect(self, msg: &str) -> T {
        match self {
            Some(val) => val,
            None => fail!(msg),
        }
    }

    /// Moves a value out of an option type and returns it, consuming the `Option`.
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
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            Some(val) => val,
            None => fail!("called `Option::unwrap()` on a `None` value"),
        }
    }

    /// Returns the contained value or a default.
    #[inline]
    pub fn unwrap_or(self, def: T) -> T {
        match self {
            Some(x) => x,
            None => def
        }
    }

    /// Returns the contained value or computes it from a closure.
    #[inline]
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
    pub fn map<U>(self, f: |T| -> U) -> Option<U> {
        match self { Some(x) => Some(f(x)), None => None }
    }

    /// Applies a function to the contained value or returns a default.
    #[inline]
    pub fn map_or<U>(self, def: U, f: |T| -> U) -> U {
        match self { None => def, Some(t) => f(t) }
    }

    /// Applies a function to the contained value or does nothing.
    /// Returns true if the contained value was mutated.
    pub fn mutate(&mut self, f: |T| -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
            true
        } else { false }
    }

    /// Applies a function to the contained value or sets it to a default.
    /// Returns true if the contained value was mutated, or false if set to the default.
    pub fn mutate_or_set(&mut self, def: T, f: |T| -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
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
    #[inline]
    pub fn iter<'r>(&'r self) -> Item<&'r T> {
        Item{opt: self.as_ref()}
    }

    /// Returns a mutable iterator over the possibly contained value.
    #[inline]
    pub fn mut_iter<'r>(&'r mut self) -> Item<&'r mut T> {
        Item{opt: self.as_mut()}
    }

    /// Returns a consuming iterator over the possibly contained value.
    #[inline]
    pub fn move_iter(self) -> Item<T> {
        Item{opt: self}
    }

    /////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `None` if the option is `None`, otherwise returns `optb`.
    #[inline]
    pub fn and<U>(self, optb: Option<U>) -> Option<U> {
        match self {
            Some(_) => optb,
            None => None,
        }
    }

    /// Returns `None` if the option is `None`, otherwise calls `f` with the
    /// wrapped value and returns the result.
    #[inline]
    pub fn and_then<U>(self, f: |T| -> Option<U>) -> Option<U> {
        match self {
            Some(x) => f(x),
            None => None,
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
    #[inline]
    pub fn take(&mut self) -> Option<T> {
        mem::replace(self, None)
    }

    /// Filters an optional value using a given function.
    #[inline(always)]
    pub fn filtered(self, f: |t: &T| -> bool) -> Option<T> {
        match self {
            Some(x) => if f(&x) { Some(x) } else { None },
            None => None
        }
    }

    /// Applies a function zero or more times until the result is `None`.
    #[inline]
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

    /// The option dance. Moves a value out of an option type and returns it,
    /// replacing the original with `None`.
    ///
    /// # Failure
    ///
    /// Fails if the value equals `None`.
    #[inline]
    pub fn take_unwrap(&mut self) -> T {
        match self.take() {
            Some(x) => x,
            None => fail!("called `Option::take_unwrap()` on a `None` value")
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
            None => fail!("called `Option::get_ref()` on a `None` value"),
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

impl<T> Default for Option<T> {
    #[inline]
    fn default() -> Option<T> { None }
}

/////////////////////////////////////////////////////////////////////////////
// The Option Iterator
/////////////////////////////////////////////////////////////////////////////

/// An `Option` iterator that yields either one or zero elements
///
/// The `Item` iterator is returned by the `iter`, `mut_iter` and `move_iter`
/// methods on `Option`.
#[deriving(Clone)]
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

/// Takes each element in the `Iterator`: if it is `None`, no further
/// elements are taken, and the `None` is returned. Should no `None` occur, a
/// vector containing the values of each `Option` is returned.
///
/// Here is an example which increments every integer in a vector,
/// checking for overflow:
///
/// ```rust
/// use std::option;
/// use std::uint;
///
/// let v = vec!(1u, 2u);
/// let res: Option<Vec<uint>> = option::collect(v.iter().map(|x: &uint|
///     if *x == uint::MAX { None }
///     else { Some(x + 1) }
/// ));
/// assert!(res == Some(vec!(2u, 3u)));
/// ```
#[inline]
pub fn collect<T, Iter: Iterator<Option<T>>, V: FromIterator<T>>(iter: Iter) -> Option<V> {
    // FIXME(#11084): This should be twice as fast once this bug is closed.
    let mut iter = iter.scan(false, |state, x| {
        match x {
            Some(x) => Some(x),
            None => {
                *state = true;
                None
            }
        }
    });

    let v: V = FromIterator::from_iter(iter.by_ref());

    if iter.state {
        None
    } else {
        Some(v)
    }
}

/////////////////////////////////////////////////////////////////////////////
// Tests
/////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use realstd::vec::Vec;
    use realstd::string::String;
    use option::collect;
    use prelude::*;
    use realstd::str::{Str, StrAllocating};
    use iter::range;

    use str::StrSlice;
    use kinds::marker;
    use slice::ImmutableVector;

    #[test]
    fn test_get_ptr() {
        unsafe {
            let x = box 0i;
            let addr_x: *const int = ::mem::transmute(&*x);
            let opt = Some(x);
            let y = opt.unwrap();
            let addr_y: *const int = ::mem::transmute(&*y);
            assert_eq!(addr_x, addr_y);
        }
    }

    #[test]
    fn test_get_str() {
        let x = "test".to_string();
        let addr_x = x.as_slice().as_ptr();
        let opt = Some(x);
        let y = opt.unwrap();
        let addr_y = y.as_slice().as_ptr();
        assert_eq!(addr_x, addr_y);
    }

    #[test]
    fn test_get_resource() {
        use realstd::rc::Rc;
        use cell::RefCell;

        struct R {
           i: Rc<RefCell<int>>,
        }

        #[unsafe_destructor]
        impl ::ops::Drop for R {
           fn drop(&mut self) {
                let ii = &*self.i;
                let i = *ii.borrow();
                *ii.borrow_mut() = i + 1;
            }
        }

        fn r(i: Rc<RefCell<int>>) -> R {
            R {
                i: i
            }
        }

        fn realclone<T: ::realstd::clone::Clone>(t: &T) -> T {
            use realstd::clone::Clone;
            t.clone()
        }

        let i = Rc::new(RefCell::new(0i));
        {
            let x = r(realclone(&i));
            let opt = Some(x);
            let _y = opt.unwrap();
        }
        assert_eq!(*i.borrow(), 1);
    }

    #[test]
    fn test_option_dance() {
        let x = Some(());
        let mut y = Some(5i);
        let mut y2 = 0;
        for _x in x.iter() {
            y2 = y.take_unwrap();
        }
        assert_eq!(y2, 5);
        assert!(y.is_none());
    }

    #[test] #[should_fail]
    fn test_option_too_much_dance() {
        let mut y = Some(marker::NoCopy);
        let _y2 = y.take_unwrap();
        let _y3 = y.take_unwrap();
    }

    #[test]
    fn test_and() {
        let x: Option<int> = Some(1i);
        assert_eq!(x.and(Some(2i)), Some(2));
        assert_eq!(x.and(None::<int>), None);

        let x: Option<int> = None;
        assert_eq!(x.and(Some(2i)), None);
        assert_eq!(x.and(None::<int>), None);
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
        let mut i = 0i;
        Some(10i).while_some(|j| {
            i += 1;
            if j > 0 {
                Some(j-1)
            } else {
                None
            }
        });
        assert_eq!(i, 11);
    }

    #[test]
    fn test_unwrap() {
        assert_eq!(Some(1i).unwrap(), 1);
        let s = Some("hello".to_string()).unwrap();
        assert_eq!(s.as_slice(), "hello");
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
        let x: Option<String> = None;
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
    fn test_filtered() {
        let some_stuff = Some(42i);
        let modified_stuff = some_stuff.filtered(|&x| {x < 10});
        assert_eq!(some_stuff.unwrap(), 42);
        assert!(modified_stuff.is_none());
    }

    #[test]
    fn test_iter() {
        let val = 5i;

        let x = Some(val);
        let mut it = x.iter();

        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next(), Some(&val));
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_mut_iter() {
        let val = 5i;
        let new_val = 11i;

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
        let small = Some(1.0f64);
        let big = Some(5.0f64);
        let nan = Some(0.0f64/0.0);
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
        assert!(x.mutate_or_set(0, |i| i+1));
        assert_eq!(x, Some(5i));
        x = None;
        assert!(!x.mutate(|i| i+1));
        assert_eq!(x, None);
        assert!(!x.mutate_or_set(0i, |i| i+1));
        assert_eq!(x, Some(0i));
    }

    #[test]
    fn test_collect() {
        let v: Option<Vec<int>> = collect(range(0i, 0)
                                          .map(|_| Some(0i)));
        assert!(v == Some(vec![]));

        let v: Option<Vec<int>> = collect(range(0i, 3)
                                          .map(|x| Some(x)));
        assert!(v == Some(vec![0, 1, 2]));

        let v: Option<Vec<int>> = collect(range(0i, 3)
                                          .map(|x| if x > 1 { None } else { Some(x) }));
        assert!(v == None);

        // test that it does not take more elements than it needs
        let mut functions = [|| Some(()), || None, || fail!()];

        let v: Option<Vec<()>> = collect(functions.mut_iter().map(|f| (*f)()));

        assert!(v == None);
    }
}
