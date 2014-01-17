// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Optionally nullable values (`Option` type)
//!
//! Type `Option` represents an optional value.
//!
//! Every `Option<T>` value can either be `Some(T)` or `None`. Where in other
//! languages you might use a nullable type, in Rust you would use an option
//! type.
//!
//! Options are most commonly used with pattern matching to query the presence
//! of a value and take action, always accounting for the `None` case.
//!
//! # Example
//!
//! ```
//! let msg = Some(~"howdy");
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
//!     None => ~"default message"
//! };
//! ```

use any::Any;
use clone::Clone;
use clone::DeepClone;
use cmp::{Eq, TotalEq, TotalOrd};
use default::Default;
use fmt;
use iter::{Iterator, DoubleEndedIterator, FromIterator, ExactSize};
use kinds::Send;
use str::OwnedStr;
use to_str::ToStr;
use util;
use vec;

/// The option type
#[deriving(Clone, DeepClone, Eq, Ord, TotalEq, TotalOrd, ToStr)]
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

    /// Returns true if the option contains a `Some` value
    #[inline]
    pub fn is_some(&self) -> bool {
        match *self {
            Some(_) => true,
            None => false
        }
    }

    /// Returns true if the option equals `None`
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Convert from `Option<T>` to `Option<&T>`
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
            Some(ref x) => vec::ref_slice(x),
            None => &[]
        }
    }

    /// Convert from `Option<T>` to `&[T]` (without copying)
    #[inline]
    pub fn as_mut_slice<'r>(&'r mut self) -> &'r mut [T] {
        match *self {
            Some(ref mut x) => vec::mut_ref_slice(x),
            None => &mut []
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Getting to contained values
    /////////////////////////////////////////////////////////////////////////

    /// Unwraps a option, yielding the content of a `Some`
    /// Fails if the value is a `None` with a custom failure message provided by `msg`.
    #[inline]
    pub fn expect<M: Any + Send>(self, msg: M) -> T {
        match self {
            Some(val) => val,
            None => fail!(msg),
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
            Some(val) => val,
            None => fail!("called `Option::unwrap()` on a `None` value"),
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
    pub fn unwrap_or_else(self, f: || -> T) -> T {
        match self {
            Some(x) => x,
            None => f()
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Transforming contained values
    /////////////////////////////////////////////////////////////////////////

    /// Maps an `Option<T>` to `Option<U>` by applying a function to a contained value.
    #[inline]
    pub fn map<U>(self, f: |T| -> U) -> Option<U> {
        match self { Some(x) => Some(f(x)), None => None }
    }

    /// Applies a function to the contained value or returns a default.
    #[inline]
    pub fn map_or<U>(self, def: U, f: |T| -> U) -> U {
        match self { None => def, Some(t) => f(t) }
    }

    /// Apply a function to the contained value or do nothing.
    /// Returns true if the contained value was mutated.
    pub fn mutate(&mut self, f: |T| -> T) -> bool {
        if self.is_some() {
            *self = Some(f(self.take_unwrap()));
            true
        } else { false }
    }

    /// Apply a function to the contained value or set it to a default.
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
            None => f(),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Misc
    /////////////////////////////////////////////////////////////////////////

    /// Take the value out of the option, leaving a `None` in its place.
    #[inline]
    pub fn take(&mut self) -> Option<T> {
        util::replace(self, None)
    }

    /// Filters an optional value using a given function.
    #[inline(always)]
    pub fn filtered(self, f: |t: &T| -> bool) -> Option<T> {
        match self {
            Some(x) => if(f(&x)) {Some(x)} else {None},
            None => None
        }
    }

    /// Applies a function zero or more times until the result is `None`.
    #[inline]
    pub fn while_some(self, blk: |v: T| -> Option<T>) {
        let mut opt = self;
        while opt.is_some() {
            opt = blk(opt.unwrap());
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
        if self.is_none() {
            fail!("called `Option::take_unwrap()` on a `None` value")
        }
        self.take().unwrap()
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
    /// Returns the contained value or default (for this type)
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

impl<T: fmt::Default> fmt::Default for Option<T> {
    #[inline]
    fn fmt(s: &Option<T>, f: &mut fmt::Formatter) {
        match *s {
            Some(ref t) => write!(f.buf, "Some({})", *t),
            None        => write!(f.buf, "None")
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
///     fn inc_conditionally(x: uint) -> Option<uint> {
///         if x == uint::max_value { return None; }
///         else { return Some(x+1u); }
///     }
///     let v = [1u, 2, 3];
///     let res = collect(v.iter().map(|&x| inc_conditionally(x)));
///     assert!(res == Some(~[2u, 3, 4]));
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

    let v: V = FromIterator::from_iterator(&mut iter);

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
    use super::*;
    use prelude::*;

    use iter::range;
    use str::StrSlice;
    use util;
    use vec::ImmutableVector;

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
        let addr_x = x.as_ptr();
        let opt = Some(x);
        let y = opt.unwrap();
        let addr_y = y.as_ptr();
        assert_eq!(addr_x, addr_y);
    }

    #[test]
    fn test_get_resource() {
        use rc::Rc;
        use cell::RefCell;

        struct R {
           i: Rc<RefCell<int>>,
        }

        #[unsafe_destructor]
        impl ::ops::Drop for R {
           fn drop(&mut self) {
                let ii = self.i.borrow();
                ii.set(ii.get() + 1);
            }
        }

        fn R(i: Rc<RefCell<int>>) -> R {
            R {
                i: i
            }
        }

        let i = Rc::new(RefCell::new(0));
        {
            let x = R(i.clone());
            let opt = Some(x);
            let _y = opt.unwrap();
        }
        assert_eq!(i.borrow().get(), 1);
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
        assert_eq!(x.and(None::<int>), None);

        let x: Option<int> = None;
        assert_eq!(x.and(Some(2)), None);
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
        let mut i = 0;
        Some(10).while_some(|j| {
            i += 1;
            if (j > 0) {
                Some(j-1)
            } else {
                None
            }
        });
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
        let v: Option<~[int]> = collect(range(0, 0)
                                        .map(|_| Some(0)));
        assert_eq!(v, Some(~[]));

        let v: Option<~[int]> = collect(range(0, 3)
                                        .map(|x| Some(x)));
        assert_eq!(v, Some(~[0, 1, 2]));

        let v: Option<~[int]> = collect(range(0, 3)
                                        .map(|x| if x > 1 { None } else { Some(x) }));
        assert_eq!(v, None);

        // test that it does not take more elements than it needs
        let functions = [|| Some(()), || None, || fail!()];

        let v: Option<~[()]> = collect(functions.iter().map(|f| (*f)()));

        assert_eq!(v, None);
    }
}
