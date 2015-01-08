// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A unique pointer type.

#![stable]

use core::any::Any;
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::fmt;
use core::hash::{self, Hash};
use core::marker::Sized;
use core::mem;
use core::option::Option;
use core::ptr::Unique;
use core::raw::TraitObject;
use core::result::Result;
use core::result::Result::{Ok, Err};
use core::ops::{Deref, DerefMut};

/// A value that represents the global exchange heap. This is the default
/// place that the `box` keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
/// ```rust
/// #![feature(box_syntax)]
/// use std::boxed::HEAP;
///
/// fn main() {
/// # struct Bar;
/// # impl Bar { fn new(_a: int) { } }
///     let foo = box(HEAP) Bar::new(2);
///     let foo = box Bar::new(2);
/// }
/// ```
#[lang = "exchange_heap"]
#[unstable = "may be renamed; uncertain about custom allocator design"]
pub static HEAP: () = ();

/// A type that represents a uniquely-owned value.
#[lang = "owned_box"]
#[stable]
pub struct Box<T>(Unique<T>);

impl<T> Box<T> {
    /// Moves `x` into a freshly allocated box on the global exchange heap.
    #[stable]
    pub fn new(x: T) -> Box<T> {
        box x
    }
}

#[stable]
impl<T: Default> Default for Box<T> {
    #[stable]
    fn default() -> Box<T> { box Default::default() }
}

#[stable]
impl<T> Default for Box<[T]> {
    #[stable]
    fn default() -> Box<[T]> { box [] }
}

#[stable]
impl<T: Clone> Clone for Box<T> {
    /// Returns a copy of the owned box.
    #[inline]
    fn clone(&self) -> Box<T> { box {(**self).clone()} }

    /// Performs copy-assignment from `source` by reusing the existing allocation.
    #[inline]
    fn clone_from(&mut self, source: &Box<T>) {
        (**self).clone_from(&(**source));
    }
}

#[stable]
impl<T: ?Sized + PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool { PartialEq::ne(&**self, &**other) }
}
#[stable]
impl<T: ?Sized + PartialOrd> PartialOrd for Box<T> {
    #[inline]
    fn partial_cmp(&self, other: &Box<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    #[inline]
    fn lt(&self, other: &Box<T>) -> bool { PartialOrd::lt(&**self, &**other) }
    #[inline]
    fn le(&self, other: &Box<T>) -> bool { PartialOrd::le(&**self, &**other) }
    #[inline]
    fn ge(&self, other: &Box<T>) -> bool { PartialOrd::ge(&**self, &**other) }
    #[inline]
    fn gt(&self, other: &Box<T>) -> bool { PartialOrd::gt(&**self, &**other) }
}
#[stable]
impl<T: ?Sized + Ord> Ord for Box<T> {
    #[inline]
    fn cmp(&self, other: &Box<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
#[stable]
impl<T: ?Sized + Eq> Eq for Box<T> {}

#[cfg(stage0)]
impl<S: hash::Writer, T: ?Sized + Hash<S>> Hash<S> for Box<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}
#[cfg(not(stage0))]
impl<S: hash::Hasher, T: ?Sized + Hash<S>> Hash<S> for Box<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

/// Extension methods for an owning `Any` trait object.
#[unstable = "post-DST and coherence changes, this will not be a trait but \
              rather a direct `impl` on `Box<Any>`"]
pub trait BoxAny {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    #[stable]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Self>;
}

impl BoxAny for Box<Any> {
    #[inline]
    #[unstable = "method may be renamed with respect to other downcasting \
                  methods"]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Box<Any>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject =
                    mem::transmute::<Box<Any>, TraitObject>(self);

                // Extract the data pointer
                Ok(mem::transmute(to.data))
            }
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized + fmt::Show> fmt::Show for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Box({:?})", &**self)
    }
}

#[stable]
impl<T: ?Sized + fmt::String> fmt::String for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::String::fmt(&**self, f)
    }
}

impl fmt::Show for Box<Any> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Box<Any>")
    }
}

#[stable]
impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T { &**self }
}

#[stable]
impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T { &mut **self }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_owned_clone() {
        let a = Box::new(5i);
        let b: Box<int> = a.clone();
        assert!(a == b);
    }

    #[test]
    fn any_move() {
        let a = Box::new(8u) as Box<Any>;
        let b = Box::new(Test) as Box<Any>;

        match a.downcast::<uint>() {
            Ok(a) => { assert!(a == Box::new(8u)); }
            Err(..) => panic!()
        }
        match b.downcast::<Test>() {
            Ok(a) => { assert!(a == Box::new(Test)); }
            Err(..) => panic!()
        }

        let a = Box::new(8u) as Box<Any>;
        let b = Box::new(Test) as Box<Any>;

        assert!(a.downcast::<Box<Test>>().is_err());
        assert!(b.downcast::<Box<uint>>().is_err());
    }

    #[test]
    fn test_show() {
        let a = Box::new(8u) as Box<Any>;
        let b = Box::new(Test) as Box<Any>;
        let a_str = a.to_str();
        let b_str = b.to_str();
        assert_eq!(a_str, "Box<Any>");
        assert_eq!(b_str, "Box<Any>");

        let a = &8u as &Any;
        let b = &Test as &Any;
        let s = format!("{}", a);
        assert_eq!(s, "&Any");
        let s = format!("{}", b);
        assert_eq!(s, "&Any");
    }

    #[test]
    fn deref() {
        fn homura<T: Deref<Target=i32>>(_: T) { }
        homura(Box::new(765i32));
    }
}
