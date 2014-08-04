// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A unique pointer type.

use core::any::{Any, AnyRefExt};
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::fmt;
use core::intrinsics;
use core::mem;
use core::option::Option;
use core::raw::TraitObject;
use core::result::{Ok, Err, Result};

/// A value that represents the global exchange heap. This is the default
/// place that the `box` keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
/// ```rust
/// use std::boxed::HEAP;
///
/// # struct Bar;
/// # impl Bar { fn new(_a: int) { } }
/// let foo = box(HEAP) Bar::new(2);
/// let foo = box Bar::new(2);
/// ```
#[lang = "exchange_heap"]
#[experimental = "may be renamed; uncertain about custom allocator design"]
pub static HEAP: () = ();

/// A type that represents a uniquely-owned value.
#[lang = "owned_box"]
#[unstable = "custom allocators will add an additional type parameter (with default)"]
pub struct Box<T>(*mut T);

impl<T: Default> Default for Box<T> {
    fn default() -> Box<T> { box Default::default() }
}

#[unstable]
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

impl<T:PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool { *(*self) != *(*other) }
}
impl<T:PartialOrd> PartialOrd for Box<T> {
    #[inline]
    fn partial_cmp(&self, other: &Box<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
    #[inline]
    fn lt(&self, other: &Box<T>) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &Box<T>) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &Box<T>) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &Box<T>) -> bool { *(*self) > *(*other) }
}
impl<T: Ord> Ord for Box<T> {
    #[inline]
    fn cmp(&self, other: &Box<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: Eq> Eq for Box<T> {}

/// Extension methods for an owning `Any` trait object.
#[unstable = "post-DST and coherence changes, this will not be a trait but \
              rather a direct `impl` on `Box<Any>`"]
pub trait BoxAny {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    #[unstable = "naming conventions around accessing innards may change"]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Self>;

    /// Deprecated; this method has been renamed to `downcast`.
    #[deprecated = "use downcast instead"]
    fn move<T: 'static>(self) -> Result<Box<T>, Self> {
        self.downcast::<T>()
    }
}

#[stable]
impl BoxAny for Box<Any> {
    #[inline]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Box<Any>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject =
                    *mem::transmute::<&Box<Any>, &TraitObject>(&self);

                // Prevent destructor on self being run
                intrinsics::forget(self);

                // Extract the data pointer
                Ok(mem::transmute(to.data))
            }
        } else {
            Err(self)
        }
    }
}

impl<T: fmt::Show> fmt::Show for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl fmt::Show for Box<Any> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Box<Any>")
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_owned_clone() {
        let a = box 5i;
        let b: Box<int> = a.clone();
        assert!(a == b);
    }

    #[test]
    fn any_move() {
        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;

        match a.downcast::<uint>() {
            Ok(a) => { assert!(a == box 8u); }
            Err(..) => fail!()
        }
        match b.downcast::<Test>() {
            Ok(a) => { assert!(a == box Test); }
            Err(..) => fail!()
        }

        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;

        assert!(a.downcast::<Box<Test>>().is_err());
        assert!(b.downcast::<Box<uint>>().is_err());
    }

    #[test]
    fn test_show() {
        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;
        let a_str = a.to_str();
        let b_str = b.to_str();
        assert_eq!(a_str.as_slice(), "Box<Any>");
        assert_eq!(b_str.as_slice(), "Box<Any>");

        let a = &8u as &Any;
        let b = &Test as &Any;
        let s = format!("{}", a);
        assert_eq!(s.as_slice(), "&Any");
        let s = format!("{}", b);
        assert_eq!(s.as_slice(), "&Any");
    }
}
