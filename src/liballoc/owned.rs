// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A unique pointer type

use core::any::{Any, AnyRefExt};
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::fmt;
use core::intrinsics;
use core::kinds::Send;
use core::mem;
use core::raw::TraitObject;
use core::result::{Ok, Err, Result};

/// A value that represents the global exchange heap. This is the default
/// place that the `box` keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
///     use std::owned::HEAP;
///
///     # struct Bar;
///     # impl Bar { fn new(_a: int) { } }
///     let foo = box(HEAP) Bar::new(2);
///     let foo = box Bar::new(2);
#[lang="exchange_heap"]
pub static HEAP: () = ();

/// A type that represents a uniquely-owned value.
#[lang="owned_box"]
pub struct Box<T>(*mut T);

impl<T: Default> Default for Box<T> {
    fn default() -> Box<T> { box Default::default() }
}

#[unstable]
impl<T: Clone> Clone for Box<T> {
    /// Return a copy of the owned box.
    #[inline]
    fn clone(&self) -> Box<T> { box {(**self).clone()} }

    /// Perform copy-assignment from `source` by reusing the existing allocation.
    #[inline]
    fn clone_from(&mut self, source: &Box<T>) {
        (**self).clone_from(&(**source));
    }
}

// box pointers
impl<T:PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool { *(*self) != *(*other) }
}
impl<T:PartialOrd> PartialOrd for Box<T> {
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
    fn cmp(&self, other: &Box<T>) -> Ordering { (**self).cmp(*other) }
}
impl<T: Eq> Eq for Box<T> {}

/// Extension methods for an owning `Any` trait object
pub trait AnyOwnExt {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    fn move<T: 'static>(self) -> Result<Box<T>, Self>;
}

impl AnyOwnExt for Box<Any> {
    #[inline]
    fn move<T: 'static>(self) -> Result<Box<T>, Box<Any>> {
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

/// Extension methods for an owning `Any+Send` trait object
pub trait AnySendOwnExt {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    fn move_send<T: 'static>(self) -> Result<Box<T>, Self>;
}

impl AnySendOwnExt for Box<Any+Send> {
    #[inline]
    fn move_send<T: 'static>(self) -> Result<Box<T>, Box<Any+Send>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject =
                    *mem::transmute::<&Box<Any+Send>, &TraitObject>(&self);

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
