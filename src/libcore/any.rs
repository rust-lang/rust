// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for dynamic typing of any `'static` type (through runtime reflection)
//!
//! This module implements the `Any` trait, which enables dynamic typing
//! of any `'static` type through runtime reflection.
//!
//! `Any` itself can be used to get a `TypeId`, and has more features when used
//! as a trait object. As `&Any` (a borrowed trait object), it has the `is` and
//! `as_ref` methods, to test if the contained value is of a given type, and to
//! get a reference to the inner value as a type. As`&mut Any`, there is also
//! the `as_mut` method, for getting a mutable reference to the inner value.
//! `Box<Any>` adds the `move` method, which will unwrap a `Box<T>` from the
//! object.  See the extension traits (`*Ext`) for the full details.
//!
//! Note that &Any is limited to testing whether a value is of a specified
//! concrete type, and cannot be used to test whether a type implements a trait.
//!
//! # Examples
//!
//! Consider a situation where we want to log out a value passed to a function.
//! We know the value we're working on implements Show, but we don't know its
//! concrete type.  We want to give special treatment to certain types: in this
//! case printing out the length of String values prior to their value.
//! We don't know the concrete type of our value at compile time, so we need to
//! use runtime reflection instead.
//!
//! ```rust
//! use std::fmt::Show;
//! use std::any::{Any, AnyRefExt};
//!
//! // Logger function for any type that implements Show.
//! fn log<T: Any+Show>(value: &T) {
//!     let value_any = value as &Any;
//!
//!     // try to convert our value to a String.  If successful, we want to
//!     // output the String's length as well as its value.  If not, it's a
//!     // different type: just print it out unadorned.
//!     match value_any.downcast_ref::<String>() {
//!         Some(as_string) => {
//!             println!("String ({}): {}", as_string.len(), as_string);
//!         }
//!         None => {
//!             println!("{}", value);
//!         }
//!     }
//! }
//!
//! // This function wants to log its parameter out prior to doing work with it.
//! fn do_work<T: Show+'static>(value: &T) {
//!     log(value);
//!     // ...do some other work
//! }
//!
//! fn main() {
//!     let my_string = "Hello World".to_string();
//!     do_work(&my_string);
//!
//!     let my_i8: i8 = 100;
//!     do_work(&my_i8);
//! }
//! ```

#![stable]

use mem::{transmute, transmute_copy};
use option::{Option, Some, None};
use raw::TraitObject;
use intrinsics::TypeId;

/// A type with no inhabitants
#[deprecated = "this type is being removed, define a type locally if \
                necessary"]
pub enum Void { }

///////////////////////////////////////////////////////////////////////////////
// Any trait
///////////////////////////////////////////////////////////////////////////////

/// The `Any` trait is implemented by all `'static` types, and can be used for
/// dynamic typing
///
/// Every type with no non-`'static` references implements `Any`, so `Any` can
/// be used as a trait object to emulate the effects dynamic typing.
#[stable]
pub trait Any: 'static {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId;
}

impl<T: 'static> Any for T {
    fn get_type_id(&self) -> TypeId { TypeId::of::<T>() }
}

///////////////////////////////////////////////////////////////////////////////
// Extension methods for Any trait objects.
// Implemented as three extension traits so that the methods can be generic.
///////////////////////////////////////////////////////////////////////////////

/// Extension methods for a referenced `Any` trait object
#[unstable = "this trait will not be necessary once DST lands, it will be a \
              part of `impl Any`"]
pub trait AnyRefExt<'a> {
    /// Returns true if the boxed type is the same as `T`
    #[stable]
    fn is<T: 'static>(self) -> bool;

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[unstable = "naming conventions around acquiring references may change"]
    fn downcast_ref<T: 'static>(self) -> Option<&'a T>;

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[deprecated = "this function has been renamed to `downcast_ref`"]
    fn as_ref<T: 'static>(self) -> Option<&'a T> {
        self.downcast_ref::<T>()
    }
}

#[stable]
impl<'a> AnyRefExt<'a> for &'a Any {
    #[inline]
    #[stable]
    fn is<T: 'static>(self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    #[unstable = "naming conventions around acquiring references may change"]
    fn downcast_ref<T: 'static>(self) -> Option<&'a T> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute_copy(&self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }
}

/// Extension methods for a mutable referenced `Any` trait object
#[unstable = "this trait will not be necessary once DST lands, it will be a \
              part of `impl Any`"]
pub trait AnyMutRefExt<'a> {
    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[unstable = "naming conventions around acquiring references may change"]
    fn downcast_mut<T: 'static>(self) -> Option<&'a mut T>;

    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[deprecated = "this function has been renamed to `downcast_mut`"]
    fn as_mut<T: 'static>(self) -> Option<&'a mut T> {
        self.downcast_mut::<T>()
    }
}

#[stable]
impl<'a> AnyMutRefExt<'a> for &'a mut Any {
    #[inline]
    #[unstable = "naming conventions around acquiring references may change"]
    fn downcast_mut<T: 'static>(self) -> Option<&'a mut T> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute_copy(&self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }
}
