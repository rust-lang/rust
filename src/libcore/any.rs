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
//! use std::fmt::Debug;
//! use std::any::Any;
//!
//! // Logger function for any type that implements Debug.
//! fn log<T: Any + Debug>(value: &T) {
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
//!             println!("{:?}", value);
//!         }
//!     }
//! }
//!
//! // This function wants to log its parameter out prior to doing work with it.
//! fn do_work<T: Debug + 'static>(value: &T) {
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

use mem::transmute;
use option::Option::{self, Some, None};
use raw::TraitObject;
use intrinsics;
use marker::Sized;

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
    #[unstable = "this method will likely be replaced by an associated static"]
    fn get_type_id(&self) -> TypeId;
}

impl<T: 'static> Any for T {
    fn get_type_id(&self) -> TypeId { TypeId::of::<T>() }
}

///////////////////////////////////////////////////////////////////////////////
// Extension methods for Any trait objects.
///////////////////////////////////////////////////////////////////////////////

impl Any {
    /// Returns true if the boxed type is the same as `T`
    #[stable]
    #[inline]
    pub fn is<T: 'static>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[stable]
    #[inline]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute(self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[stable]
    #[inline]
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute(self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// TypeID and its methods
///////////////////////////////////////////////////////////////////////////////

/// A `TypeId` represents a globally unique identifier for a type.
///
/// Each `TypeId` is an opaque object which does not allow inspection of what's
/// inside but does allow basic operations such as cloning, comparison,
/// printing, and showing.
///
/// A `TypeId` is currently only available for types which ascribe to `'static`,
/// but this limitation may be removed in the future.
#[cfg_attr(stage0, lang = "type_id")]
#[derive(Clone, Copy, PartialEq, Eq, Show, Hash)]
#[stable]
pub struct TypeId {
    t: u64,
}

impl TypeId {
    /// Returns the `TypeId` of the type this generic function has been
    /// instantiated with
    #[unstable = "may grow a `Reflect` bound soon via marker traits"]
    pub fn of<T: ?Sized + 'static>() -> TypeId {
        TypeId {
            t: unsafe { intrinsics::type_id::<T>() },
        }
    }
}
