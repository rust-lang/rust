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
//! `Any` itself can be used to get a `TypeId`, and has more features when used as a trait object.
//! As `&Any` (a borrowed trait object), it has the `is` and `as_ref` methods, to test if the
//! contained value is of a given type, and to get a reference to the inner value as a type. As
//! `&mut Any`, there is also the `as_mut` method, for getting a mutable reference to the inner
//! value. `Box<Any>` adds the `move` method, which will unwrap a `Box<T>` from the object.  See
//! the extension traits (`*Ext`) for the full details.

use mem::{transmute, transmute_copy};
use option::{Option, Some, None};
use raw::TraitObject;
use intrinsics::TypeId;

/// A type with no inhabitants
pub enum Void { }

///////////////////////////////////////////////////////////////////////////////
// Any trait
///////////////////////////////////////////////////////////////////////////////

/// The `Any` trait is implemented by all `'static` types, and can be used for dynamic typing
///
/// Every type with no non-`'static` references implements `Any`, so `Any` can be used as a trait
/// object to emulate the effects dynamic typing.
pub trait Any {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId;
}

impl<T: 'static> Any for T {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
}

///////////////////////////////////////////////////////////////////////////////
// Extension methods for Any trait objects.
// Implemented as three extension traits so that the methods can be generic.
///////////////////////////////////////////////////////////////////////////////

/// Extension methods for a referenced `Any` trait object
pub trait AnyRefExt<'a> {
    /// Returns true if the boxed type is the same as `T`
    fn is<T: 'static>(self) -> bool;

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    fn as_ref<T: 'static>(self) -> Option<&'a T>;
}

impl<'a> AnyRefExt<'a> for &'a Any {
    #[inline]
    fn is<T: 'static>(self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    fn as_ref<T: 'static>(self) -> Option<&'a T> {
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
pub trait AnyMutRefExt<'a> {
    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    fn as_mut<T: 'static>(self) -> Option<&'a mut T>;
}

impl<'a> AnyMutRefExt<'a> for &'a mut Any {
    #[inline]
    fn as_mut<T: 'static>(self) -> Option<&'a mut T> {
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
