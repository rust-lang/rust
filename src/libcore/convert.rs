// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for conversions between types.
//!
//! The traits in this module provide a general way to talk about
//! conversions from one type to another. They follow the standard
//! Rust conventions of `as`/`to`/`into`/`from`.

#![unstable(feature = "convert",
            reason = "recently added, experimental traits")]

use marker::Sized;

/// A cheap, reference-to-reference conversion.
pub trait AsRef<T: ?Sized> {
    /// Perform the conversion.
    fn as_ref(&self) -> &T;
}

/// A cheap, mutable reference-to-mutable reference conversion.
pub trait AsMut<T: ?Sized> {
    /// Perform the conversion.
    fn as_mut(&mut self) -> &mut T;
}

/// A conversion that consumes `self`, which may or may not be
/// expensive.
pub trait Into<T>: Sized {
    /// Perform the conversion.
    fn into(self) -> T;
}

/// Construct `Self` via a conversion.
pub trait From<T> {
    /// Perform the conversion.
    fn from(T) -> Self;
}

////////////////////////////////////////////////////////////////////////////////
// GENERIC IMPLS
////////////////////////////////////////////////////////////////////////////////

// As implies Into
impl<'a, T: ?Sized, U: ?Sized> Into<&'a U> for &'a T where T: AsRef<U> {
    fn into(self) -> &'a U {
        self.as_ref()
    }
}

// As lifts over &
impl<'a, T: ?Sized, U: ?Sized> AsRef<U> for &'a T where T: AsRef<U> {
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(*self)
    }
}

// As lifts over &mut
impl<'a, T: ?Sized, U: ?Sized> AsRef<U> for &'a mut T where T: AsRef<U> {
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(*self)
    }
}

// FIXME (#23442): replace the above impls for &/&mut with the following more general one:
// // As lifts over Deref
// impl<D: ?Sized + Deref, U: ?Sized> AsRef<U> for D where D::Target: AsRef<U> {
//     fn as_ref(&self) -> &U {
//         self.deref().as_ref()
//     }
// }

// AsMut implies Into
impl<'a, T: ?Sized, U: ?Sized> Into<&'a mut U> for &'a mut T where T: AsMut<U> {
    fn into(self) -> &'a mut U {
        (*self).as_mut()
    }
}

// AsMut lifts over &mut
impl<'a, T: ?Sized, U: ?Sized> AsMut<U> for &'a mut T where T: AsMut<U> {
    fn as_mut(&mut self) -> &mut U {
        (*self).as_mut()
    }
}

// FIXME (#23442): replace the above impl for &mut with the following more general one:
// // AsMut lifts over DerefMut
// impl<D: ?Sized + Deref, U: ?Sized> AsMut<U> for D where D::Target: AsMut<U> {
//     fn as_mut(&mut self) -> &mut U {
//         self.deref_mut().as_mut()
//     }
// }

// From itself is always itself
impl<T> From<T> for T {
    fn from(t: T) -> T {
        t
    }
}

// From implies Into
impl<T, U> Into<U> for T where U: From<T> {
    fn into(self) -> U {
        U::from(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// CONCRETE IMPLS
////////////////////////////////////////////////////////////////////////////////

impl<T> AsRef<[T]> for [T] {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T> AsMut<[T]> for [T] {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl AsRef<str> for str {
    fn as_ref(&self) -> &str {
        self
    }
}
