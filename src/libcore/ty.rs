// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Types dealing with unsafe actions.

use kinds::marker;

/// Unsafe type that wraps a type T and indicates unsafe interior operations on the
/// wrapped type. Types with an `Unsafe<T>` field are considered to have an *unsafe
/// interior*. The Unsafe type is the only legal way to obtain aliasable data that is
/// considered mutable. In general, transmuting an &T type into an &mut T is considered
/// undefined behavior.
///
/// Although it is possible to put an Unsafe<T> into static item, it is not permitted to
/// take the address of the static item if the item is not declared as mutable. This rule
/// exists because immutable static items are stored in read-only memory, and thus any
/// attempt to mutate their interior can cause segfaults. Immutable static items containing
/// Unsafe<T> instances are still useful as read-only initializers, however, so we do not
/// forbid them altogether.
///
/// Types like `Cell` and `RefCell` use this type to wrap their internal data.
///
/// Unsafe doesn't opt-out from any kind, instead, types with an `Unsafe` interior
/// are expected to opt-out from kinds themselves.
///
/// # Example:
///
/// ```rust
/// use std::ty::Unsafe;
/// use std::kinds::marker;
///
/// struct NotThreadSafe<T> {
///     value: Unsafe<T>,
///     marker1: marker::NoShare
/// }
/// ```
///
/// **NOTE:** Unsafe<T> fields are public to allow static initializers. It is not recommended
/// to access its fields directly, `get` should be used instead.
#[lang="unsafe"]
pub struct Unsafe<T> {
    /// Wrapped value
    pub value: T,

    /// Invariance marker
    pub marker1: marker::InvariantType<T>
}

impl<T> Unsafe<T> {

    /// Static constructor
    pub fn new(value: T) -> Unsafe<T> {
        Unsafe{value: value, marker1: marker::InvariantType}
    }

    /// Gets a mutable pointer to the wrapped value
    #[inline]
    pub unsafe fn get(&self) -> *mut T { &self.value as *T as *mut T }

    /// Unwraps the value
    #[inline]
    pub unsafe fn unwrap(self) -> T { self.value }
}
