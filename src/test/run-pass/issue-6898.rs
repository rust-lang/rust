// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::intrinsics;

/// Returns the size of a type
pub fn size_of<T>() -> uint {
    TypeInfo::size_of(None::<T>)
}

/// Returns the size of the type that `val` points to
pub fn size_of_val<T>(val: &T) -> uint {
    val.size_of_val()
}

pub trait TypeInfo {
    fn size_of(_lame_type_hint: Option<Self>) -> uint;
    fn size_of_val(&self) -> uint;
}

impl<T> TypeInfo for T {
    /// The size of the type in bytes.
    fn size_of(_lame_type_hint: Option<T>) -> uint {
        unsafe { intrinsics::size_of::<T>() }
    }

    /// Returns the size of the type of `self` in bytes.
    fn size_of_val(&self) -> uint {
        TypeInfo::size_of(None::<T>)
    }
}

pub fn main() {}
