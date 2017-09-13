// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::mem;

/// Returns the size of a type
pub fn size_of<T>() -> usize {
    TypeInfo::size_of(None::<T>)
}

/// Returns the size of the type that `val` points to
pub fn size_of_val<T>(val: &T) -> usize {
    val.size_of_val()
}

pub trait TypeInfo: Sized {
    fn size_of(_lame_type_hint: Option<Self>) -> usize;
    fn size_of_val(&self) -> usize;
}

impl<T> TypeInfo for T {
    /// The size of the type in bytes.
    fn size_of(_lame_type_hint: Option<T>) -> usize {
        mem::size_of::<T>()
    }

    /// Returns the size of the type of `self` in bytes.
    fn size_of_val(&self) -> usize {
        TypeInfo::size_of(None::<T>)
    }
}

pub fn main() {}
