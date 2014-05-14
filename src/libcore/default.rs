// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `Default` trait for types which may have meaningful default values

/// A trait that types which have a useful default value should implement.
pub trait Default {
    /// Return the "default value" for a type.
    fn default() -> Self;
}

impl<T: Default + 'static> Default for @T {
    fn default() -> @T { @Default::default() }
}
