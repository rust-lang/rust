// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for generic collections

/// A trait to represent the abstract idea of a container. The only concrete
/// knowledge known is the number of elements contained within.
pub trait Collection {
    /// Return the number of elements in the container
    fn len(&self) -> uint;

    /// Return true if the container contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
