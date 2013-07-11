// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Container traits for extra

use std::container::Mutable;

/// A double-ended sequence that allows querying, insertion and deletion at both ends.
pub trait Deque<T> : Mutable {
    /// Provide a reference to the front element, or None if the sequence is empty
    fn front<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the front element, or None if the sequence is empty
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Provide a reference to the back element, or None if the sequence is empty
    fn back<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the back element, or None if the sequence is empty
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Insert an element first in the sequence
    fn push_front(&mut self, elt: T);

    /// Insert an element last in the sequence
    fn push_back(&mut self, elt: T);

    /// Remove the last element and return it, or None if the sequence is empty
    fn pop_back(&mut self) -> Option<T>;

    /// Remove the first element and return it, or None if the sequence is empty
    fn pop_front(&mut self) -> Option<T>;
}
