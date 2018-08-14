// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #48728, an ICE that occurred computing
// coherence "help" information.

#[derive(Clone)] //~ ERROR conflicting implementations of trait `std::clone::Clone`
struct Node<T: ?Sized>(Box<T>);

impl<T: Clone + ?Sized> Clone for Node<[T]> {
    fn clone(&self) -> Self {
        Node(Box::clone(&self.0))
    }
}

fn main() {}
