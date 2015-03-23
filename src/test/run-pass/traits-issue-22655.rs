// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #22655: This test should not lead to
// infinite recursion.

// pretty-expanded FIXME #23616

unsafe impl<T: Send + ?Sized> Send for Unique<T> { }

pub struct Unique<T:?Sized> {
    pointer: *const T,
}

pub struct Node<V> {
    vals: V,
    edges: Unique<Node<V>>,
}

fn is_send<T: Send>() {}

fn main() {
    is_send::<Node<&'static ()>>();
}
