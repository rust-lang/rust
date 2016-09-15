// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #35546. Check that we are able to codegen
// this. Before we had problems because of the drop glue signature
// around dropping a trait object (specifically, when dropping the
// `value` field of `Node<Send>`).

struct Node<T: ?Sized + Send> {
    next: Option<Box<Node<Send>>>,
    value: T,
}

fn clear(head: &mut Option<Box<Node<Send>>>) {
    match head.take() {
        Some(node) => *head = node.next,
        None => (),
    }
}

fn main() {}
