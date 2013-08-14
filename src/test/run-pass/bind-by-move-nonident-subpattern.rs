// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #3761

struct Foo(~str);

enum Tree {
    Leaf(uint),
    Node(~Tree, ~Tree)
}

fn main() {
    match Foo(~"hi") {
        _msg @ Foo(_) => {}
    }

    match Node(~Node(~Leaf(1), ~Leaf(2)), ~Leaf(3)) {
        leaf @ Leaf(*) => { fail!() }
        two_subnodes @ Node(~Node(*), ~Node(*)) => { fail!() }
        other @ Node(_, _) => { /* ok */ }
    }
}
