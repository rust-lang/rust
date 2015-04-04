// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum BtNode {
    Node(u32,Box<BtNode>,Box<BtNode>),
    Leaf(u32),
}

fn main() {
    let y = match x {
        Foo<T>::A(value) => value, //~ error: expected one of `=>`, `@`, `if`, or `|`, found `<`
        Foo<T>::B => 7,
    };
}
