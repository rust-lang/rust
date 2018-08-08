// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct Node {
    elem: i32,
    next: Option<Box<Node>>,
}

fn a() {
    let mut node = Node {
        elem: 5,
        next: None,
    };

    let mut src = &mut node;
    {src};
    src.next = None; //~ ERROR use of moved value: `src` [E0382]
}

fn b() {
    let mut src = &mut (22, 44);
    {src};
    src.0 = 66; //~ ERROR use of moved value: `src` [E0382]
}

fn main() {
    a();
    b();
}
