// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_patterns)]
#![feature(box_syntax)]
#![allow(dead_code)]
#![deny(unreachable_patterns)]

enum Foo { A(Box<Foo>, isize), B(usize), }

fn main() {
    match Foo::B(1) {
        Foo::B(_) | Foo::A(box _, 1) => { }
        Foo::A(_, 1) => { } //~ ERROR unreachable pattern
        _ => { }
    }
}

