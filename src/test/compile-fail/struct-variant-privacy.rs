// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
mod foo {
    enum Bar {
        Baz { a: int }
    }
}

fn f(b: foo::Bar) { //~ ERROR enum `Bar` is private
    match b {
        foo::Bar::Baz { a: _a } => {} //~ ERROR variant `Baz` is inaccessible
        // ^~ ERROR enum `Bar` is private
    }
}

fn main() {}
