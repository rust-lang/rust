// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:namespaced_enums.rs
#![feature(globs)]

extern crate namespaced_enums;

fn _f(f: namespaced_enums::Foo) {
    use namespaced_enums::Foo::*;

    match f {
        A | B(_) | C { .. } => {}
    }
}

mod m {
    pub use namespaced_enums::Foo::*;
}

fn _f2(f: namespaced_enums::Foo) {
    match f {
        m::A | m::B(_) | m::C { .. } => {}
    }
}

pub fn main() {}
