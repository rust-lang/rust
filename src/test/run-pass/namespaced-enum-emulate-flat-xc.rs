// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:namespaced_enum_emulate_flat.rs

extern crate namespaced_enum_emulate_flat;

use namespaced_enum_emulate_flat::{Foo, A, B, C};
use namespaced_enum_emulate_flat::nest::{Bar, D, E, F};

fn _f(f: Foo) {
    match f {
        A | B(_) | C { .. } => {}
    }
}

fn _f2(f: Bar) {
    match f {
        D | E(_) | F { .. } => {}
    }
}

pub fn main() {}

