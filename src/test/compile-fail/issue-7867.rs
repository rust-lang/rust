// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum A { B, C }

mod foo { pub fn bar() {} }

fn main() {
    match (true, false) {
        A::B => (), //~ ERROR expected `(bool, bool)`, found `A` (expected tuple, found enum A)
        _ => ()
    }

    match &Some(42is) {
        Some(x) => (), //~ ERROR expected `&core::option::Option<isize>`,
                       //        found `core::option::Option<_>`
        None => ()     //~ ERROR expected `&core::option::Option<isize>`,
                       //        found `core::option::Option<_>`
    }
}
