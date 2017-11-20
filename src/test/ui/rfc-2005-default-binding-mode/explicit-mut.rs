// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

// Verify the binding mode shifts - only when no `&` are auto-dereferenced is the
// final default binding mode mutable.

fn main() {
    match &&Some(5i32) {
        Some(n) => {
            *n += 1; //~ ERROR cannot assign to immutable
            let _ = n;
        }
        None => {},
    };

    match &mut &Some(5i32) {
        Some(n) => {
            *n += 1; //~ ERROR cannot assign to immutable
            let _ = n;
        }
        None => {},
    };

    match &&mut Some(5i32) {
        Some(n) => {
            *n += 1; //~ ERROR cannot assign to immutable
            let _ = n;
        }
        None => {},
    };
}
