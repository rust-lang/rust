// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:nested-item-spans.rs

extern crate nested_item_spans;

use nested_item_spans::foo;

#[foo]
fn another() {
    fn bar() {
        let x: u32 = "x"; //~ ERROR: mismatched types
    }

    bar();
}

fn main() {
    #[foo]
    fn bar() {
        let x: u32 = "x"; //~ ERROR: mismatched types
    }

    bar();
    another();
}
