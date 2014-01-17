// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the borrow checker prevents pointers to temporaries
// with statement lifetimes from escaping.

#[feature(macro_rules)];

use std::ops::Drop;

static mut FLAGS: u64 = 0;

struct Box<T> { f: T }
struct AddFlags { bits: u64 }

fn AddFlags(bits: u64) -> AddFlags {
    AddFlags { bits: bits }
}

fn arg<'a>(x: &'a AddFlags) -> &'a AddFlags {
    x
}

impl AddFlags {
    fn get<'a>(&'a self) -> &'a AddFlags {
        self
    }
}

pub fn main() {
    let _x = arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let _x = AddFlags(1).get(); //~ ERROR value does not live long enough
    let _x = &*arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let ref _x = *arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let &ref _x = arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let _x = AddFlags(1).get(); //~ ERROR value does not live long enough
    let Box { f: _x } = Box { f: AddFlags(1).get() }; //~ ERROR value does not live long enough
}
