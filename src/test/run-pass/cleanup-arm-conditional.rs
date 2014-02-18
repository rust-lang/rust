// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// copyright 2014 the rust project developers. see the copyright
// file at the top-level directory of this distribution and at
// http://rust-lang.org/copyright.
//
// licensed under the apache license, version 2.0 <license-apache or
// http://www.apache.org/licenses/license-2.0> or the mit license
// <license-mit or http://opensource.org/licenses/mit>, at your
// option. this file may not be copied, modified, or distributed
// except according to those terms.

// Test that cleanup scope for temporaries created in a match
// arm is confined to the match arm itself.

use std::os;

struct Test { x: int }

impl Test {
    fn get_x(&self) -> Option<~int> {
        Some(~self.x)
    }
}

fn do_something(t: &Test) -> int {

    // The cleanup scope for the result of `t.get_x()` should be the
    // arm itself and not the match, otherwise we'll (potentially) get
    // a crash trying to free an uninitialized stack slot.

    match t {
        &Test { x: 2 } if t.get_x().is_some() => {
            t.x * 2
        }
        _ => { 22 }
    }
}

pub fn main() {
    let t = Test { x: 1 };
    do_something(&t);
}

