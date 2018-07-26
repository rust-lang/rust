// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(unreachable_code, unused_labels)]
fn main() {
    'foo: loop {
        break 'fo; //~ ERROR use of undeclared label
    }

    'bar: loop {
        continue 'bor; //~ ERROR use of undeclared label
    }

    'longlabel: loop {
        'longlabel1: loop {
            break 'longlable; //~ ERROR use of undeclared label
        }
    }
}
