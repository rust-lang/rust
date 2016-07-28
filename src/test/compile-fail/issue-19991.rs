// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test if the sugared if-let construct correctly prints "missing an else clause" when an else
// clause does not exist, instead of the unsympathetic "match arms have incompatible types"

fn main() {
    if let Some(homura) = Some("madoka") { //~  ERROR missing an else clause
                                           //~| expected type `()`
                                           //~| found type `{integer}`
                                           //~| expected (), found integral variable
        765
    };
}
