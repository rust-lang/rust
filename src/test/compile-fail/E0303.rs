// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    match Some("hi".to_string()) {
        ref op_string_ref @ Some(s) => {},
        //~^ ERROR pattern bindings are not allowed after an `@` [E0303]
        //~| NOTE not allowed after `@`
        //~| ERROR E0009
        //~| NOTE by-move pattern here
        //~| NOTE both by-ref and by-move used
        None => {},
    }
}
