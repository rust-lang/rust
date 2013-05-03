// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn not_bool(f: &fn(int) -> ~str) -> bool {}

fn main() {
    for uint::range(0, 100000) |_i| { //~ ERROR A for-loop body must return (), but
        false
    };
    for not_bool |_i| {
    //~^ ERROR A `for` loop iterator should expect a closure that returns `bool`
        ~"hi"
    };
    for uint::range(0, 100000) |_i| { //~ ERROR A for-loop body must return (), but
        ~"hi"
    };
    for not_bool() |_i| {
    //~^ ERROR A `for` loop iterator should expect a closure that returns `bool`
    };
}
