// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::container::Map;
use std::hashmap::HashMap;

// Test that trait types printed in error msgs include the type arguments.

fn main() {
    let x: ~HashMap<~str, ~str> = ~HashMap::new();
    let x: ~Map<~str, ~str> = x;
    let y: ~Map<uint, ~str> = ~x;
    //~^ ERROR failed to find an implementation of trait std::container::Map<uint,~str> for ~std::container::Map<~str,~str>:Send
}
