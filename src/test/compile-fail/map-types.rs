// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate collections;

use std::collections::HashMap;

// Test that trait types printed in error msgs include the type arguments.

fn main() {
    let x: Box<HashMap<int, int>> = box HashMap::new();
    let x: Box<Map<int, int>> = x;
    let y: Box<Map<uint, int>> = box x;
    //~^ ERROR failed to find an implementation of trait collections::Map<uint,int>
    //~^^ ERROR failed to find an implementation of trait core::collections::Collection
}
