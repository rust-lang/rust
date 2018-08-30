// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::collections::hash_map::Entry::Vacant;

pub fn foo() {
    type F = Box<Fn(&()) + 'static>;
    let mut map: HashMap<(), F> = HashMap::new();
    let x: &mut F = match map.entry(()) {
        Vacant(_) => unimplemented!(),
        _ => unimplemented!()
    };
}

fn main() {}
