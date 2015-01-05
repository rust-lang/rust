// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a regression test for something that only came up while
// attempting to bootstrap librustc with new destructor lifetime
// semantics.

use std::collections::HashMap;
use std::cell::RefCell;

fn foo(map: RefCell<HashMap<&'static str, &[uint]>>) {
    // assert_eq!(map.borrow().get("one"), Some(&1u));
    let map = map.borrow();
    for (i, &x) in map.get("one").unwrap().iter().enumerate() {
        assert_eq!((i, x), (0u, 1u));
    }
}

fn main() {
    let zer = [0u];
    let one = [1u];
    let two = [2u];
    let mut map = HashMap::new();
    map.insert("zero", zer.as_slice());
    map.insert("one",  one.as_slice());
    map.insert("two",  two.as_slice());
    let map = RefCell::new(map);
    foo(map);
}
