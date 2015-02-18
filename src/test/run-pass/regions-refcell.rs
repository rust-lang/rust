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

// This version does not yet work (associated type issues)...
#[cfg(cannot_use_this_yet)]
fn foo<'a>(map: RefCell<HashMap<&'static str, &'a [u8]>>) {
    let one = [1_usize];
    assert_eq!(map.borrow().get("one"), Some(&one[..]));
}

#[cfg(cannot_use_this_yet_either)]
// ... and this version does not work (the lifetime of `one` is
// supposed to match the lifetime `'a`) ...
fn foo<'a>(map: RefCell<HashMap<&'static str, &'a [u8]>>) {
    let one = [1_usize];
    assert_eq!(map.borrow().get("one"), Some(&one.as_slice()));
}

#[cfg(all(not(cannot_use_this_yet),not(cannot_use_this_yet_either)))]
fn foo<'a>(map: RefCell<HashMap<&'static str, &'a [u8]>>) {
    // ...so instead we walk through the trivial slice and make sure
    // it contains the element we expect.

    for (i, &x) in map.borrow().get("one").unwrap().iter().enumerate() {
        assert_eq!((i, x), (0, 1));
    }
}

fn main() {
    let zer = [0u8];
    let one = [1u8];
    let two = [2u8];
    let mut map = HashMap::new();
    map.insert("zero", &zer[..]);
    map.insert("one",  &one[..]);
    map.insert("two",  &two[..]);
    let map = RefCell::new(map);
    foo(map);
}
