//@ run-pass
// This is a regression test for something that only came up while
// attempting to bootstrap librustc with new destructor lifetime
// semantics.

#![allow(unexpected_cfgs)] // for the cfg-as-descriptions

use std::collections::HashMap;
use std::cell::RefCell;

// This version does not yet work (associated type issues)...
#[cfg(cannot_use_this_yet)]
fn foo<'a>(map: RefCell<HashMap<&'static str, &'a [u8]>>) {
    let one = [1];
    assert_eq!(map.borrow().get("one"), Some(&one[..]));
}

#[cfg(cannot_use_this_yet_either)]
// ... and this version does not work (the lifetime of `one` is
// supposed to match the lifetime `'a`) ...
fn foo<'a>(map: RefCell<HashMap<&'static str, &'a [u8]>>) {
    let one = [1];
    assert_eq!(map.borrow().get("one"), Some(&&one[..]));
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
    let zer = [0];
    let one = [1];
    let two = [2];
    let mut map = HashMap::new();
    map.insert("zero", &zer[..]);
    map.insert("one",  &one[..]);
    map.insert("two",  &two[..]);
    let map = RefCell::new(map);
    foo(map);
}
