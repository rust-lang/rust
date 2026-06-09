//@ run-rustfix
// https://github.com/rust-lang/rust/issues/82081

use std::collections::HashMap;

struct Test {
    v: u32,
}

fn main() {
    let mut map = HashMap::new();
    map.insert("a", Test { v: 0 });

    for (_k, v) in map.iter() {
        //~^ HELP use mutable method
        //~| NOTE this iterator yields `&` references
        v.v += 1;
        //~^ ERROR cannot assign to `v.v`
        //~| NOTE `v` is a `&` reference
    }
}
