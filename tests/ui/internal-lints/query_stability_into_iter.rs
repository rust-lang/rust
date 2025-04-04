//@ compile-flags: -Z unstable-options

#![deny(rustc::potential_query_instability)]

use std::collections::HashSet;

fn main() {
    let set = HashSet::<u32>::default();
    HashSet::<u32>::default().extend(set);
    //~^ ERROR using `into_iter` can result in unstable query results
}
