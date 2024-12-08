//@ compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(rustc::potential_query_instability)]

extern crate rustc_data_structures;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};

fn main() {
    let mut x = FxHashMap::<u32, i32>::default();

    for _ in x.drain() {}
    //~^ ERROR using `drain` can result in unstable

    for _ in x.iter() {}
    //~^ ERROR using `iter`

    for _ in Some(&mut x).unwrap().iter_mut() {}
    //~^ ERROR using `iter_mut`

    for _ in x {}
    //~^ ERROR using `into_iter`

    let x = FxHashMap::<u32, i32>::default();
    let _ = x.keys();
    //~^ ERROR using `keys` can result in unstable query results

    let _ = x.values();
    //~^ ERROR using `values` can result in unstable query results

    let mut x = FxHashMap::<u32, i32>::default();
    for val in x.values_mut() {
        //~^ ERROR using `values_mut` can result in unstable query results
        *val = *val + 10;
    }
}
