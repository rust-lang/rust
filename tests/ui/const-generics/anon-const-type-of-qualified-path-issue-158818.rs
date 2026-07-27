//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]

fn with_option<const O: u32>() {
    crate::with_option::<{ async || {} }>();
    //~^ ERROR mismatched types
}
