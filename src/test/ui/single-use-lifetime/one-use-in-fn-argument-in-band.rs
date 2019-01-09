// run-rustfix

#![feature(in_band_lifetimes)]
#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO warn when lifetime name is used only
// once in a fn argument, even with in band lifetimes.

fn a(x: &'a u32, y: &'b u32) {
    //~^ ERROR `'a` only used once
    //~| ERROR `'b` only used once
    //~| HELP elide the single-use lifetime
    //~| HELP elide the single-use lifetime
}

fn main() { }
