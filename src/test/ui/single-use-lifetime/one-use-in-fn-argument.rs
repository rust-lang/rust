#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO warn when lifetime name is used only
// once in a fn argument.

fn a<'a>(x: &'a u32) { //~ ERROR `'a` only used once
    //~^ HELP elide the single-use lifetime
}

fn main() { }
