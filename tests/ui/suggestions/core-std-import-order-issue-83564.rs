//@ edition:2018
//
// This is a regression test for #83564.
// For some reason, Rust 2018 or higher is required to reproduce the bug.
#![feature(generic_nonzero)]

fn main() {
    //~^ HELP consider importing one of these items
    let _x = NonZero::new(5u32).unwrap();
    //~^ ERROR failed to resolve: use of undeclared type `NonZero`
}
