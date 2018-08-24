// Test that an assignment of type ! makes the rest of the block dead code.

#![feature(never_type)]
#![feature(rustc_attrs)]
#![warn(unused)]

#[rustc_error]
fn main() { //~ ERROR: compilation successful
    let x: ! = panic!("aah"); //~ WARN unused
    drop(x); //~ WARN unreachable
    //~^ WARN unreachable
}
