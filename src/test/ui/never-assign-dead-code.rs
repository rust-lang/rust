// Test that an assignment of type ! makes the rest of the block dead code.

#![feature(never_type)]
// build-pass (FIXME(62277): could be check-pass?)
#![warn(unused)]


fn main() {
    let x: ! = panic!("aah"); //~ WARN unused
    drop(x); //~ WARN unreachable
    //~^ WARN unreachable
}
