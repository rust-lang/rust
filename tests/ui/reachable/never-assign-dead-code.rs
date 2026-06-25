// Test that an assignment of type ! makes the rest of the block dead code.
//
//@ check-pass

#![expect(dropping_copy_types)]
#![warn(unused)]

fn main() {
    let x: ! = panic!("aah"); //~ WARN unused
    drop(x); //~ WARN unreachable
    //~^ WARN unreachable
}
