//! A test to ensure that helpful `note` messages aren't emitted more often
//! than necessary.

#![feature(rustc_attrs)]

// Although there are three warnings, we should only get two "lint level defined
// here" notes pointing at the `warnings` span, one for each error type.
#![warn(unused)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let theTwo = 2; //~ WARN should have a snake case name
    let theOtherTwo = 2; //~ WARN should have a snake case name
    //~^ WARN unused variable
    println!("{}", theTwo);
}
