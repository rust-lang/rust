// Regression test for #66975
#![warn(const_err)]
#![feature(const_panic)]

const VOID: ! = panic!();
//~^ WARN any use of this value will cause an error

fn main() {
    let _ = VOID;
    //~^ ERROR erroneous constant used
}
