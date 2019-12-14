// build-fail

// Regression test for #66975
#![warn(const_err)]
#![feature(const_panic)]
#![feature(never_type)]

const VOID: ! = panic!();
//~^ WARN any use of this value will cause an error

fn main() {
    let _ = VOID;
    //~^ ERROR erroneous constant used
}
