// build-fail

// Regression test for #66975
#![warn(const_err)]
#![feature(const_panic)]
#![feature(never_type)]

const VOID: ! = panic!();
//~^ WARN any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

fn main() {
    let _ = VOID;
    //~^ ERROR erroneous constant used
}
