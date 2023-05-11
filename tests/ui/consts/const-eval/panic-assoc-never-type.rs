// build-fail

// Regression test for #66975
#![feature(never_type)]

struct PrintName;

impl PrintName {
    const VOID: ! = panic!();
    //~^ ERROR evaluation of constant value failed
}

fn main() {
    let _ = PrintName::VOID; //~ erroneous constant used
}
