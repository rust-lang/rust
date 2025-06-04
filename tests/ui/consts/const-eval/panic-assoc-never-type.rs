//@ build-fail
//@ dont-require-annotations: NOTE

// Regression test for #66975
#![feature(never_type)]

struct PrintName;

impl PrintName {
    const VOID: ! = panic!();
    //~^ ERROR explicit panic
}

fn main() {
    let _ = PrintName::VOID; //~ NOTE erroneous constant encountered
}
