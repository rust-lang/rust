// Regression test for #66975
#![feature(never_type)]

const VOID: ! = panic!();
//~^ ERROR explicit panic

fn main() {
    let _ = VOID;
}
