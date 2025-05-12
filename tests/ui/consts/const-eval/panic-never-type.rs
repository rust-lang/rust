// Regression test for #66975
#![feature(never_type)]

const VOID: ! = panic!();
//~^ ERROR evaluation of constant value failed

fn main() {
    let _ = VOID;
}
