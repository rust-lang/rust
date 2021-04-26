// Regression test for #66975
#![warn(const_err)]

const VOID: ! = panic!();
//~^ ERROR evaluation of constant value failed

fn main() {
    let _ = VOID;
}
