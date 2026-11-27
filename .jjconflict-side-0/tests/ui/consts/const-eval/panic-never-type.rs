// Regression test for #66975

const VOID: ! = panic!();
//~^ ERROR explicit panic

fn main() {
    let _ = VOID;
}
