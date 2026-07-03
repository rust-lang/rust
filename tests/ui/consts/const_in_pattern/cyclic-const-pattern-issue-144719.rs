//! Regression test for #144719: using a self-referential static in a
//! pattern position caused a stack overflow during valtree construction.

#[derive(PartialEq)]
struct Thing(&'static Thing);

static X: Thing = Thing(&X);
const Y: &Thing = &X;

fn main() {
    if let Y = Y {}
    //~^ ERROR constant Y cannot be used as pattern
}
