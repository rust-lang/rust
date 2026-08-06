//! Regression test for #144719: mutually recursive statics forming a
//! reference cycle caused a stack overflow during valtree construction.

#[derive(PartialEq)]
struct Thing(&'static Thing);

static A: Thing = Thing(&B);
static B: Thing = Thing(&A);
const C: &Thing = &A;

fn main() {
    if let C = C {}
    //~^ ERROR constant C cannot be used as pattern
}
