//! Regression test for #144719: long reference cycles shouldn't
//! overflow the stack.
//@ rustc-env:RUST_MIN_STACK=3000000

#[derive(PartialEq, Copy, Clone)]
struct Thing(&'static Thing);

const N: usize = 8000;
static A: Thing = Thing(&B[0]);
static B: [Thing; N] = {
    let mut x = [Thing(&A); N];
    let mut i = 0;
    while i < N - 1 {
        x[i] = Thing(&B[i + 1]);
        i += 1;
    }
    x
};
const C: &Thing = &A;

fn main() {
    if let C = C {}
    //~^ ERROR constant C cannot be used as pattern
}
