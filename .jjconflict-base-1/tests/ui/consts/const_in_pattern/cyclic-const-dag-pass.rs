//@ run-pass
//! Regression test: shared references to the same static (DAG) must not be
//! misidentified as cyclic during valtree construction.

#[derive(PartialEq)]
struct Pair(&'static i32, &'static i32);

static X: i32 = 42;
const P: Pair = Pair(&X, &X);

fn main() {
    if let P = P {}
}
