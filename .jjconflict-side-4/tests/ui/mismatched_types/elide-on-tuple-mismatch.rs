//! Regression test for issue #50333: elide irrelevant E0277 errors on tuple mismatch

// Hide irrelevant E0277 errors (#50333)

trait T {}

struct A;

impl T for A {}

impl A {
    fn new() -> Self {
        Self {}
    }
}

fn main() {
    // This creates a tuple type mismatch: 2-element tuple destructured into 3 variables
    let (a, b, c) = (A::new(), A::new());
    //~^ ERROR mismatched types

    // This line should NOT produce an E0277 error about `Sized` trait bounds,
    // because `a`, `b`, and `c` are `TyErr` due to the mismatch above
    let _ts: Vec<&dyn T> = vec![&a, &b, &c];
}
