// Point at blanket `impl`, which might help the user figure out that `B::from` can only convert to
// `B`. Issue #104712.

struct A;
struct B;

fn foo() {
    let a = A;
    let b = B::from(a); //~ ERROR: mismatched types
}

fn main() {}
