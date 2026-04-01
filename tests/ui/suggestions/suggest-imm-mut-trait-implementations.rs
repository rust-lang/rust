trait Trait {}

struct A;
struct B;
struct C;

impl Trait for &A {}
impl Trait for &mut A {}

impl Trait for &B {}

impl Trait for &mut C {}

fn foo<X: Trait>(_: X) {}

fn main() {
    let a = A;
    let b = B;
    let c = C;
    foo(a); //~ ERROR the trait bound `A: Trait` is not satisfied
    foo(b); //~ ERROR the trait bound `B: Trait` is not satisfied
    foo(c); //~ ERROR the trait bound `C: Trait` is not satisfied
}
