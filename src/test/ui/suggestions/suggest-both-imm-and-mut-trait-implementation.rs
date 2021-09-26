trait Trait {}

struct S;

impl Trait for &S {}
impl Trait for &mut S {}

fn foo<X: Trait>(_: X) {}

fn main() {
    let s = S;
    foo(s); //~ ERROR the trait bound `S: Trait` is not satisfied
}
