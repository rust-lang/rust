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
    let (a, b, c) = (A::new(), A::new()); // This tuple is 2 elements, should be three
    //~^ ERROR mismatched types
    let ts: Vec<&dyn T> = vec![&a, &b, &c];
    // There is no E0277 error above, as `a`, `b` and `c` are `TyErr`
}
