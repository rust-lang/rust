//@ run-rustfix
struct X {}
fn main() {
    let _ = vec![X]; //…
    //~^ ERROR cannot find value `X` in this scope
}
