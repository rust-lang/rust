#[derive(Clone)] //~  expected a type, found a trait
struct Foo;
trait Foo {} //~ the name `Foo` is defined multiple times
fn main() {}
