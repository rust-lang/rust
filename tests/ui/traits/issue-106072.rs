#[derive(Clone)] //~ ERROR: expected a type, found a trait
struct Foo; //~ ERROR: expected a type, found a trait
trait Foo {} //~ ERROR: the name `Foo` is defined multiple times
fn main() {}
