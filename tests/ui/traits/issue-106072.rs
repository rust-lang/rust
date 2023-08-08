#[derive(Clone)] //~  trait objects must include the `dyn` keyword
struct Foo;
trait Foo {} //~ the name `Foo` is defined multiple times
fn main() {}
