#[derive(Clone)] //~  trait objects must include the `dyn` keyword
//~^ ERROR: the size for values of type `(dyn Foo + 'static)` cannot be known
struct Foo;
trait Foo {} //~ the name `Foo` is defined multiple times
fn main() {}
