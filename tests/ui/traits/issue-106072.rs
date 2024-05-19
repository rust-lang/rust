#[derive(Clone)] //~  trait objects must include the `dyn` keyword
//~^ ERROR: the size for values of type `(dyn Foo + 'static)` cannot be known
//~| ERROR: return type cannot have an unboxed trait object
struct Foo;
trait Foo {} //~ the name `Foo` is defined multiple times
fn main() {}
