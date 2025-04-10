#[derive(Clone)]
//~^ ERROR expected a type, found a trait
//~| ERROR expected a type, found a trait
struct Foo;
trait Foo {} //~ ERROR the name `Foo` is defined multiple times
fn main() {}
