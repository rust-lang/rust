#[derive(Clone)]
struct Foo;
trait Foo {} //~ ERROR: the name `Foo` is defined multiple times
fn main() {}
