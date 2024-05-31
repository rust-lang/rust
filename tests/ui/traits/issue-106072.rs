#[derive(Clone)] //~ERROR trait objects must include the `dyn` keyword
struct Foo;
trait Foo {} //~ERROR the name `Foo` is defined multiple times
fn main() {}
