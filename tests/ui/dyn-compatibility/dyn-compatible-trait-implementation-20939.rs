// https://github.com/rust-lang/rust/issues/20939
trait Foo {}

impl<'a> Foo for dyn Foo + 'a {}
//~^ ERROR the object type `(dyn Foo + 'a)` automatically implements the trait `Foo`

fn main() {}
