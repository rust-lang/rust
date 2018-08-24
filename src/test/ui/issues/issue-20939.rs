trait Foo {}

impl<'a> Foo for Foo+'a {}
//~^ ERROR the object type `(dyn Foo + 'a)` automatically implements the trait `Foo`

fn main() {}
