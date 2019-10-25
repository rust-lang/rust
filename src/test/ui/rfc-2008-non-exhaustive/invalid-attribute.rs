#![feature(non_exhaustive)]

#[non_exhaustive(anything)]
//~^ ERROR malformed `non_exhaustive` attribute
struct Foo;

#[non_exhaustive]
//~^ ERROR attribute can only be applied to a struct or enum [E0701]
trait Bar { }

#[non_exhaustive]
//~^ ERROR attribute can only be applied to a struct or enum [E0701]
union Baz {
    f1: u16,
    f2: u16
}

fn main() { }
