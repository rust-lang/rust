#[non_exhaustive(anything)]
//~^ ERROR malformed `non_exhaustive` attribute
struct Foo;

#[non_exhaustive]
//~^ ERROR attribute cannot be used on
trait Bar { }

#[non_exhaustive]
//~^ ERROR attribute cannot be used on
union Baz {
    f1: u16,
    f2: u16
}

fn main() { }
