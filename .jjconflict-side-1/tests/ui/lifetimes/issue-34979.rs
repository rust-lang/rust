trait Foo {}
impl<'a, T> Foo for &'a T {}

struct Ctx<'a>(&'a ())
where
    &'a (): Foo, //~ ERROR: type annotations needed
    &'static (): Foo;

fn main() {}
