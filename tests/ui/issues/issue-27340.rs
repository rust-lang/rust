struct Foo;
#[derive(Copy, Clone)]
struct Bar(Foo);
//~^ ERROR: the trait `Copy` cannot be implemented for this type
//~| ERROR: `Foo: Clone` is not satisfied

fn main() {}
