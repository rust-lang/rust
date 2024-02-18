struct Foo;
#[derive(Copy, Clone)]
//~^ ERROR the trait `Copy` cannot be implemented for this type
struct Bar(Foo);
//~^ ERROR `Foo: Clone` is not satisfied

fn main() {}
