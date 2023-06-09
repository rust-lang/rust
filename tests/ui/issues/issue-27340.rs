struct Foo;
#[derive(Copy, Clone)]
//~^ ERROR the trait `Copy` cannot be implemented for this type
struct Bar(Foo);

fn main() {}
