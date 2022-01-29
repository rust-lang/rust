struct Foo;
#[derive(Copy, Clone)]
//~^ ERROR the trait `Copy` may not be implemented for this type
struct Bar(Foo);

fn main() {}
