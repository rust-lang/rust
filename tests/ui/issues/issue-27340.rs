struct Foo;
#[derive(Copy, Clone)]
//~^ ERROR the trait `Copy` cannot be implemented for this type
struct Bar(Foo);
//~^ ERROR trait `Clone` is not implemented for `Foo`

fn main() {}
