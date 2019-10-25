pub trait Foo {
    type Type;
}

pub struct Bar<T>(<Self as Foo>::Type) where Self: ;
//~^ ERROR the trait bound `Bar<T>: Foo` is not satisfied

fn main() {}
