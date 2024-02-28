pub trait Foo {
    type Type;
}

pub struct Bar<T>(<Self as Foo>::Type) where Self: ;
//~^ ERROR trait `Foo` is not implemented for `Bar<T>`

fn main() {}
