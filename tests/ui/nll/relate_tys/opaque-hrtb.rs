trait MyTrait<T> {}

struct Foo;
impl<T> MyTrait<T> for Foo {}

fn bar<Input>() -> impl MyTrait<Input> {
    Foo
}

fn foo() -> impl for<'a> MyTrait<&'a str> {
    bar() //~ ERROR implementation of `MyTrait` is not general enough
}

fn main() {}
