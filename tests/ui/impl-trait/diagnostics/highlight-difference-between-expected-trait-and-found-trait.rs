//@ only-linux
//@ compile-flags: --error-format=human --color=always

trait Foo<T>: Bar<T> {}

trait Bar<T> {}

struct Struct;

impl<T, K> Foo<K> for T where T: Bar<K>
{}

impl<'a> Bar<()> for Struct {}

fn foo() -> impl Foo<i32> {
    Struct
}

fn main() {}

//~? RAW the trait bound
