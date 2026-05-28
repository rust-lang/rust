//@ edition: 2021

trait Foo {
    async fn foo<T>();
}

impl Foo for () {
    async fn foo<const N: usize>() {}
    //~^ ERROR: associated function `foo` has an incompatible generic parameter for trait `Foo` [E0053]
}

fn main() {}
