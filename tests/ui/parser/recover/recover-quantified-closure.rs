fn main() {
    for<'a> |x: &'a u8| *x + 1;
    //~^ ERROR `for<...>` binders for closures are experimental
    //~^^ ERROR implicit types in closure signatures are forbidden when `for<...>` is present
}

enum Foo { Bar }
fn foo(x: impl Iterator<Item = Foo>) {
    for <Foo>::Bar in x {}
    //~^ ERROR expected one of `move`, `static`, `use`, `|`
    //~^^ ERROR `for<...>` binders for closures are experimental
}
