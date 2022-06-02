fn main() {
    for<'a> |x: &'a u8| *x + 1;
    //~^ ERROR use of undeclared lifetime name `'a`
    //~^^ ERROR `for<...>` binders for closures are experimental
    //~^^^ ERROR `for<...>` binders for closures are not yet supported
}

enum Foo { Bar }
fn foo(x: impl Iterator<Item = Foo>) {
    for <Foo>::Bar in x {}
    //~^ ERROR expected one of `move`, `static`, `|`
    //~^^ ERROR `for<...>` binders for closures are experimental
}
