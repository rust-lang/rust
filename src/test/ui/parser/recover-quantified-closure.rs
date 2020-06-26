fn main() {
    for<'a> |x: &'a u8| *x + 1;
    //~^ ERROR cannot introduce explicit parameters for a closure
}

enum Foo { Bar }
fn foo(x: impl Iterator<Item = Foo>) {
    for <Foo>::Bar in x {}
    //~^ ERROR expected one of `move`, `static`, `|`
}
