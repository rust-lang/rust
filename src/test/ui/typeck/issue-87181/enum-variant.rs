struct Bar<T> {
    bar: T
}

enum Foo{
    Tup()
}
impl Foo {
    fn foo(&self) { }
}

fn main() {
    let thing = Bar { bar: Foo::Tup };
    thing.bar.foo();
    //~^ ERROR no method named `foo` found for fn item `fn() -> Foo {Foo::Tup}` in the current scope [E0599]
}
