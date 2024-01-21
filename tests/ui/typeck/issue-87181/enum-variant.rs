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
    //~^ ERROR no method named `foo` found for enum constructor `{fn item Foo::Tup: fn() -> Foo}` in the current scope [E0599]
}
