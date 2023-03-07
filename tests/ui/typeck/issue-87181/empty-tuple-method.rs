struct Bar<T> {
    bar: T
}

struct Foo();
impl Foo {
    fn foo(&self) { }
}

fn main() {
    let thing = Bar { bar: Foo };
    thing.bar.foo();
    //~^ ERROR no method named `foo` found for struct constructor `fn() -> Foo {Foo}` in the current scope [E0599]
}
