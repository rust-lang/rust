struct Bar<T> {
    bar: T
}

struct Foo(u8, i32);
impl Foo {
    fn foo() { }
}

fn main() {
    let thing = Bar { bar: Foo };
    thing.bar.foo();
    //~^ ERROR no method named `foo` found for struct constructor `fn(u8, i32) -> Foo {Foo}` in the current scope [E0599]
}
