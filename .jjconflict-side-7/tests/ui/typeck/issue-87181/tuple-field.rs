struct Bar<T> {
    bar: T
}

struct Foo(char, u16);
impl Foo {
    fn foo() { }
}

fn main() {
    let thing = Bar { bar: Foo };
    thing.bar.0;
    //~^ ERROR no field `0` on type `fn(char, u16) -> Foo {Foo}` [E0609]
}
