struct Qux;

trait Foo {
    fn foo();
}

trait FooBar {
    fn foo() {}
}

fn main() {
    Qux.foo();
    //~^ ERROR no method named `foo` found for struct `Qux` in the current scope
}
