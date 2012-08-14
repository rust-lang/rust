trait Foo {
    fn foo() {}
}

impl int : Foo {
    fn foo() {
        io::println("Hello world!");
    }
}

fn main() {
    let x = 3 as @Foo;
    x.foo();
}

