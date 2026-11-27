mod inner {
    pub struct Foo;

    impl Foo {
        fn method(&self) {}
    }
}

fn main() {
    let foo = inner::Foo;
    foo.method(); //~ ERROR method `method` is private [E0624]
}
