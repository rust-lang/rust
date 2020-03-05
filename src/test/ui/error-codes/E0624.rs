mod inner {
    pub struct Foo;

    impl Foo {
        fn method(&self) {}
    }
}

fn main() {
    let foo = inner::Foo;
    foo.method(); //~ ERROR associated function `method` is private [E0624]
}
