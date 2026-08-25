pub struct Bar(pub u8, u8, u8);

pub fn make_bar() -> Bar {
    Bar(1, 12, 10)
}

mod inner {
    pub struct Foo(u8, pub u8, u8);

    impl Foo {
        pub fn new() -> Foo {
            Foo(1, 12, 10)
        }
    }
}

pub use inner::Foo;
