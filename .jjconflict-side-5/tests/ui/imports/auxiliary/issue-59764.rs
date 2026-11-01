pub mod foo {
    #[macro_export]
    macro_rules! makro {
        ($foo:ident) => {
            fn $foo() { }
        }
    }

    pub fn baz() {}

    pub fn foobar() {}

    pub mod barbaz {
        pub fn barfoo() {}
    }
}

pub fn foobaz() {}
