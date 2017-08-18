#![feature(plugin)]
#![plugin(clippy)]
#![warn(use_self)]
#![allow(dead_code)]


fn main() {}

mod use_self {
    struct Foo {}

    impl Foo {
        fn new() -> Foo {
            Foo {}
        }
        fn test() -> Foo {
            Foo::new()
        }
    }

    impl Default for Foo {
        fn default() -> Foo {
            Foo::new()
        }
    }
}

mod better {
    struct Foo {}

    impl Foo {
        fn new() -> Self {
            Self {}
        }
        fn test() -> Self {
            Self::new()
        }
    }

    impl Default for Foo {
        fn default() -> Self {
            Self::new()
        }
    }
}
