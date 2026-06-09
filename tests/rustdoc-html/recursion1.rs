#![crate_type = "lib"]

mod m {
    pub use self::a::Foo;

    mod a {
        pub struct Foo;
    }

    mod b {
        pub use super::*;
    }
}
