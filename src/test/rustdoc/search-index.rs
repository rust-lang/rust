#![crate_name = "rustdoc_test"]

use std::ops::Deref;

// @has search-index.js Foo
pub use private::Foo;

mod private {
    pub struct Foo;
    impl Foo {
        pub fn test_method() {} // @has - test_method
        fn priv_method() {} // @!has - priv_method
    }

    pub trait PrivateTrait {
        fn trait_method(&self) {} // @!has - priv_method
    }
}

pub struct Bar;

impl Deref for Bar {
    // @!has search-index.js Target
    type Target = Bar;
    fn deref(&self) -> &Bar { self }
}
