#![crate_name = "rustdoc_test"]

use std::ops::Deref;

//@ hasraw search.index/name/*.js Foo
pub use private::Foo;

mod private {
    pub struct Foo;
    impl Foo {
        pub fn test_method() {} //@ hasraw - test_method
        fn priv_method() {} //@ !hasraw - priv_method
    }

    pub trait PrivateTrait {
        fn trait_method(&self) {} //@ !hasraw - priv_method
    }
}

pub struct Bar;

impl Deref for Bar {
    //@ !hasraw search.index/name/*.js Target
    type Target = Bar;
    fn deref(&self) -> &Bar { self }
}
