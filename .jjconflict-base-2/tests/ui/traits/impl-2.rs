//@ run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]


pub mod Foo {
    pub trait Trait {
        fn foo(&self);
    }
}

mod Bar {
    impl<'a> dyn crate::Foo::Trait + 'a {
        fn bar(&self) { self.foo() }
    }
}

fn main() {}
