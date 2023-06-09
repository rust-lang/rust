// run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

// pretty-expanded FIXME #23616

pub mod Foo {
    pub trait Trait {
        fn foo(&self);
    }
}

mod Bar {
    impl<'a> dyn (::Foo::Trait) + 'a {
        fn bar(&self) { self.foo() }
    }
}

fn main() {}
