//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// test that the type alias impl trait defining use is in a submodule

fn main() {}

type Foo = impl std::fmt::Display;
type Bar = impl std::fmt::Display;

mod foo {
    #[define_opaque(super::Foo)]
    pub(crate) fn foo() -> super::Foo {
        "foo"
    }

    pub(crate) mod bar {
        #[define_opaque(crate::Bar)]
        pub(crate) fn bar() -> crate::Bar {
            1
        }
    }
}

mod bar {
    pub type Baz = impl std::fmt::Display;
}

#[define_opaque(bar::Baz)]
fn baz() -> bar::Baz {
    "boom"
}
