#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

//@ check-pass

mod m {
    pub type Foo = impl std::fmt::Debug;
    #[define_opaque(Foo)]
    pub fn foo() -> Foo {
        22_u32
    }

    pub fn bar() {
        is_send(foo());
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
