// check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    type Foo = impl std::fmt::Debug;

    pub fn foo() -> Foo {
        22_u32
    }
}

fn is_send<T: Send>(_: T) {}

fn main() {
    is_send(m::foo());
}
