#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// FIXME This should compile, but it currently doesn't

mod m {
    pub type Foo = impl std::fmt::Debug;
    //~^ ERROR: cycle detected when computing type of `m::Foo::{opaque#0}` [E0391]
    //~| ERROR: cycle detected when computing type of `m::Foo::{opaque#0}` [E0391]

    pub fn foo() -> Foo {
        22_u32
    }

    pub fn bar() {
        is_send(foo());
        //~^ ERROR: cannot check whether the hidden type of `auto_trait_leakage3[211d]::m::Foo::{opaque#0}
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
