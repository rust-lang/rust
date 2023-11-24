#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    pub type Foo = impl std::fmt::Debug;

    pub fn foo() -> Foo {
        is_send(bar())
    }

    pub fn bar() {
        // Cycle: error today, but it'd be nice if it eventually worked
        is_send(foo());
        //~^ ERROR: cannot check whether the hidden type of `inference_cycle[4ecc]::m::Foo::{opaque#0}` satisfies auto traits
    }

    fn baz() -> Foo {
        ()
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
