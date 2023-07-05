#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    pub type Foo = impl std::fmt::Debug;
    //~^ ERROR cycle detected
    //~| ERROR cycle detected

    // Cycle: error today, but it'd be nice if it eventually worked

    pub fn foo() -> Foo {
        is_send(bar())
    }

    pub fn bar() {
        is_send(foo()); // Today: error
        //~^ ERROR: cannot check whether the hidden type of `inference_cycle[4ecc]::m::Foo::{opaque#0}` satisfies auto traits
    }

    fn baz() {
        let f: Foo = ();
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
