#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    type Foo = impl std::fmt::Debug;
    //~^ ERROR cycle detected
    //~| ERROR cycle detected

    // Cycle: error today, but it'd be nice if it eventually worked

    pub fn foo() -> Foo {
        is_send(bar())
    }

    pub fn bar() {
        is_send(foo()); // Today: error
    }

    fn baz() {
        let f: Foo = 22_u32;
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
