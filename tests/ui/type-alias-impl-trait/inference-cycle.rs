#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    type Foo = impl std::fmt::Debug;
    //~^ ERROR cycle detected

    // Cycle: error today, but it'd be nice if it eventually worked

    pub fn foo() -> Foo {
        is_send(bar())
    }

    pub fn bar() {
        is_send(foo()); // Today: error
    }

    fn baz() -> Foo {
        let f: Foo = 22_u32;
        f
    }

    fn bak() {
        let f: Foo = 22_u32;
        //~^ ERROR constrained without being represented in the signature
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
