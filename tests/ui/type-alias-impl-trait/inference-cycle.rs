#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    type Foo = impl std::fmt::Debug;
    //~^ ERROR cycle detected

    #[defines(Foo)]
    pub fn foo() -> Foo {
        is_send(bar())
    }

    // Cycle: this function is not defining, but marked as such.
    // trying to prove `Foo: Send` within it will reveal the hidden type
    // to check `u32: Send`, but that revealing will look at all functions
    // with a `defines(Foo)` attribute, including this one, causing the cycle.
    #[defines(Foo)]
    pub fn bar() {
        is_send(foo()); // Today: error
    }

    #[defines(Foo)]
    fn baz() -> Foo {
        let f: Foo = 22_u32;
        f
    }

    #[defines(Foo)]
    fn bak() {
        let f: Foo = 22_u32;
    }

    fn is_send<T: Send>(_: T) {}
}

fn main() {}
