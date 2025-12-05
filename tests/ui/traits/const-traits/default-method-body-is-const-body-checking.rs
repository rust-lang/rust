#![feature(const_trait_impl)]

const trait Tr {}
impl Tr for () {}

const fn foo<T>() where T: [const] Tr {}

pub const trait Foo {
    fn foo() {
        foo::<()>();
        //~^ ERROR the trait bound `(): [const] Tr` is not satisfied
    }
}

fn main() {}
