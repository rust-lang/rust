#![feature(const_trait_impl)]
//! Test that we can actually use `[const] Trait` bounds written on the impl block

//@ check-pass

struct Foo<T>(T);

const trait Trait {
    fn method() {}
}

const impl Trait for () {}

const impl<T: [const] Trait> Foo<T> {
    fn bar() {
        T::method();
    }
}

const _: () = Foo::<()>::bar();

fn main() {
    Foo::<()>::bar();
}
