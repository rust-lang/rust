#![feature(const_trait_impl)]

struct Foo<T>(T);

const trait Trait {
    fn method() {}
}

const impl Trait for () {}

const impl<T: [const] Trait> Foo<T> {
    fn bar() {
        T::method();
        //~^ ERROR: the trait bound `T: [const] Trait` is not satisfied
    }
}

const _: () = Foo::<()>::bar();

fn main() {
    Foo::<()>::bar();
}
