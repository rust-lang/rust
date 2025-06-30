//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

struct S;

#[const_trait]
trait Foo {
    fn eq(&self, _: &Self) -> bool;
}

impl Foo for S {
    fn eq(&self, _: &S) -> bool {
        true
    }
}

const fn equals_self<T: [const] Foo>(t: &T) -> bool {
    true
}

// Calling `equals_self` with something that has a non-const impl should throw an error, despite
// it not using the impl.

pub const EQ: bool = equals_self(&S);
//~^ ERROR: the trait bound `S: const Foo` is not satisfied

fn main() {}
