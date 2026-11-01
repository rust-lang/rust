//! Regression test for https://github.com/rust-lang/rust/issues/143896.

trait TraitA<'a> {
    const K: usize = 0;
}

impl<T> TraitA<'_> for () {}
//~^ ERROR the type parameter `T` is not constrained

impl dyn TraitA<'_> where (): TraitA<'a, K = 0> {}
//~^ ERROR use of undeclared lifetime name `'a`
//~| ERROR associated const equality is incomplete

pub fn main() {}
