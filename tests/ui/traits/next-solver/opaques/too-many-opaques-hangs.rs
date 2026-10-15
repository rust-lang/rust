//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/151806>.
// This used to hang due to excessive opaques being registered endlessly.

//~^^^^^ ERROR: overflow evaluating the requirement `<I as UnwrapItemsExt>::Iter == {type error}`

#![feature(impl_trait_in_assoc_type)]
trait UnwrapItemsExt {
    type Iter;
}

trait MyTrait<'a> {}

impl<'a> MyTrait<'a> for <() as UnwrapItemsExt>::Iter {}

impl<'a, I> UnwrapItemsExt for I {
    //~^ ERROR: the lifetime parameter `'a` is not constrained
    type Iter = impl MyTrait<'a>;
    fn unwrap_items() -> Self::Iter {}
    //~^ ERROR: overflow evaluating the requirement `<I as UnwrapItemsExt>::Iter == ()`
    //~| ERROR: method `unwrap_items` is not a member of trait `UnwrapItemsExt`
}

fn main() {}
