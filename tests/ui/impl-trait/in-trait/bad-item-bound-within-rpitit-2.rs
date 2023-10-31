// issue: 114146

#![feature(return_position_impl_trait_in_trait)]

trait Foo {
    fn bar<'other: 'a>() -> impl Sized + 'a {}
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR use of undeclared lifetime name `'a`
}

fn main() {}
