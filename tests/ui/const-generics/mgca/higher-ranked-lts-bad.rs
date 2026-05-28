#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait<const N: usize> {}

fn foo<T>()
where
    for<'a> T: Trait<const { let a: &'a (); 1 }>
    //~^ ERROR cannot capture late-bound lifetime in constant
{}

fn main() {}
