#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait<const N: usize> {}

fn foo<'a, T>()
where
    T: Trait<const { let a: &'a (); 1 }>
    //~^ ERROR cannot capture late-bound lifetime in constant
{}

fn main() {}
