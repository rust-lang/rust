#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait<const N: usize> {}

fn foo<'a, T>()
where
    'a: 'a,
    T: Trait<const { let a: &'a (); 1 }>
    //~^ ERROR generic parameters may not be used in const operations
{}

fn main() {}
