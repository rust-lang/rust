//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait<const N: usize> {}

fn foo<T>()
where
    T: Trait<const { let a: for<'a> fn(&'a ()); 1 }>
{}

fn main() {}
