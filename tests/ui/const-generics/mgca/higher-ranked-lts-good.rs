//@ check-pass

#![feature(gca_min_const_items)]
#![expect(incomplete_features)]

trait Trait<const N: usize> {}

fn foo<T>()
where
    T: Trait<const { let a: for<'a> fn(&'a ()); 1 }>
{}

fn main() {}
