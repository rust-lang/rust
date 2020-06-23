// check-pass

#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

struct A<'a>(&'a ());

trait Trait<T> {}

impl<T> Trait<T> for () {}

pub fn foo<'a>() {
    let _x: impl Trait<A<'a>> = ();
}

fn main() {}
