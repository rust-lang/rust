//@ check-pass
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

use std::fmt::Debug;
use std::marker::PhantomData;

trait Foo {
    type Gat<'a> where Self: 'a;
}

struct Bar<'a, T: Foo + 'a>(T::Gat<'a>);

struct Baz<T>(PhantomData<T>);

impl<T> Foo for Baz<T> {
    type Gat<'a> = T where Self: 'a;
}

fn main() {
    let x = Bar::<'_, Baz<()>>(());
    let y: &Bar<'_, Baz<dyn Debug>> = &x;
}
