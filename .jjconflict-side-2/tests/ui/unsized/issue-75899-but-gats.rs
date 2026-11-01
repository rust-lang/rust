//@ check-pass

use std::fmt::Debug;
use std::marker::PhantomData;

trait Foo {
    type Gat<'a>: ?Sized where Self: 'a;
}

struct Bar<'a, T: Foo + 'a>(T::Gat<'a>);

struct Baz<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Foo for Baz<T> {
    type Gat<'a> = T where Self: 'a;
}

fn main() {
    let x = Bar::<'_, Baz<()>>(());
    let y: &Bar<'_, Baz<dyn Debug>> = &x;
}
