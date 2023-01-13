// check-pass

use std::ops::Deref;

trait Foo {
    type Bar<'a, 'b>;
}

trait Baz {
    type Quux<'a>: Foo where Self: 'a;

    // This weird type tests that we can use universal function call syntax to access the Item on
    type Baa<'a>: Deref<Target = <Self::Quux<'a> as Foo>::Bar<'a, 'static>>  where Self: 'a;
}

impl<T> Baz for T where T: Foo {
    type Quux<'a> = T where T: 'a;

    type Baa<'a> = &'a <T as Foo>::Bar<'a, 'static> where T: 'a;
}

fn main() {}
