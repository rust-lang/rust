#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

use std::ops::Deref;

// FIXME(#44265): "lifetime arguments are not allowed for this type" errors will be addressed in a
// follow-up PR.

trait Foo {
    type Bar<'a, 'b>;
}

trait Baz {
    type Quux<'a>: Foo;

    // This weird type tests that we can use universal function call syntax to access the Item on
    type Baa<'a>: Deref<Target = <Self::Quux<'a> as Foo>::Bar<'a, 'static>>;
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
    //~| ERROR lifetime arguments are not allowed for this type [E0109]
}

impl<T> Baz for T where T: Foo {
    type Quux<'a> = T;

    type Baa<'a> = &'a <T as Foo>::Bar<'a, 'static>;
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
}

fn main() {}
