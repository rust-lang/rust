#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Test that when we have an unnormalized projection in the IAT self ty
// we don't normalize it to determine which IAT to resolve to.

struct Foo<T>(T);
impl Foo<u8> {
    type Inherent = u16;
}
impl Foo<u16> {
    type Inherent = u32;
}

struct Bar {
    field: <Foo<<u8 as Identity>::This>>::Inherent,
    //~^ ERROR: multiple applicable items in scope
}

trait Identity {
    type This;
}
impl<T> Identity for T { type This = T; }

fn main() {
    Bar {
        field: 1_u16,
    };
}
