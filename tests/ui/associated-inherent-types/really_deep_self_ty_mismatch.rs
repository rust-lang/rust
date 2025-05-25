//@ check-pass

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Test that IAT resolution doesn't bail out when the self type is
// very nested.

struct Foo<T>(T);
#[rustfmt::skip]
impl Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<u8>>>>>>>>>>> {
    type Inherent = u16;
}
#[rustfmt::skip]
impl Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<u16>>>>>>>>>>> {
    type Inherent = u32;
}

#[rustfmt::skip]
struct Bar {
    field: <Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<Foo<u8>>>>>>>>>>>>::Inherent,
}

fn main() {
    Bar { field: 1_u16 };
}
